import torch
import torch.nn as nn
import torch.nn.functional as F


# Lightweight private branch used by each routed expert.
class LightMLPExpert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Augment a pretrained CLIP feed-forward block with routed private experts.
class TrueMoE_FFN(nn.Module):
    def __init__(
        self,
        orig_mlp: nn.Module,
        n_private_experts: int = 4,
        temperature: float = 1.0,
        expert_hidden_dim: int = 256,
        topk: int = 2,
        noise_std: float = 1e-2,
    ):
        super().__init__()
        # Reuse the pretrained CLIP MLP as the always-active shared expert.
        self.shared_fc1 = orig_mlp.c_fc
        self.shared_act = orig_mlp.act if hasattr(orig_mlp, "act") else nn.GELU()
        self.shared_fc2 = (
            orig_mlp.c_proj if hasattr(orig_mlp, "c_proj") else orig_mlp.c_proj
        )
        self.shared_expert = nn.Sequential(
            self.shared_fc1,
            self.shared_act,
            self.shared_fc2,
        )

        d_model = self.shared_fc1.in_features

        # Private experts learn residual task-specific transformations.
        self.private_experts = nn.ModuleList(
            [
                LightMLPExpert(d_model, expert_hidden_dim)
                for _ in range(n_private_experts)
            ]
        )

        self.gate = nn.Linear(d_model, n_private_experts)
        nn.init.xavier_uniform_(self.gate.weight, gain=1e-1)
        nn.init.zeros_(self.gate.bias)

        self.temperature = temperature
        self.topk = topk
        self.n_private_experts = n_private_experts

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.noise_std = noise_std

    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False,
    ):
        original_shape = x.shape
        # Route tokens independently while restoring sequence shape afterward.
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
        else:
            x_flat = x
            B = None
            T = None

        main_out_flat = self.shared_expert(x_flat)

        logits_private = self.gate(x_flat)
        # Noisy routing is active only during training.
        if self.training:
            noise = torch.randn_like(logits_private) * self.noise_std
            logits_private = logits_private + noise

        weights_private = F.softmax(
            logits_private / self.temperature,
            dim=-1,
        )

        # Keep the historical top-k weights without renormalizing them.
        values_topk, indices_topk = torch.topk(
            weights_private,
            k=self.topk,
            dim=-1,
        )
        weight_mask = torch.zeros_like(weights_private)
        weight_mask.scatter_(
            -1,
            indices_topk,
            values_topk,
        )

        N, D = x_flat.shape
        k = self.topk

        outs = torch.zeros(
            N,
            k,
            D,
            device=x_flat.device,
            dtype=x_flat.dtype,
        )

        indices_flat = indices_topk.reshape(-1)
        # Dispatch routes in the original flattened top-k traversal order.
        for ki in range(k):
            idx_chunk = indices_flat[ki * N : (ki + 1) * N]
            unique_experts = idx_chunk.unique()
            out_chunk = torch.zeros(
                N,
                D,
                device=x_flat.device,
                dtype=x_flat.dtype,
            )
            for e in unique_experts:
                mask_e = idx_chunk == e
                x_e = x_flat[mask_e]
                y_e = self.private_experts[e](x_e)
                y_e = y_e.to(out_chunk.dtype)
                out_chunk[mask_e] = y_e
            outs[:, ki, :] = out_chunk

        values_topk_exp = values_topk.unsqueeze(-1)
        side_flat = (outs * values_topk_exp).sum(dim=1)

        # Add the routed residual to the shared pretrained response.
        out_flat = main_out_flat + self.alpha * side_flat

        if B is not None and T is not None:
            out = out_flat.view(B, T, D)
        else:
            out = out_flat

        if return_gates:
            return (
                out,
                weight_mask.view(
                    *original_shape[:-1],
                    -1,
                ),
            )
        return out


def patch_visual_moe(
    model: nn.Module,
    n_private_experts: int = 4,
    expert_hidden_dim: int = 256,
    noise_std: float = 1e-2,
):
    vt = model.visual
    blocks = vt.transformer.resblocks
    total = len(blocks)
    # Patch only the final six visual transformer feed-forward blocks.
    for idx in range(total - 6, total):
        orig_block = blocks[idx]
        orig_mlp = orig_block.mlp
        blocks[idx].mlp = TrueMoE_FFN(
            orig_mlp,
            n_private_experts=n_private_experts,
            expert_hidden_dim=expert_hidden_dim,
            topk=2,
            noise_std=noise_std,
        )

    return vt
