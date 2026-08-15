import pytest
import torch

from ssm.models.temporal import Temporal_Transformer_All


def _build_temporal():
    torch.manual_seed(17)
    return Temporal_Transformer_All(
        num_patches=6,
        input_dim=8,
        depth=1,
        heads=2,
        mlp_dim=16,
        dim_head=4,
        local_radius=1,
        max_update=0.25,
    )


def test_au_temporal_is_an_exact_identity_at_initialization():
    module = _build_temporal()
    inputs = torch.randn(2, 6, 8)

    module.train()
    train_output = module(inputs)
    module.eval()
    eval_output = module(inputs)

    assert train_output.shape == inputs.shape
    assert torch.equal(train_output, inputs)
    assert torch.equal(eval_output, inputs)
    assert torch.isfinite(train_output).all()
    assert not hasattr(module, "pos_embedding")
    assert torch.equal(
        module.layers[0].to_q.weight,
        module.layers[0].to_k.weight,
    )


@pytest.mark.parametrize(
    "invalid_input, message",
    [
        (torch.randn(6, 8), r"shape \[B, T, D\]"),
        (torch.randn(2, 5, 8), "length mismatch"),
        (torch.randn(2, 6, 7), "feature mismatch"),
    ],
)
def test_au_temporal_rejects_shape_mismatches(invalid_input, message):
    module = _build_temporal()
    with pytest.raises(ValueError, match=message):
        module(invalid_input)


def test_au_temporal_cannot_mix_frames_outside_its_local_radius():
    module = _build_temporal().eval()
    with torch.no_grad():
        module.layers[0].channel_gate.fill_(1.0)

    inputs = torch.randn(1, 6, 8)
    baseline = module(inputs)

    far_perturbation = inputs.clone()
    far_perturbation[:, 2] += 25.0
    far_output = module(far_perturbation)
    assert torch.equal(baseline[:, 0], far_output[:, 0])

    neighbor_perturbation = inputs.clone()
    neighbor_perturbation[:, 1] += 25.0
    neighbor_output = module(neighbor_perturbation)
    assert not torch.allclose(baseline[:, 0], neighbor_output[:, 0])


def test_au_temporal_gate_learns_before_the_attention_branch():
    module = _build_temporal()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    inputs = torch.randn(2, 6, 8)
    targets = torch.randn(2, 6, 8)

    first_loss = torch.nn.functional.mse_loss(module(inputs), targets)
    first_loss.backward()
    layer = module.layers[0]
    assert layer.channel_gate.grad is not None
    assert torch.isfinite(layer.channel_gate.grad).all()
    assert layer.channel_gate.grad.abs().sum() > 0
    assert layer.to_q.weight.grad is not None
    assert torch.count_nonzero(layer.to_q.weight.grad) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.count_nonzero(layer.channel_gate) > 0

    second_loss = torch.nn.functional.mse_loss(module(inputs), targets)
    second_loss.backward()
    assert layer.to_q.weight.grad is not None
    assert torch.isfinite(layer.to_q.weight.grad).all()
    assert layer.to_q.weight.grad.abs().sum() > 0
    assert layer.to_k.weight.grad is not None
    assert torch.isfinite(layer.to_k.weight.grad).all()
    assert layer.to_k.weight.grad.abs().sum() > 0


def test_au_temporal_state_round_trip_preserves_a_learned_update():
    module = _build_temporal().eval()
    with torch.no_grad():
        module.layers[0].channel_gate.copy_(torch.linspace(-0.4, 0.4, 8))
    inputs = torch.randn(2, 6, 8)
    expected = module(inputs)

    restored = _build_temporal().eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    actual = restored(inputs)

    assert torch.equal(actual, expected)
