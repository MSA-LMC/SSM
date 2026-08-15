"""Segment-safe joint search for AU macro-F1.

The evaluator searches one *global* probability threshold and one global
``(K, p)`` voting rule.  It deliberately returns a single F1 value so that
checkpoint selection and reporting cannot accidentally mix two protocols.

For a window of width ``W = 2*K + 1``, the legacy rule

    count >= p * W

depends on ``p`` only through ``ceil(p * W)``.  Equivalent p candidates are
therefore collapsed before scoring.  Threshold search is accelerated by
turning every probability into the number of threshold candidates it passes;
the temporal vote then becomes an order-statistic filter over these integer
ranks.  This is exactly equivalent to thresholding first and voting second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


_EPSILON = 1.0e-12


@dataclass(frozen=True)
class _VoteRule:
    K: int
    p: float
    required_votes: int


def _as_sorted_unique_floats(
    values: Iterable[float],
    *,
    name: str,
    lower_exclusive: float,
    upper_inclusive: float,
) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional list.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    if np.any(array <= lower_exclusive) or np.any(array > upper_inclusive):
        raise ValueError(
            f"{name} values must be in ({lower_exclusive}, {upper_inclusive}]."
        )
    return np.unique(array)


def _validate_segments(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    expected_au_count: int | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    if not segments:
        raise ValueError("At least one non-empty segment is required.")

    checked: list[tuple[np.ndarray, np.ndarray]] = []
    au_count: int | None = None
    for index, pair in enumerate(segments):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"segments[{index}] must be a (targets, probabilities) pair."
            )
        target = np.asarray(pair[0])
        probability = np.asarray(pair[1])
        if target.ndim != 2 or probability.ndim != 2:
            raise ValueError(f"segments[{index}] arrays must both have shape [T, A].")
        if target.shape != probability.shape:
            raise ValueError(
                f"segments[{index}] shape mismatch: "
                f"targets={target.shape}, probabilities={probability.shape}."
            )
        if target.shape[0] == 0 or target.shape[1] == 0:
            raise ValueError(f"segments[{index}] must not be empty.")
        if au_count is None:
            au_count = int(target.shape[1])
        elif target.shape[1] != au_count:
            raise ValueError("All segments must have the same AU count.")
        if not np.all(np.isfinite(probability)):
            raise ValueError(
                f"segments[{index}] probabilities contain non-finite values."
            )
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError(f"segments[{index}] probabilities must lie in [0, 1].")
        if not np.all((target == 0) | (target == 1)):
            raise ValueError(
                f"segments[{index}] targets must contain only zero and one."
            )
        checked.append(
            (
                target.astype(np.uint8, copy=False),
                probability.astype(np.float64, copy=False),
            )
        )

    assert au_count is not None
    if expected_au_count is not None and au_count != int(expected_au_count):
        raise ValueError(f"Expected {expected_au_count} AUs, received {au_count}.")
    return checked, au_count


def _required_votes(K: int, p: float) -> int:
    width = 2 * K + 1
    # The tiny subtraction protects exact integer products such as p=1.0 from
    # representation noise while preserving the mathematical ceil(p * W).
    return int(np.ceil(float(p) * width - _EPSILON))


def _collapsed_vote_rules(
    K_values: Iterable[int],
    p_values: Iterable[float],
    *,
    reference_p: float,
) -> tuple[list[_VoteRule], int]:
    raw_K = tuple(K_values)
    if not raw_K:
        raise ValueError("K_values must be non-empty.")
    if any(isinstance(value, bool) or int(value) != value for value in raw_K):
        raise ValueError("K_values must contain integers.")
    K_candidates = sorted(set(int(value) for value in raw_K))
    if K_candidates[0] < 0:
        raise ValueError("K_values must be non-negative.")
    p_candidates = _as_sorted_unique_floats(
        p_values,
        name="p_values",
        lower_exclusive=0.0,
        upper_inclusive=1.0,
    )

    rules: list[_VoteRule] = []
    raw_count = len(K_candidates) * len(p_candidates)
    for K in K_candidates:
        equivalents: dict[int, list[float]] = {}
        for p in p_candidates:
            required = _required_votes(K, float(p))
            equivalents.setdefault(required, []).append(float(p))
        for required, equivalent_ps in sorted(equivalents.items()):
            # Keep a user-supplied p value in logs/configs.  If several p values
            # implement the same integer rule, prefer the legacy-nearest one.
            representative = min(
                equivalent_ps,
                key=lambda value: (abs(value - reference_p), value),
            )
            rules.append(
                _VoteRule(
                    K=K,
                    p=representative,
                    required_votes=required,
                )
            )
    return rules, raw_count


def _f1_grid_from_rank(
    prediction_rank: np.ndarray,
    targets: np.ndarray,
    threshold_count: int,
) -> np.ndarray:
    """Return [threshold, AU] F1 without materializing [threshold, frame, AU]."""
    au_count = targets.shape[1]
    output = np.empty((threshold_count, au_count), dtype=np.float64)
    for au_index in range(au_count):
        ranks = prediction_rank[:, au_index]
        positives = targets[:, au_index].astype(bool, copy=False)
        positive_histogram = np.bincount(
            ranks[positives], minlength=threshold_count + 1
        )
        negative_histogram = np.bincount(
            ranks[~positives], minlength=threshold_count + 1
        )

        # rank r means thresholds with indices 0..r-1 are predicted positive.
        # Thus, at threshold index j, sum bins r >= j+1.
        true_positive = np.cumsum(positive_histogram[::-1])[::-1][1:]
        false_positive = np.cumsum(negative_histogram[::-1])[::-1][1:]
        total_positive = int(positives.sum())
        false_negative = total_positive - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        output[:, au_index] = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros(threshold_count, dtype=np.float64),
            where=denominator != 0,
        )
    return output


def _tie_key(
    *,
    K: int,
    p: float,
    threshold: float,
    reference: tuple[int, float, float],
) -> tuple[float, float, float, int, float, float]:
    """Prefer the smallest departure from the legacy protocol on exact ties."""
    reference_K, reference_p, reference_threshold = reference
    return (
        abs(K - reference_K),
        abs(p - reference_p),
        abs(threshold - reference_threshold),
        K,
        p,
        threshold,
    )


def search_segment_safe_vote(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    thresholds: Iterable[float] = tuple(index / 100.0 for index in range(1, 100)),
    K_values: Iterable[int] = tuple(range(0, 8)),
    p_values: Iterable[float] = tuple(index / 10.0 for index in range(1, 10)),
    expected_au_count: int | None = None,
    tie_reference: tuple[int, float, float] = (3, 0.3, 0.5),
) -> dict:
    """Jointly select threshold, K and p by maximum AU-macro F1.

    ``segments`` must already represent true continuous sequences.  Edge
    padding and voting are performed independently for every item, including
    when a segment is shorter than the requested voting window.

    The defaults search thresholds 0.01..0.99, K 0..7 and p 0.1..0.9.
    All three choices are global across the configured AUs.
    """
    checked, au_count = _validate_segments(
        segments, expected_au_count=expected_au_count
    )
    threshold_values = _as_sorted_unique_floats(
        thresholds,
        name="thresholds",
        lower_exclusive=0.0,
        upper_inclusive=1.0,
    )
    if len(tie_reference) != 3:
        raise ValueError("tie_reference must be (K, p, threshold).")
    reference_K = int(tie_reference[0])
    reference_p = float(tie_reference[1])
    reference_threshold = float(tie_reference[2])
    if reference_K < 0 or not (0.0 < reference_p <= 1.0):
        raise ValueError("tie_reference contains an invalid K or p.")
    if not (0.0 < reference_threshold <= 1.0):
        raise ValueError("tie_reference threshold must be in (0, 1].")
    reference = (reference_K, reference_p, reference_threshold)

    rules, raw_rule_count = _collapsed_vote_rules(
        K_values, p_values, reference_p=reference_p
    )
    targets = np.concatenate([target for target, _ in checked], axis=0)
    # rank is the number of searched thresholds that the probability passes.
    segment_ranks = [
        np.searchsorted(threshold_values, probability, side="right").astype(
            np.int16, copy=False
        )
        for _, probability in checked
    ]

    rules_by_K: dict[int, list[_VoteRule]] = {}
    for rule in rules:
        rules_by_K.setdefault(rule.K, []).append(rule)

    best_score = -1.0
    best_tie: tuple[float, ...] | None = None
    best_rule: _VoteRule | None = None
    best_threshold_index = -1
    best_per_au: np.ndarray | None = None

    for K, K_rules in sorted(rules_by_K.items()):
        width = 2 * K + 1
        sorted_windows = []
        for rank in segment_ranks:
            if K:
                padded = np.pad(rank, ((K, K), (0, 0)), mode="edge")
            else:
                padded = rank
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=width, axis=0
            )
            # Shape is [segment_length, AU, window].  One sort yields every
            # attainable p/required-vote rule for this K.
            sorted_windows.append(np.sort(windows, axis=-1))
        ordered_rank = np.concatenate(sorted_windows, axis=0)

        for rule in K_rules:
            # count(binary >= threshold) >= m iff the m-th largest probability
            # (or threshold rank) in the window passes that threshold.
            effective_rank = ordered_rank[..., width - rule.required_votes]
            per_au_grid = _f1_grid_from_rank(
                effective_rank, targets, len(threshold_values)
            )
            macro_grid = per_au_grid.mean(axis=1)
            rule_best_score = float(np.max(macro_grid))
            tied_thresholds = np.flatnonzero(
                np.isclose(
                    macro_grid,
                    rule_best_score,
                    rtol=0.0,
                    atol=_EPSILON,
                )
            )
            threshold_index = min(
                tied_thresholds.tolist(),
                key=lambda index: (
                    abs(float(threshold_values[index]) - reference_threshold),
                    float(threshold_values[index]),
                ),
            )
            threshold = float(threshold_values[threshold_index])
            candidate_tie = _tie_key(
                K=rule.K,
                p=rule.p,
                threshold=threshold,
                reference=reference,
            )
            is_better = rule_best_score > best_score + _EPSILON
            is_equal = abs(rule_best_score - best_score) <= _EPSILON
            if is_better or (
                is_equal and (best_tie is None or candidate_tie < best_tie)
            ):
                best_score = rule_best_score
                best_tie = candidate_tie
                best_rule = rule
                best_threshold_index = threshold_index
                best_per_au = per_au_grid[threshold_index].copy()

    assert best_rule is not None and best_per_au is not None
    width = 2 * best_rule.K + 1
    lower_open = (best_rule.required_votes - 1) / width
    upper_closed = best_rule.required_votes / width
    return {
        "protocol": (
            "single_global_threshold_plus_adaptive_segment_safe_majority_voting"
        ),
        "f1": float(best_score),
        "per_au_f1": best_per_au.tolist(),
        "threshold": float(threshold_values[best_threshold_index]),
        "K": int(best_rule.K),
        "p": float(best_rule.p),
        "required_votes": int(best_rule.required_votes),
        "window_size": int(width),
        "p_equivalence_interval": {
            "lower_open": float(lower_open),
            "upper_closed": float(upper_closed),
        },
        "evaluated_frames": int(targets.shape[0]),
        "au_count": int(au_count),
        "threshold_candidates": int(len(threshold_values)),
        "raw_K_p_candidates": int(raw_rule_count),
        "unique_vote_rules": int(len(rules)),
        "tie_break": (
            "max_macro_f1_then_nearest_(K,p,threshold)_to_"
            f"({reference_K},{reference_p:g},{reference_threshold:g})"
        ),
    }


def apply_segment_safe_vote(
    probabilities: np.ndarray,
    *,
    threshold: float,
    K: int,
    p: float | None = None,
    required_votes: int | None = None,
) -> np.ndarray:
    """Apply one selected rule to a single continuous segment.

    Passing ``required_votes`` is preferred because it reproduces the selected
    discrete rule without floating-point ambiguity.  ``p`` remains supported
    for compatibility with the legacy configuration.
    """
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("probabilities must have non-empty shape [T, A].")
    if isinstance(K, bool) or int(K) != K or int(K) < 0:
        raise ValueError("K must be a non-negative integer.")
    K = int(K)
    width = 2 * K + 1
    if required_votes is None:
        if p is None or not (0.0 < float(p) <= 1.0):
            raise ValueError("Provide p in (0, 1] or required_votes.")
        required_votes = _required_votes(K, float(p))
    if (
        isinstance(required_votes, bool)
        or int(required_votes) != required_votes
        or not (1 <= int(required_votes) <= width)
    ):
        raise ValueError(f"required_votes must be an integer in [1, {width}].")
    binary = values >= float(threshold)
    padded = np.pad(binary, ((K, K), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=width, axis=0
    )
    counts = windows.sum(axis=-1)
    return (counts >= int(required_votes)).astype(np.uint8)
