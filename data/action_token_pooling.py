"""Helpers for selecting/aggregating VLA action-token hidden states."""

from __future__ import annotations

from typing import Literal

import numpy as np

TokenSelection = Literal["mean", "first", "last", "first+last"]
TOKEN_CHOICES: tuple[str, ...] = ("mean", "first", "last", "first+last")

_TOKEN_ALIASES = {
    "first_last": "first+last",
    "first-last": "first+last",
    "firstlast": "first+last",
}

_TOKEN_DESCRIPTIONS = {
    "mean": "mean over action tokens",
    "first": "first action token",
    "last": "last action token",
    "first+last": "first + last action tokens",
}


def normalize_tokens(value: str) -> TokenSelection:
    """Normalize CLI token-pooling aliases to the canonical value."""

    normalized = value.strip().lower()
    normalized = _TOKEN_ALIASES.get(normalized, normalized)
    if normalized not in TOKEN_CHOICES:
        raise ValueError(
            f"unsupported --tokens={value!r}; choose one of {', '.join(TOKEN_CHOICES)}"
        )
    return normalized  # type: ignore[return-value]


def token_pool_description(tokens: str) -> str:
    return _TOKEN_DESCRIPTIONS[normalize_tokens(tokens)]


def token_pool_tag(tokens: str) -> str:
    return normalize_tokens(tokens).replace("+", "_")


def pool_action_tokens(
    hidden_states: np.ndarray,
    n_layers: int,
    hidden_dim: int,
    tokens: str,
) -> np.ndarray:
    """Return per-layer features after selecting/pooling action-token states.

    Parameters
    ----------
    hidden_states:
        Array shaped ``(steps, action_tokens, n_layers * hidden_dim)``.
    n_layers:
        Number of saved VLA layers in the final dimension.
    hidden_dim:
        Hidden dimension per saved layer.
    tokens:
        One of ``mean``, ``first``, ``last``, or ``first+last``.

    Returns
    -------
    np.ndarray
        Array shaped ``(steps, n_layers, feature_dim)``. ``feature_dim`` is
        ``hidden_dim`` except for ``first+last``, where it is ``2 * hidden_dim``.
    """

    tokens = normalize_tokens(tokens)
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be 3-D (steps, action_tokens, width), got {hidden_states.shape}"
        )

    steps, action_tokens, width = hidden_states.shape
    expected_width = n_layers * hidden_dim
    if width != expected_width:
        raise ValueError(
            f"hidden width mismatch: got {width}, expected {n_layers} layers * "
            f"{hidden_dim} hidden = {expected_width}"
        )
    if action_tokens < 1:
        raise ValueError("hidden_states must contain at least one action token")

    per_layer = hidden_states.reshape(steps, action_tokens, n_layers, hidden_dim)
    if tokens == "mean":
        return per_layer.mean(axis=1)
    if tokens == "first":
        return per_layer[:, 0, :, :]
    if tokens == "last":
        return per_layer[:, -1, :, :]
    if tokens == "first+last":
        return np.concatenate([per_layer[:, 0, :, :], per_layer[:, -1, :, :]], axis=-1)

    raise AssertionError(f"unreachable token selection: {tokens}")
