"""Scoring functions for probabilistic classifiers.

Domain-agnostic math. Projects apply these to their own data structures
and define their own result types.
"""

from __future__ import annotations

import numpy as np


def brier_score(y_true, y_prob) -> float:
    """Mean Brier score. Lower is better; 0.25 = random binary classifier."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true, y_prob, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss. Lower is better; 0.693 = random binary classifier."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
