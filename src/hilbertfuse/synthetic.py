"""Synthetic expression data for demos and tests."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic(n_samples=600, n_genes=1500, n_classes=5, n_informative=40, noise=1.0, seed=0):
    """Gene-expression-like table: informative genes are shifted per class.

    Returns a DataFrame (samples x genes + 'label'). Gene names are G0000.. and the first
    `n_informative` genes (split across classes) carry the signal.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_samples)
    X = rng.gamma(2.0, 1.0, size=(n_samples, n_genes)).astype(np.float32)
    per = max(1, n_informative // n_classes)
    for c in range(n_classes):
        idx = np.arange(c * per, (c + 1) * per)
        X[np.ix_(y == c, idx)] += 3.0 / noise
    df = pd.DataFrame(X, columns=[f"G{i:04d}" for i in range(n_genes)])
    df["label"] = [f"T{c}" for c in y]
    return df
