"""Deterministic train-only PCA for T11P token features.

The T11P probes use the 1024-d Prithvi / 1536-d AnySat temporal-mean tokens.
This helper fits the projection on training tokens only and applies it to the
validation/test splits, with the same sign convention as the Clay T11D probe:
the largest-magnitude loading of every component is made positive.
"""

from __future__ import annotations

import numpy as np


def fit_token_pca(features: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA on ``(N, D, H, W)`` features and return mean, components, variance."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"expected (N, D, H, W), got shape {array.shape}")
    count, dims, height, width = array.shape
    if components < 1 or components > dims:
        raise ValueError(f"components must be in 1..{dims}, got {components}")
    matrix = array.transpose(0, 2, 3, 1).reshape(-1, dims).astype(np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    basis = eigenvectors[:, order].T
    for row in range(basis.shape[0]):
        pivot = int(np.argmax(np.abs(basis[row])))
        if basis[row, pivot] < 0:
            basis[row] *= -1
    explained = float(eigenvalues[order].sum() / eigenvalues.sum())
    return mean.astype(np.float32), basis.astype(np.float32), explained


def project_token_pca(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project ``(N, D, H, W)`` features onto the fitted PCA basis."""
    array = np.asarray(features, dtype=np.float32)
    count, dims, height, width = array.shape
    if mean.shape != (dims,) or components.shape[1] != dims:
        raise ValueError(
            f"PCA basis does not match features: mean {mean.shape}, "
            f"components {components.shape}, features {array.shape}"
        )
    matrix = array.transpose(0, 2, 3, 1).reshape(-1, dims)
    projected = (matrix - mean) @ components.T
    return (
        projected.reshape(count, height, width, components.shape[0])
        .transpose(0, 3, 1, 2)
        .astype(np.float32)
    )
