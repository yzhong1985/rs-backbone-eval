"""CPU-only tests for the train-only PCA used by the T11P PCA128 probes."""

from __future__ import annotations

import numpy as np
import pytest

from rs_backbone_eval.pca import fit_token_pca, project_token_pca


def _features(seed: int = 0, count: int = 20, dims: int = 8, size: int = 3) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(count, dims, size, size)).astype(np.float32)


def test_pca_projection_shape_and_mean_centring() -> None:
    train = _features()
    mean, components, explained = fit_token_pca(train, 3)

    assert mean.shape == (8,)
    assert components.shape == (3, 8)
    assert 0.0 < explained <= 1.0
    projected = project_token_pca(train, mean, components)
    assert projected.shape == (20, 3, 3, 3)
    # The fitted projection is centred on the train tokens.
    assert np.allclose(projected.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)


def test_pca_is_fitted_on_train_only() -> None:
    train = _features(seed=0)
    other = _features(seed=1)
    mean, components, _ = fit_token_pca(train, 4)

    first = project_token_pca(other, mean, components)
    second = project_token_pca(other, mean, components)
    assert np.allclose(first, second)
    # Fitting on the other split would give a different basis.
    other_mean, other_components, _ = fit_token_pca(other, 4)
    assert not np.allclose(mean, other_mean)


def test_pca_full_rank_reconstructs_with_unit_explained_variance() -> None:
    train = _features(count=6, dims=5, size=2)
    mean, components, explained = fit_token_pca(train, 5)
    projected = project_token_pca(train, mean, components)

    reconstructed = (
        (projected.transpose(0, 2, 3, 1).reshape(-1, 5) @ components + mean)
        .reshape(train.shape[0], 2, 2, 5)
        .transpose(0, 3, 1, 2)
    )
    assert explained == pytest.approx(1.0)
    assert np.allclose(reconstructed, train, atol=1e-4)


def test_pca_sign_convention_is_deterministic() -> None:
    train = _features(seed=3)
    _, components, _ = fit_token_pca(train, 6)

    for row in components:
        assert row[int(np.argmax(np.abs(row)))] > 0


def test_pca_rejects_bad_component_count() -> None:
    train = _features()
    with pytest.raises(ValueError):
        fit_token_pca(train, 0)
    with pytest.raises(ValueError):
        fit_token_pca(train, 9)
