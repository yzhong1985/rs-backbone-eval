import numpy as np
import pytest
from pydantic import ValidationError

from rs_backbone_eval.normalization import (
    DATASET_STATS_SEED,
    IMAGENET_RGB_ID,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    imagenet_rgb,
    normalization_seed,
)
from rs_backbone_eval.paths import RuntimePaths
from rs_backbone_eval.registry import Registry


def test_imagenet_constants_match_the_pretraining_values() -> None:
    # upstream/selfsup/main_pretrain_multi.py: mean [0.485, 0.456, 0.406],
    # std [0.229, 0.224, 0.225] on 0-1 inputs.
    assert np.allclose(IMAGENET_RGB_MEAN, [0.485 * 255, 0.456 * 255, 0.406 * 255])
    assert np.allclose(IMAGENET_RGB_STD, [0.229 * 255, 0.224 * 255, 0.225 * 255])
    mean, std = imagenet_rgb()
    assert mean.shape == (3,) and std.shape == (3,)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert IMAGENET_RGB_ID == "imagenet_rgb_0_255"


def test_dataset_statistics_use_a_fixed_seed() -> None:
    # The sampling seed must not follow request.seed: that made each seed of a
    # cell see a different input distribution (review 2026-11-03).
    assert normalization_seed() == DATASET_STATS_SEED == 0


def test_lim_declares_imagenet_normalization() -> None:
    registry = Registry(RuntimePaths.from_environment().config_root)
    models = {model.model_id: model for model in registry.models()}
    assert models["lim_mit_huge"].normalization == "imagenet"
    # Other workers keep their own preprocessing; the field is opt-in.
    assert models["prithvi_eo_v2_300_tl"].normalization is None
    assert models["clay_v1_5"].normalization is None


def test_unknown_normalization_is_rejected(tmp_path) -> None:
    from rs_backbone_eval.registry import ModelSpec, load_yaml_spec

    path = tmp_path / "model.yaml"
    path.write_text(
        "model_id: m\ndisplay_name: M\nworker_id: w\ncheckpoint_env: C\n"
        "upstream_url: u\nnormalization: imagenet_mean\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_yaml_spec(path, ModelSpec)
