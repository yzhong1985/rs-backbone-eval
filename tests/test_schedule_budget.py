"""resolve_budget: the protocol's declared budget reaches the worker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rs_backbone_eval.schedule import resolve_budget


def proto(**kw):
    base = {"protocol_id": "p", "epochs": None, "learning_rate": None, "lr_schedule": None}
    return SimpleNamespace(**{**base, **kw})


def test_protocol_declaration_wins_over_defaults() -> None:
    spec = proto(epochs=20, learning_rate=3e-3, lr_schedule="warmup1-cosine")
    assert resolve_budget(spec, official_epochs=8, default_lr=1e-3) == (20, 3e-3, "warmup1-cosine")


def test_env_override_only_when_the_protocol_is_silent() -> None:
    assert resolve_budget(proto(), official_epochs=8, default_lr=1e-3, epochs_override="2") == (
        2,
        1e-3,
        None,
    )
    with pytest.raises(ValueError, match="declares epochs=20"):
        resolve_budget(proto(epochs=20), official_epochs=8, default_lr=1e-3, epochs_override="8")
    with pytest.raises(ValueError, match="declares learning_rate"):
        resolve_budget(
            proto(learning_rate=3e-3), official_epochs=8, default_lr=1e-3, lr_override="1e-3"
        )
    # a matching override is harmless
    assert (
        resolve_budget(proto(epochs=8), official_epochs=8, default_lr=1e-3, epochs_override="8")[0]
        == 8
    )


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        resolve_budget(proto(lr_schedule="step"), official_epochs=8, default_lr=1e-3)
