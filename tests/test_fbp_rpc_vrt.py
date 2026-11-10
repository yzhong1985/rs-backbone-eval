"""Unit tests for the GF-2 .rpb -> RPC VRT converter (no GDAL, no data needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fbp_rpc_vrt.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fbp_rpc_vrt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RPB = """satId = "XXX";
bandId = "XXX";
SpecId = "XXX";
BEGIN_GROUP = IMAGE
\terrBias =   1.0;
\terrRand =    0.0;
\tlineOffset = 3454.0;
\tsampOffset = 3650.0;
\tlatOffset =   28.5;
\tlongOffset = 115.8;
\theightOffset = 42.5;
\tlineScale =   3454.0;
\tsampScale =   3650.0;
\tlatScale =   0.15;
\tlongScale =   0.29;
\theightScale =  500.0;
\tlineNumCoef = ( %s );
\tlineDenCoef = ( %s );
\tsampNumCoef = ( %s );
\tsampDenCoef = ( %s );
END_GROUP = IMAGE
END;
"""


def rpb_text() -> str:
    ones = ", ".join(["1.0"] * 20)
    return RPB % (ones, ones, ones, ones)


def test_parse_scalars_and_coefficients() -> None:
    module = load_module()
    values = module.parse_rpb(rpb_text())

    assert values["heightOffset"] == [42.5]
    assert values["latOffset"] == [28.5]
    assert len(values["lineNumCoef"]) == 20
    assert len(values["sampDenCoef"]) == 20


def test_rpc_metadata_has_all_sixteen_tags() -> None:
    module = load_module()
    meta = module.rpc_metadata(module.parse_rpb(rpb_text()))

    assert len(meta) == 16
    assert meta["HEIGHT_OFF"] == f"{42.5:.10f}"
    assert meta["LINE_OFF"] == f"{3454.0:.10f}"
    assert len(meta["SAMP_NUM_COEFF"].split()) == 20


def test_rpc_metadata_rejects_truncated_coefficients() -> None:
    module = load_module()
    text = rpb_text().replace(", ".join(["1.0"] * 20), ", ".join(["1.0"] * 19), 1)

    with pytest.raises(ValueError, match="expected 20"):
        module.rpc_metadata(module.parse_rpb(text))


def test_build_vrt_carries_rpc_domain_and_bands() -> None:
    module = load_module()
    meta = module.rpc_metadata(module.parse_rpb(rpb_text()))
    vrt = module.build_vrt(meta, Path("/data/scene.tiff"), 7300, 6908, 4, "UInt16")

    assert 'rasterXSize="7300"' in vrt and 'rasterYSize="6908"' in vrt
    assert 'domain="RPC"' in vrt
    assert vrt.count("<MDI key=") == 16
    assert vrt.count("<VRTRasterBand") == 4
    assert "/data/scene.tiff" in vrt
