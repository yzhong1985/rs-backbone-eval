"""Run the bootstrap offline with real manifests and a fake network command."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def prepare(tmp_path, task, rows, *, fail=False):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts/colab_bootstrap.sh", scripts)
    data = tmp_path / "data with spaces"
    manifest = data / "manifests" / task
    manifest.mkdir(parents=True)
    (manifest / "fixture.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / ".env").write_text(f"RS_EVAL_DATA_ROOT='{data}'\nRS_COLAB_PARALLEL=1\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wget = bin_dir / "wget"
    wget.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "dest = pathlib.Path(sys.argv[sys.argv.index('-O') + 1])\n"
        "assert dest.parent.is_dir(), dest\n"
        + ("dest.write_bytes(b'partial'); sys.exit(8)\n" if fail else "dest.write_bytes(b'tiff')\n")
    )
    wget.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return data, lambda: subprocess.run(
        ["bash", str(scripts / "colab_bootstrap.sh"), task],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_t1_downloads_to_manifest_paths_and_can_be_repeated(tmp_path):
    image = "raw/sen1floods11/v1.1/data/HandLabeled/S2Hand/chip.tif"
    label = image.replace("S2Hand", "LabelHand")
    data, run = prepare(tmp_path, "T1", [{"image_paths": [image], "label_path": label}])
    first = run()
    assert first.returncode == 0, first.stderr
    assert "BOOTSTRAP-OK" in first.stdout
    assert (data / image).read_bytes() == b"tiff"
    assert (data / label).read_bytes() == b"tiff"
    assert not (data / "raw/sen1floods11/v1.1/v1.1").exists()
    second = run()
    assert second.returncode == 0 and "skip" in second.stdout


def test_t1_failed_download_never_marks_partial_file_complete(tmp_path):
    image = "raw/sen1floods11/v1.1/data/S2Hand/chip.tif"
    data, run = prepare(tmp_path, "T1", [{"image_paths": [image], "label_path": image}], fail=True)
    result = run()
    assert result.returncode != 0
    assert "BOOTSTRAP-OK" not in result.stdout
    assert not (data / image).exists()
    assert Path(str(data / image) + ".part").is_file()


def test_t3_partial_layout_is_not_a_successful_bootstrap(tmp_path):
    image = "raw/burn_intensity/extracted/pre/image.tiff"
    data, run = prepare(
        tmp_path,
        "T3",
        [{"image_paths": [image], "label_path": image}],
        fail=True,
    )
    (data / image).parent.mkdir(parents=True)
    result = run()
    assert result.returncode != 0
    assert "BOOTSTRAP-OK" not in result.stdout
    assert "skip" not in result.stdout
