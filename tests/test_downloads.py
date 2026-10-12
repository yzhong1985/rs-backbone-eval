import hashlib
import time
from pathlib import Path

import yaml

from rs_backbone_eval.downloads import DownloadCatalog, DownloadManager

ROOT = Path(__file__).resolve().parents[1]


def write_catalog(path: Path) -> None:
    content = b"abc"
    payload = {
        "schema_version": "1.0",
        "resources": [
            {
                "resource_id": "model_fixture",
                "kind": "model",
                "display_name": "Fixture model",
                "description": "A local test fixture.",
                "task_ids": ["T1"],
                "license": "MIT",
                "source_page": "https://example.com/model",
                "root": "weights",
                "target_dir": "fixture",
                "method": "http_files",
                "expected_size_bytes": len(content),
                "size_is_estimate": False,
                "files": [
                    {
                        "url": "https://example.com/model.bin",
                        "filename": "model.bin",
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
                "completion_paths": ["model.bin"],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_official_download_catalog_has_models_and_all_tasks() -> None:
    catalog = DownloadCatalog.load(ROOT / "configs" / "downloads" / "catalog.yaml")

    assert len(catalog.resources) == 10
    assert {resource.kind for resource in catalog.resources} == {"dataset", "model"}
    assert {task for resource in catalog.resources for task in resource.task_ids} == {
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6a",
        "T6b",
        "T7",
    }
    assert all(".." not in Path(resource.target_dir).parts for resource in catalog.resources)


def test_download_status_and_local_verification(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    write_catalog(catalog_path)
    manager = DownloadManager(
        catalog_path,
        tmp_path / "data",
        tmp_path / "weights",
        tmp_path / "cache",
    )

    missing = manager.status("model_fixture")
    assert missing["state"] == "missing"
    assert missing["bytes_downloaded"] == 0

    target = tmp_path / "weights" / "fixture"
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"abc")
    assert manager.status("model_fixture")["state"] == "present"

    manager.verify("model_fixture")
    deadline = time.monotonic() + 2
    while manager.status("model_fixture")["state"] == "verifying":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    verified = manager.status("model_fixture")
    assert verified["state"] == "complete"
    assert verified["progress"] == 1
    assert (target / ".rs-download-complete.json").is_file()
