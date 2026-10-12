"""Export the worker/portal run contract as JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from rs_backbone_eval.contracts import RunRecord


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "schemas" / "run.schema.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(RunRecord.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
