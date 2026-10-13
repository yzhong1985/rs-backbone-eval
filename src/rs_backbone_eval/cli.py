"""Control-plane commands. Model execution is delegated to isolated workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .contracts import RunRecord
from .data.datasets import builder_for
from .data.manifest import audit_manifest, audit_task_manifest, write_manifest
from .downloads import DownloadManager
from .paths import RuntimePaths
from .registry import Registry

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()


def _format_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


@app.command("doctor")
def doctor() -> None:
    """Show resolved external paths and whether they exist."""
    paths = RuntimePaths.from_environment()
    table = Table("Purpose", "Path", "Exists")
    for name in ("data_root", "weights_root", "runs_root", "cache_root", "config_root"):
        path = getattr(paths, name)
        table.add_row(name, str(path), "yes" if path.exists() else "no")
    console.print(table)


@app.command("list-tasks")
def list_tasks(
    config_root: Annotated[Path | None, typer.Option(help="Registry root")] = None,
) -> None:
    """List the 8 executable task configurations."""
    root = (config_root or RuntimePaths.from_environment().config_root).resolve()
    table = Table("ID", "Application", "Dataset", "Family", "T", "Bands", "Primary metric")
    for task in Registry(root).tasks():
        table.add_row(
            task.task_id,
            task.application,
            task.dataset_id,
            task.family,
            str(task.time_steps),
            str(len(task.bands)),
            task.primary_metric.name,
        )
    console.print(table)


@app.command("list-downloads")
def list_downloads() -> None:
    """Show the local state of cataloged datasets and model weights."""
    paths = RuntimePaths.from_environment()
    manager = DownloadManager(
        paths.config_root / "downloads" / "catalog.yaml",
        paths.data_root,
        paths.weights_root,
        paths.cache_root,
    )
    table = Table("ID", "Kind", "Tasks", "State", "Local / Total", "Engine")
    for item in manager.summary()["items"]:
        total = _format_bytes(item["bytes_total"]) if item["bytes_total"] else "unknown"
        table.add_row(
            item["resource_id"],
            item["kind"],
            ",".join(item["task_ids"]),
            item["state"],
            f"{_format_bytes(item['bytes_downloaded'])} / {total}",
            item["engine"] or "missing",
        )
    console.print(table)


@app.command("plan")
def plan(
    protocol_id: Annotated[str, typer.Option(help="Protocol registry ID")] = "full_finetune",
    config_root: Annotated[Path | None, typer.Option(help="Registry root")] = None,
) -> None:
    """Print the model × task × seed execution matrix without starting workers."""
    root = (config_root or RuntimePaths.from_environment().config_root).resolve()
    registry = Registry(root)
    protocols = {protocol.protocol_id: protocol for protocol in registry.protocols()}
    if protocol_id not in protocols:
        raise typer.BadParameter(f"unknown protocol: {protocol_id}")
    protocol = protocols[protocol_id]
    tasks = registry.tasks()
    models = [model for model in registry.models() if model.enabled]
    table = Table("Task", "Model", "Protocol", "Seeds", "Runs")
    for task in tasks:
        for model in models:
            table.add_row(
                task.task_id,
                model.model_id,
                protocol.protocol_id,
                ",".join(str(seed) for seed in protocol.seeds),
                str(len(protocol.seeds)),
            )
    console.print(table)
    console.print(
        f"[bold]{len(tasks) * len(models) * len(protocol.seeds)} runs[/bold] "
        f"({len(tasks)} configs × {len(models)} models × {len(protocol.seeds)} seeds)"
    )


@app.command("validate-run")
def validate_run(path: Path) -> None:
    """Validate a run.json before publishing it to the result directory."""
    with path.open("r", encoding="utf-8") as handle:
        record = RunRecord.model_validate(json.load(handle))
    console.print(f"[green]valid[/green] {record.run_id}")


@app.command("audit-manifest")
def audit_manifest_command(path: Path) -> None:
    """Validate a JSONL data manifest and reject group leakage across splits."""
    audit = audit_manifest(path.resolve())
    table = Table("Item", "Value")
    table.add_row("samples", str(audit.sample_count))
    table.add_row("splits", json.dumps(audit.split_counts, ensure_ascii=False, sort_keys=True))
    table.add_row("tasks", json.dumps(audit.task_counts, ensure_ascii=False, sort_keys=True))
    table.add_row("schema errors", str(len(audit.errors)))
    table.add_row("leaking groups", str(len(audit.leakage)))
    console.print(table)
    for error in audit.errors[:10]:
        console.print(f"[red]- {error}[/red]")
    for group, splits in list(audit.leakage.items())[:10]:
        console.print(f"[red]- leakage {group}: {', '.join(splits)}[/red]")
    if not audit.valid:
        raise typer.Exit(code=1)


@app.command("audit-data")
def audit_data(
    manifest: Annotated[Path, typer.Argument(help="External JSONL manifest to audit")],
    task: Annotated[str | None, typer.Option(help="Task registry ID for contract checks")] = None,
    config_root: Annotated[Path | None, typer.Option(help="Registry root")] = None,
) -> None:
    """Audit a manifest against its TaskSpec: bands, sensor, time steps, and splits."""
    path = manifest.resolve()
    if task is None:
        audit_manifest_command(path)
        return

    root = (config_root or RuntimePaths.from_environment().config_root).resolve()
    tasks = {spec.task_id: spec for spec in Registry(root).tasks()}
    if task not in tasks:
        raise typer.BadParameter(f"unknown task: {task}; known: {', '.join(sorted(tasks))}")
    audit = audit_task_manifest(path, tasks[task])

    table = Table("Item", "Value")
    table.add_row("task", task)
    table.add_row("samples", str(audit.sample_count))
    table.add_row("splits", json.dumps(audit.split_counts, ensure_ascii=False, sort_keys=True))
    table.add_row("schema errors", str(len(audit.schema_errors)))
    table.add_row("duplicate samples", str(len(audit.duplicate_samples)))
    table.add_row("band mismatches", str(len(audit.band_mismatches)))
    table.add_row("sensor mismatches", str(len(audit.sensor_mismatches)))
    table.add_row("time-step mismatches", str(len(audit.time_step_mismatches)))
    table.add_row("unknown splits", str(len(audit.unknown_splits)))
    table.add_row("missing splits", str(len(audit.missing_splits)))
    table.add_row("leaking groups", str(len(audit.leakage)))
    console.print(table)
    for error in audit.all_errors()[:15]:
        console.print(f"[red]- {error}[/red]")
    if not audit.valid:
        raise typer.Exit(code=1)


@app.command("build-manifest")
def build_manifest(
    task: Annotated[str, typer.Option(help="Task registry ID")],
    config_root: Annotated[Path | None, typer.Option(help="Registry root")] = None,
    data_root: Annotated[Path | None, typer.Option(help="External data root")] = None,
) -> None:
    """Build, write, and audit the external JSONL manifest for one task."""
    root = (config_root or RuntimePaths.from_environment().config_root).resolve()
    tasks = {spec.task_id: spec for spec in Registry(root).tasks()}
    if task not in tasks:
        raise typer.BadParameter(f"unknown task: {task}; known: {', '.join(sorted(tasks))}")
    task_spec = tasks[task]
    data = (data_root or RuntimePaths.from_environment().data_root).resolve()

    try:
        samples = builder_for(task_spec.dataset_id)(data, task_spec)
    except ValueError as exc:
        console.print(f"[red]build failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    out_dir = data / "manifests" / task_spec.task_id
    staging = out_dir / ".building.jsonl"
    digest = write_manifest(staging, samples)
    final = out_dir / f"{digest}.jsonl"
    if final.exists():
        staging.unlink(missing_ok=True)
    else:
        staging.replace(final)

    audit = audit_task_manifest(final, task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    table = Table("Item", "Value")
    table.add_row("task", task)
    table.add_row("samples", str(audit.sample_count))
    table.add_row("splits", json.dumps(audit.split_counts, ensure_ascii=False, sort_keys=True))
    table.add_row("manifest", str(final))
    table.add_row("manifest_sha256", digest)
    table.add_row("errors", str(strict_errors))
    console.print(table)
    for group, splits in audit.leakage.items():
        console.print(
            f"[yellow]- warning: group {group} spans {', '.join(splits)} "
            f"(official splits may mix locations by design)[/yellow]"
        )
    for error in audit.all_errors()[:15]:
        if not any(error.startswith(prefix) for prefix in ("leakage",)):
            console.print(f"[red]- {error}[/red]")
    if strict_errors:
        raise typer.Exit(code=1)


@app.command("dispatch")
def dispatch_command(
    action: Annotated[str, typer.Option(help="probe / train / evaluate / ...")],
    task: Annotated[str, typer.Option(help="Task registry ID")],
    model: Annotated[str, typer.Option(help="Model registry ID")],
    protocol: Annotated[str, typer.Option(help="Protocol registry ID")],
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    run_id: Annotated[str | None, typer.Option(help="Override the generated run ID")] = None,
) -> None:
    """Validate a run, write its WorkerRequest, and execute the model worker."""
    from .dispatch import DispatchError, dispatch

    try:
        code = dispatch(
            action=action,
            task_id=task,
            model_id=model,
            protocol_id=protocol,
            seed=seed,
            run_id=run_id,
        )
    except DispatchError as exc:
        console.print(f"[red]dispatch rejected: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if code != 0:
        raise typer.Exit(code=code)


@app.command("portal")
def portal(
    runs_root: Annotated[Path | None, typer.Option(help="External run directory")] = None,
    host: Annotated[str | None, typer.Option(help="Bind address")] = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
) -> None:
    """Serve result browsing and catalog-controlled external downloads."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("install the portal extra: uv sync --extra portal") from exc

    from .portal.app import create_app

    root = (runs_root or RuntimePaths.from_environment().runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bind = host or os.getenv("RS_EVAL_PORTAL_HOST", "127.0.0.1")
    listen_port = port or int(os.getenv("RS_EVAL_PORTAL_PORT", "8080"))
    uvicorn.run(create_app(root), host=bind, port=listen_port)


if __name__ == "__main__":
    app()
