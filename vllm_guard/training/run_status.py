#!/usr/bin/env python3
"""Helpers for writing lightweight training run status artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def write_run_status(
    output_dir: str | Path,
    status: str,
    *,
    run_name: str = "",
    log_file: str = "",
    wandb_project: str = "",
    wandb_run_name: str = "",
    notes: str = "",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "log_file": log_file,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "notes": notes,
    }

    status_path = output_dir / "run_status.json"
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if status == "done":
        (output_dir / "RUN_DONE").write_text(payload["timestamp"] + "\n", encoding="utf-8")
        failed = output_dir / "RUN_FAILED"
        if failed.exists():
            failed.unlink()
    elif status == "failed":
        (output_dir / "RUN_FAILED").write_text(payload["timestamp"] + "\n", encoding="utf-8")
        done = output_dir / "RUN_DONE"
        if done.exists():
            done.unlink()

    return status_path
