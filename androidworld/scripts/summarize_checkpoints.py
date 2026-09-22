#!/usr/bin/env python3
"""Summarize AndroidWorld checkpoint metadata without committing raw traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import pickle


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _load_result(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty checkpoint: {path}")
        value = value[-1]
    if not isinstance(value, dict):
        raise TypeError(f"Unexpected checkpoint payload in {path}: {type(value)}")
    return value


def _summarize(label: str, run_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    for path in sorted(run_dir.glob("*.pkl.gz")):
        result = _load_result(path)
        success = _clean(result.get("is_successful"))
        has_error = bool(result.get("exception_info"))
        status = "error" if has_error else "success" if success and success > 0.5 else "failure"
        rows.append({
            "run": label,
            "task": result.get("task_template") or path.name.removesuffix("_0.pkl.gz"),
            "status": status,
            "score": success,
            "steps": _clean(result.get("episode_length")),
            "runtime_s": _clean(result.get("run_time")),
        })

    token_path = run_dir / "token_usage.json"
    tokens = json.loads(token_path.read_text()) if token_path.exists() else {}
    finite_steps = [row["steps"] for row in rows if row["steps"] is not None]
    summary = {
        "run": label,
        "path": str(run_dir),
        "tasks": len(rows),
        "successes": sum(row["status"] == "success" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "average_steps_non_error": (
            sum(finite_steps) / len(finite_steps) if finite_steps else None
        ),
        "tokens": tokens,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="May be supplied multiple times",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = []
    summaries = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"Invalid --run {spec!r}; expected LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        rows, summary = _summarize(label, Path(raw_path).resolve())
        all_rows.extend(rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "task_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "task", "status", "score", "steps", "runtime_s"
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
