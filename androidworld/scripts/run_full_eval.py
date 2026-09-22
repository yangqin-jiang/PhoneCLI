#!/usr/bin/env python3
"""Launch one side of the full AndroidWorld A/B evaluation safely.

The OpenRouter key is read from OPENROUTER_API_KEY (or API_KEY) and passed
through the child environment, keeping it out of argv, tmux history, config,
and process listings.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("m3a", "macro"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--app-maps", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--console-port", type=int, required=True)
    parser.add_argument("--grpc-port", type=int, required=True)
    parser.add_argument("--adb-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-task-combinations", type=int, default=1)
    parser.add_argument(
        "--tasks",
        help="Optional comma-separated task names (default: full suite)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = yaml.safe_load(config_path.read_text()) or {}
    agent_config = config.get("agent", {}).get("args", {})
    llm_config = config.get("llm", {})

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("API_KEY")
        or agent_config.get("api_key")
        or llm_config.get("api_key")
    )
    api_base = agent_config.get("api_base", "https://openrouter.ai/api/v1")
    model = agent_config.get("model_name", "qwen/qwen3.7-plus")
    if not api_key:
        raise RuntimeError(
            "No API key found. Export OPENROUTER_API_KEY before launching."
        )

    common = [
        "--api-base", str(api_base),
        "--model", str(model),
        "--console-port", str(args.console_port),
        "--grpc-port", str(args.grpc_port),
        "--adb-path", str(args.adb_path),
        "--checkpoint-dir", str(Path(args.checkpoint_dir).resolve()),
        "--seed", str(args.seed),
        "--n-task-combinations", str(args.n_task_combinations),
    ]
    if args.tasks:
        common.extend(["--tasks", args.tasks])
    if args.method == "m3a":
        command = [sys.executable, str(repo / "run_m3a.py"), *common]
    else:
        command = [
            sys.executable,
            str(repo / "run.py"),
            *common,
            "--app-maps", str(Path(args.app_maps).resolve()),
            "--llm-api-base", str(llm_config.get("api_base", api_base)),
            "--llm-model", str(llm_config.get("model", model)),
        ]

    child_env = os.environ.copy()
    child_env["API_KEY"] = str(api_key)
    child_env.setdefault("API_BASE", str(api_base))
    if sys.platform.startswith("linux"):
        child_env.setdefault("ANDROID_ENV_A11Y_GRPC_HOST", "127.0.0.1")
    # Login shells spawned by tmux may omit env.sh, so make the package parent
    # explicit for every evaluation child. setup.sh also installs this repo as
    # the `aworld_agent` editable package.
    package_parent = str(repo.parent)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    pythonpath_entries = [
        entry for entry in existing_pythonpath.split(os.pathsep) if entry
    ]
    if package_parent not in pythonpath_entries:
        pythonpath_entries.insert(0, package_parent)
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    os.execvpe(command[0], command, child_env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
