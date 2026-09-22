#!/usr/bin/env python3
"""Finish all maps, validate them, then launch full M3A vs Macro+Map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

from build_all_maps import APPS


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _validate_maps(output_dir: Path) -> list[str]:
    errors = []
    for slug, _app_name, package in APPS:
        path = output_dir / f"{slug}.yaml"
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(f"{slug}: unreadable ({exc})")
            continue
        if data.get("package") != package:
            errors.append(f"{slug}: package={data.get('package')!r}")
        if not data.get("screens"):
            errors.append(f"{slug}: no screens")
    return errors


def _tmux_window_exists(session: str, name: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in result.stdout.splitlines()


def _launch_window(session: str, name: str, command: list[str], log: Path) -> None:
    if _tmux_window_exists(session, name):
        raise RuntimeError(f"tmux window already exists: {session}:{name}")
    shell_command = (
        "set -o pipefail; "
        + shlex.join(command)
        + " 2>&1 | tee "
        + shlex.quote(str(log))
    )
    subprocess.run(
        [
            "tmux", "new-window", "-d", "-t", f"{session}:",
            "-n", name, "bash", "-lc", shell_command,
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adb", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Waiting for initial map builder pid={args.wait_pid}; "
        f"target now contains {len(APPS)} apps.",
        flush=True,
    )
    last_report = 0.0
    while _pid_alive(args.wait_pid):
        now = time.time()
        if now - last_report >= 60:
            count = len(list(output_dir.glob("*.yaml")))
            print(f"Still building: {count}/{len(APPS)} YAML files present", flush=True)
            last_report = now
        time.sleep(args.poll_seconds)

    # Resume in the same directory.  Valid maps are skipped, so this builds
    # the three dynamic IR apps and retries any failures from the first pass.
    resume_command = [
        args.python,
        str(repo / "scripts" / "build_all_maps.py"),
        "--devices", "emulator-5554", "emulator-5556",
        "--output-dir", str(output_dir),
        "--config", str((repo / args.config).resolve()),
        "--adb", args.adb,
        "--max-screens", "50",
        "--max-depth", "3",
        "--scroll-pages", "3",
        "--retries", "1",
    ]
    print("Starting resumable 22-app map pass...", flush=True)
    resume = subprocess.run(resume_command, cwd=repo, check=False)
    errors = _validate_maps(output_dir)
    validation = {
        "expected_maps": len(APPS),
        "valid_maps": len(APPS) - len(errors),
        "errors": errors,
        "resume_returncode": resume.returncode,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    if resume.returncode != 0 or errors:
        print(f"Map validation failed: {validation}", flush=True)
        return 2

    launcher = str(repo / "scripts" / "run_full_eval.py")
    common = [
        "--config", str((repo / args.config).resolve()),
        "--app-maps", str(output_dir),
        "--adb-path", args.adb,
        "--seed", "42",
        "--n-task-combinations", "1",
    ]
    m3a_command = [
        args.python, launcher,
        "--method", "m3a",
        "--checkpoint-dir", str(checkpoint_root / "m3a"),
        "--console-port", "5556",
        "--grpc-port", "8556",
        *common,
    ]
    macro_command = [
        args.python, launcher,
        "--method", "macro",
        "--checkpoint-dir", str(checkpoint_root / "macro_map"),
        "--console-port", "5554",
        "--grpc-port", "8554",
        *common,
    ]
    _launch_window(
        args.session, "full_m3a", m3a_command, checkpoint_root / "m3a.log"
    )
    _launch_window(
        args.session, "full_macro", macro_command, checkpoint_root / "macro_map.log"
    )
    print(f"LAUNCHED {args.session}:full_m3a and {args.session}:full_macro", flush=True)
    print(f"CHECKPOINT_ROOT {checkpoint_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
