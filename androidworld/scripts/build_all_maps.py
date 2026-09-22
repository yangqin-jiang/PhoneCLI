#!/usr/bin/env python3
"""Build maps for every app used by the AndroidWorld task registry.

The runner assigns apps dynamically to a fixed set of existing emulators,
writes one log per app, and can safely resume an interrupted output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

import yaml


# Unique launchable apps referenced by AndroidWorld's 116 registered task
# classes.  The last three are assigned dynamically by information-retrieval
# task instances, so they are not visible on the task classes themselves.
# Order roughly follows task coverage so the most useful maps are available
# early.
APPS = [
    ("settings", "Settings", "com.android.settings"),
    ("markor", "Markor", "net.gsantner.markor"),
    ("broccoli", "Broccoli", "com.flauschcode.broccoli"),
    ("pro_expense", "Pro Expense", "com.arduia.expense"),
    ("simple_calendar_pro", "Simple Calendar Pro", "com.simplemobiletools.calendar.pro"),
    ("simple_sms_messenger", "Simple SMS Messenger", "com.simplemobiletools.smsmessenger"),
    ("retro_music", "Retro Music", "code.name.monkey.retromusic"),
    ("simple_gallery_pro", "Simple Gallery Pro", "com.simplemobiletools.gallery.pro"),
    ("clock", "Clock", "com.google.android.deskclock"),
    ("camera", "Camera", "com.android.camera2"),
    ("chrome", "Chrome", "com.android.chrome"),
    ("contacts", "Contacts", "com.google.android.contacts"),
    ("audio_recorder", "Audio Recorder", "com.dimowner.audiorecorder"),
    ("osmand", "OsmAnd", "net.osmand"),
    ("vlc", "VLC", "org.videolan.vlc"),
    ("clipper", "Clipper", "ca.zgrs.clipper"),
    ("files", "Files", "com.google.android.documentsui"),
    ("dialer", "Dialer", "com.google.android.dialer"),
    ("simple_draw_pro", "Simple Draw Pro", "com.simplemobiletools.draw.pro"),
    ("tasks", "Tasks", "org.tasks"),
    ("open_tracks", "Open Tracks Sports Tracker", "de.dennisguse.opentracks"),
    ("joplin", "Joplin", "net.cozic.joplin"),
]


def _valid_existing_map(path: Path, package: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return False
    return data.get("package") == package and bool(data.get("screens"))


def _package_installed(adb: str, device: str, package: str) -> bool:
    result = subprocess.run(
        [adb, "-s", device, "shell", "pm", "path", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _restore_snapshot(adb: str, device: str, package: str) -> bool:
    """Restore AndroidWorld's clean per-app data snapshot before crawling."""
    snapshot = f"/data/data/android_world/snapshots/{package}"
    target = f"/data/data/{package}"
    exists = subprocess.run(
        [adb, "-s", device, "shell", "test", "-d", snapshot],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        return False

    subprocess.run(
        [adb, "-s", device, "shell", "am", "force-stop", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    restore = subprocess.run(
        [
            adb,
            "-s",
            device,
            "shell",
            f"cp -a {snapshot}/. {target}/ && "
            f"restorecon -RD {target} && chmod 777 -R {target}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return restore.returncode == 0


def _sync_package_policy(
    adb: str,
    source_device: str,
    target_device: str,
    package: str,
) -> tuple[int, int]:
    """Copy runtime grants and explicit allowed app-ops from a clean device."""
    package_info = subprocess.run(
        [adb, "-s", source_device, "shell", "dumpsys", "package", package],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    granted_permissions = sorted(set(re.findall(
        r"^\s+(android\.permission\.[A-Z0-9_]+): granted=true",
        package_info,
        flags=re.MULTILINE,
    )))
    for permission in granted_permissions:
        subprocess.run(
            [adb, "-s", target_device, "shell", "pm", "grant", package, permission],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    appops_output = subprocess.run(
        [adb, "-s", source_device, "shell", "appops", "get", package],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    copied_appops = set()
    for line in appops_output.splitlines():
        match = re.search(
            r"(?:Uid mode:\s*)?([A-Z][A-Z0-9_]+):\s+(allow|foreground)\b",
            line,
        )
        if not match:
            continue
        operation, mode = match.groups()
        uid_mode = line.strip().startswith("Uid mode:")
        command = [adb, "-s", target_device, "shell", "appops", "set"]
        if uid_mode:
            command.append("--uid")
        command.extend([package, operation, mode])
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        copied_appops.add((operation, mode, uid_mode))

    return len(granted_permissions), len(copied_appops)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--max-screens", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--scroll-pages", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--no-enrich", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = Path(args.config)
    if not config.is_absolute():
        config = (repo / config).resolve()
    build_script = repo / "build_map.py"

    missing = []
    for device in args.devices:
        for _, _, package in APPS:
            if not _package_installed(args.adb, device, package):
                missing.append((device, package))
    if missing:
        for device, package in missing:
            print(f"[Preflight] MISSING {package} on {device}", flush=True)
        return 2

    reference_device = args.devices[0]
    for device in args.devices[1:]:
        permission_count = 0
        appop_count = 0
        for _, _, package in APPS:
            permissions, appops = _sync_package_policy(
                args.adb, reference_device, device, package
            )
            permission_count += permissions
            appop_count += appops
        print(
            f"[Preflight] Synced policy {reference_device} -> {device}: "
            f"permissions={permission_count} appops={appop_count}",
            flush=True,
        )

    missing_snapshots = []
    for device in args.devices:
        for _, _, package in APPS:
            snapshot = f"/data/data/android_world/snapshots/{package}"
            result = subprocess.run(
                [args.adb, "-s", device, "shell", "test", "-d", snapshot],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                missing_snapshots.append((device, package))
    if missing_snapshots:
        for device, package in missing_snapshots:
            print(f"[Preflight] MISSING_SNAPSHOT {package} on {device}", flush=True)
        return 3

    work: queue.Queue[tuple[str, str, str]] = queue.Queue()
    skipped = []
    for spec in APPS:
        slug, _, package = spec
        output = output_dir / f"{slug}.yaml"
        if _valid_existing_map(output, package):
            skipped.append(slug)
        else:
            work.put(spec)

    results = []
    result_lock = threading.Lock()
    started_at = time.time()

    def record(item: dict) -> None:
        with result_lock:
            results.append(item)
            manifest = {
                "started_at_unix": started_at,
                "updated_at_unix": time.time(),
                "total_apps": len(APPS),
                "skipped_existing": skipped,
                "results": sorted(results, key=lambda r: r["slug"]),
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )

    def worker(device: str) -> None:
        while True:
            try:
                slug, app_name, package = work.get_nowait()
            except queue.Empty:
                return

            output = output_dir / f"{slug}.yaml"
            log_path = log_dir / f"{slug}.log"
            success = False
            returncode = None
            app_started = time.time()

            for attempt in range(1, args.retries + 2):
                command = [
                    sys.executable,
                    str(build_script),
                    "--package", package,
                    "--app", app_name,
                    "--device", device,
                    "--output", str(output),
                    "--config", str(config),
                    "--max-screens", str(args.max_screens),
                    "--max-depth", str(args.max_depth),
                    "--scroll-pages", str(args.scroll_pages),
                ]
                if args.no_classify:
                    command.append("--no-classify")
                if args.no_enrich:
                    command.append("--no-enrich")

                print(
                    f"[{device}] START {slug} attempt={attempt} "
                    f"remaining={work.qsize()}",
                    flush=True,
                )
                with log_path.open("a") as log:
                    log.write(
                        f"\n===== attempt {attempt} device={device} "
                        f"time={time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
                    )
                    log.flush()
                    if not _restore_snapshot(args.adb, device, package):
                        log.write(
                            f"Snapshot restore failed for {package} on {device}\n"
                        )
                        log.flush()
                        returncode = 3
                        continue
                    child_env = os.environ.copy()
                    child_env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.run(
                        command,
                        cwd=repo,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=child_env,
                    )
                returncode = proc.returncode
                success = returncode == 0 and _valid_existing_map(output, package)
                if success:
                    break
                print(
                    f"[{device}] RETRYABLE_FAIL {slug} rc={returncode} "
                    f"log={log_path}",
                    flush=True,
                )

            elapsed = round(time.time() - app_started, 1)
            status = "DONE" if success else "FAILED"
            print(
                f"[{device}] {status} {slug} elapsed={elapsed}s "
                f"output={output.name}",
                flush=True,
            )
            record({
                "slug": slug,
                "app": app_name,
                "package": package,
                "device": device,
                "success": success,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "output": str(output),
                "log": str(log_path),
            })
            work.task_done()

    print(
        f"Building {len(APPS) - len(skipped)}/{len(APPS)} maps with "
        f"{len(args.devices)} devices into {output_dir}",
        flush=True,
    )
    threads = [
        threading.Thread(target=worker, args=(device,), name=device)
        for device in args.devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [result for result in results if not result["success"]]
    print(
        f"ALL DONE success={len(results) - len(failures)} "
        f"failed={len(failures)} skipped={len(skipped)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
