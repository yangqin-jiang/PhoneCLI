"""PhoneCLI Macro Agent on AndroidWorld benchmark.

Usage:
    cd phonecli-aworld-agent
    conda activate aworld
    source env.sh

    # Pure VLM mode:
    python run.py --model=qwen/qwen3.7-plus

    # Macro routing mode (with app map):
    python run.py --model=qwen/qwen3.7-plus \\
        --app-maps=./app_maps/androidworld_22 --tasks=SystemWifiTurnOn

    # With separate text LLM:
    python run.py --model=qwen/qwen3.7-plus \\
        --llm-api-base=http://localhost:8002/v1 --llm-model=Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import logging
import os
import sys

# Add android_world repo to path (only external dependency)
_ROOT = os.path.dirname(os.path.abspath(__file__))
_AW = os.path.join(_ROOT, "android_world")
_PACKAGE_PARENT = os.path.dirname(_ROOT)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)
if os.path.isdir(_AW) and _AW not in sys.path:
    sys.path.insert(0, _AW)


def main():
    p = argparse.ArgumentParser(
        description="PhoneCLI on AndroidWorld",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- API ----
    default_api_key = (
        os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY", "EMPTY")
    )
    p.add_argument("--api-key", default=default_api_key,
                   help="VLM API key")
    p.add_argument("--api-base", default=os.getenv("API_BASE", "https://openrouter.ai/api/v1"),
                   help="VLM API base URL")
    p.add_argument("--model", default="qwen/qwen3.7-plus",
                   help="VLM model name for vision calls")

    # ---- LLM (text-only, macro mapping) ----
    p.add_argument("--llm-api-key", default=None,
                   help="Text LLM API key (defaults to --api-key)")
    p.add_argument("--llm-api-base", default=None,
                   help="Text LLM API base (defaults to --api-base)")
    p.add_argument("--llm-model", default=None,
                   help="Text LLM model (defaults to --model)")

    # ---- AndroidWorld ----
    p.add_argument("--suite-family", default="android_world")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task names (default: all)")
    p.add_argument("--n-task-combinations", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--console-port", type=int, default=5554,
                   help="Android emulator console port")
    p.add_argument("--grpc-port", type=int, default=8554,
                   help="Android emulator gRPC port")
    p.add_argument("--emulator-setup", action="store_true", default=False,
                   help="Run emulator setup before starting")
    default_sdk = os.getenv(
        "ANDROID_SDK_ROOT", os.path.expanduser("~/Library/Android/sdk")
    )
    p.add_argument(
        "--adb-path",
        default=os.path.join(default_sdk, "platform-tools", "adb"),
    )

    # ---- PhoneCLI macro routing ----
    p.add_argument("--app-maps", default=None,
                   help="Directory of app map YAMLs for macro routing")
    p.add_argument("--no-macro", action="store_true", default=False,
                   help="Disable macro routing (pure VLM mode)")

    # ---- Full PhoneCLI map-layer options (mirror evaluation/macro_agent.py) ----
    p.add_argument("--force-macro-vlm", action="store_true", default=False,
                   help="Downgrade OP → MACRO_VLM so the VLM always interacts")
    p.add_argument("--skip-landing-check", action="store_true", default=False,
                   help="Skip the landing-mismatch guard (ablation)")
    p.add_argument("--landing-hint-level", type=int, default=2,
                   help="0=bare, 1=+screen desc, 2=+expected elements (+cold read)")
    p.add_argument("--cold-read-warmup", action="store_true", default=False,
                   help="One-shot VLM cold read of the screen after macro landing")

    # ---- Output ----
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint directory (auto-created if not set)")
    p.add_argument("--output-path", default="./aw_checkpoints")

    # ---- Debug ----
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--demo", action="store_true", default=False,
                   help="Demo mode (shows scoreboard on device)")

    args = p.parse_args()

    # AndroidEnv logs the full host environment during setup.  The key has
    # already been copied into argparse, so remove it before AndroidEnv starts
    # to keep credentials out of evaluation logs and tmux scrollback.
    os.environ.pop("API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)

    # Suppress gRPC debug logging that leaks into ADB command output on Mac ARM64
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    if sys.platform.startswith("linux"):
        # Keep accessibility observations alive when a benchmark task turns
        # guest Wi-Fi off.  The controller maps this loopback endpoint through
        # an ADB reverse tunnel.
        os.environ.setdefault("ANDROID_ENV_A11Y_GRPC_HOST", "127.0.0.1")

    # Mac + conda sqlite3 FTS4 workaround (needed for Joplin tasks).
    _brew_sqlite = "/opt/homebrew/opt/sqlite/lib"
    if os.path.isdir(_brew_sqlite):
        os.environ.setdefault("DYLD_LIBRARY_PATH", "")
        if _brew_sqlite not in os.environ["DYLD_LIBRARY_PATH"]:
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{_brew_sqlite}:{os.environ['DYLD_LIBRARY_PATH']}".rstrip(":")
            )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_androidworld")

    # ---- Resolve LLM config ----
    llm_config = {
        "api_key": args.llm_api_key or args.api_key,
        "api_base": args.llm_api_base or args.api_base,
        "model": args.llm_model or args.model,
        # Full PhoneCLI map-layer options (mirror evaluation/macro_agent.py).
        "force_macro_vlm": args.force_macro_vlm,
        "skip_landing_check": args.skip_landing_check,
        "landing_hint_level": args.landing_hint_level,
        "cold_read_warmup": args.cold_read_warmup,
    }

    # ---- Load app maps ----
    app_maps = {}
    if args.app_maps and not args.no_macro:
        from aworld_agent.app_map import AppMap
        import glob as _glob
        for path in sorted(_glob.glob(os.path.join(args.app_maps, "*.yaml"))):
            try:
                am = AppMap(path)
                app_maps[am.package] = am
                logger.info("Loaded app map: %s (%s)", am.app_name, am.package)
            except Exception as e:
                logger.warning("Failed to load app map %s: %s", path, e)

    # ---- Import AndroidWorld (after arg parsing so errors are clear) ----
    from android_world import checkpointer as checkpointer_lib
    from android_world import suite_utils
    from android_world.env import env_launcher
    from android_world.registry import TaskRegistry
    from android_world.utils import datetime_utils

    from aworld_agent.m3a_map_agent import M3AMapAgent, OpenRouterGpt4Wrapper

    # Monkey-patch: settings service is unstable on Mac ARM64 emulator.
    # Make setup_datetime robust to transient failures so tasks don't crash.
    _orig_setup_dt = datetime_utils.setup_datetime
    def _robust_setup_datetime(env):
        try:
            _orig_setup_dt(env)
        except Exception as e:
            logger.warning("setup_datetime failed (non-fatal): %s", e)
    datetime_utils.setup_datetime = _robust_setup_datetime

    # Token tracking
    import datetime as _dt
    from aworld_agent.token_usage import token_usage
    _run_start = _dt.datetime.now()

    # ---- Task list ----
    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        logger.info("Tasks: %s", task_names)

    # ---- Setup AndroidWorld env ----
    logger.info("Setting up AndroidWorld environment...")
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=args.emulator_setup,
        freeze_datetime=False,  # datetime was frozen during initial setup; skip re-freeze
        adb_path=os.path.expanduser(args.adb_path),
        grpc_port=args.grpc_port,
    )
    logger.info("Environment ready. Screen: %s", env.device_screen_size)

    # ---- Create task suite ----
    registry = TaskRegistry()
    suite = suite_utils.create_suite(
        registry.get_registry(family=args.suite_family),
        n_task_combinations=args.n_task_combinations,
        seed=args.seed,
        tasks=task_names,
        use_identical_params=False,
    )
    suite.suite_family = args.suite_family
    logger.info("Suite: %d tasks from %s", len(suite), args.suite_family)

    if app_maps:
        logger.info(
            "Macro routing: %d app maps; selected per task from task.app_names",
            len(app_maps),
        )

    # ---- Create agent (M3A core + app-map macro routing) ----
    llm = OpenRouterGpt4Wrapper(
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    agent = M3AMapAgent(
        env=env,
        llm=llm,
        app_maps=app_maps,
        llm_config=llm_config,
        name="m3a_map",
    )
    agent.transition_pause = None  # auto-stabilize
    # ---- Checkpoint ----
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(args.output_path)

    checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)

    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Model: %s  |  Macro: %s",
                args.model,
                "disabled" if args.no_macro or not app_maps else "per-task",
                )

    # ---- Run ----
    logger.info("Starting evaluation...")
    results = suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer,
        demo_mode=args.demo,
    )

    # ---- Summarize ----
    success_count = sum(1 for r in results if r.get("is_successful", 0) > 0.5)
    # The lifecycle hook runs before AndroidWorld computes task success. Merge
    # the final benchmark fields into the already captured per-task token rows.
    # Match BY TASK NAME, not zip(): tasks that fail before run_episode (e.g.
    # setup error) never trigger on_task_end, so zip() would misalign.
    result_by_task = {}
    for r in results:
        result_by_task.setdefault(r.get("task_template"), r)
    for record in agent.task_usage_records:
        r = result_by_task.get(record.get("task"))
        if r is None:
            logger.warning("No result found for task %s", record.get("task"))
            continue
        record.update({
            "is_successful": r.get("is_successful"),
            "episode_length": r.get("episode_length"),
            "run_time": r.get("run_time"),
            "has_error": bool(r.get("exception_info")),
        })
    # Append placeholder records for tasks that never produced an episode so
    # the JSONL covers all tasks.
    recorded = {rec.get("task") for rec in agent.task_usage_records}
    for r in results:
        tname = r.get("task_template")
        if tname and tname not in recorded:
            agent.task_usage_records.append({
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "cache_read_tokens": 0,
                "cache_write_tokens": 0, "total_calls": 0,
                "per_label": {},
                "task": tname,
                "is_successful": r.get("is_successful"),
                "episode_length": r.get("episode_length"),
                "run_time": r.get("run_time"),
                "has_error": bool(r.get("exception_info")),
            })
            recorded.add(tname)
    logger.info("Done! %d/%d tasks completed successfully.", success_count, len(results))
    logger.info("Checkpoint: %s", checkpoint_dir)

    # Print detailed results
    print(f"\n=== Results ===")
    for r in results:
        task_name = r.get("task_template", r.get("goal", "?"))[:60]
        success = r.get("is_successful", 0)
        n_steps = r.get("episode_length", 0)
        run_time = r.get("run_time", 0)
        exc = r.get("exception_info")
        if exc:
            status = "ERROR"
        elif success > 0.5:
            status = "OK"
        else:
            status = "FAIL"
        print(f"  [{status}] success={success:.1f} steps={n_steps} time={run_time:.0f}s  {task_name}")

    # ---- Token usage report (save + print, same as Android Lab) ----
    _elapsed = (_dt.datetime.now() - _run_start).total_seconds()
    token_report = token_usage.report()
    logger.info("Wall time: %.0fs (%.0f min)", _elapsed, _elapsed / 60)
    print(f"\n{token_report}")
    token_usage.save(os.path.join(checkpoint_dir, "token_usage.json"))
    logger.info("Token usage saved to: %s", os.path.join(checkpoint_dir, "token_usage.json"))

    # One JSON object per episode, captured by the AndroidWorld lifecycle hook.
    _per_task_path = os.path.join(checkpoint_dir, "token_usage_per_task.jsonl")
    with open(_per_task_path, "w") as _ptf:
        for record in agent.task_usage_records:
            _ptf.write(json.dumps(record) + "\n")
    logger.info("Per-task token usage saved to: %s", _per_task_path)

    _summary_path = os.path.join(checkpoint_dir, "run_summary.json")
    with open(_summary_path, "w") as _summary_file:
        json.dump({
            "run_start": _run_start.isoformat(),
            "elapsed_s": _elapsed,
            "n_tasks": len(results),
            "n_success": success_count,
            "aggregate": token_usage.to_dict(),
        }, _summary_file, indent=2)
    logger.info("Run summary saved to: %s", _summary_path)


if __name__ == "__main__":
    main()
