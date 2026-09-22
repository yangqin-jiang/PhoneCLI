"""M3A baseline on AndroidWorld with OpenRouter-compatible API.

Usage:
    cd phone_cli
    conda activate aworld
    source aworld_agent/env.sh
    python aworld_agent/run_m3a.py --api-key=sk-... --tasks=SystemWifiTurnOn
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

import requests


# ---------------------------------------------------------------------------
# Thin wrapper: redirect OpenAI API calls to OpenRouter
# ---------------------------------------------------------------------------

class OpenRouterGpt4Wrapper:
    """Drop-in replacement for android_world.agents.infer.Gpt4Wrapper.

    Uses the same predict_mm() interface but sends requests to a configurable
    API base (e.g. https://openrouter.ai/api/v1) instead of api.openai.com.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(self, model_name, api_key, api_base, max_retry=3, temperature=0.0):
        self.model = model_name
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.max_retry = min(max(max_retry, 1), 5)
        self.temperature = temperature

    @staticmethod
    def encode_image(image):
        import base64
        import io
        img = Image.fromarray(image.astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def predict(self, text_prompt):
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt, images):
        import time
        import numpy as np
        from PIL import Image

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': text_prompt}],
            }],
            'max_tokens': 1000,
        }
        if (
            'openrouter.ai' in self.api_base.lower()
            and self.model.lower() in ('qwen/qwen3.7-plus', 'z-ai/glm-4.6v')
        ):
            # The model reasons by default on OpenRouter.  M3A needs a strict
            # ``Reason:/Action:`` answer, and a small completion budget can be
            # exhausted by reasoning before message.content is emitted.
            payload['reasoning'] = {'effort': 'none'}
        for image in images:
            payload['messages'][0]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self._encode(image)}',
                },
            })

        url = f'{self.api_base}/chat/completions'
        counter = self.max_retry
        while counter > 0:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                body = r.json()
                if r.ok and 'choices' in body:
                    usage = body.get('usage') or {}
                    prompt_details = usage.get('prompt_tokens_details') or {}
                    from aworld_agent.token_usage import token_usage
                    token_usage.add(
                        prompt_tokens=int(usage.get('prompt_tokens') or 0),
                        completion_tokens=int(usage.get('completion_tokens') or 0),
                        cache_read_tokens=int(
                            prompt_details.get('cached_tokens') or 0
                        ),
                        label='m3a_vlm',
                    )
                    content = body['choices'][0]['message'].get('content')
                    if isinstance(content, str) and content.strip():
                        return content, None, r
                    print('  LLM error: successful response had no final content')
                    time.sleep(self.RETRY_WAITING_SECONDS)
                    counter -= 1
                    continue
                msg = r.json().get('error', {}).get('message', str(r.status_code))
                print(f'  LLM error: {msg}')
                time.sleep(self.RETRY_WAITING_SECONDS)
                counter -= 1
            except Exception as e:
                print(f'  Request error: {e}')
                time.sleep(self.RETRY_WAITING_SECONDS)
                counter -= 1
        return None, False, None

    @staticmethod
    def _encode(image):
        import base64, io
        from PIL import Image
        img = Image.fromarray(image.astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="M3A baseline on AndroidWorld")
    default_api_key = (
        os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY", "EMPTY")
    )
    p.add_argument("--api-key", default=default_api_key)
    p.add_argument("--api-base", default=os.getenv("API_BASE", "https://openrouter.ai/api/v1"))
    p.add_argument("--model", default="qwen/qwen3.7-plus")
    p.add_argument("--tasks", default=None)
    p.add_argument("--console-port", type=int, default=5556)
    p.add_argument("--grpc-port", type=int, default=8556)
    _default_sdk = os.getenv(
        "ANDROID_SDK_ROOT", os.path.expanduser("~/Library/Android/sdk")
    )
    p.add_argument(
        "--adb-path",
        default=os.path.join(_default_sdk, "platform-tools", "adb"),
    )
    p.add_argument("--n-task-combinations", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", default="./aw_checkpoints_m3a")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    # AndroidEnv logs the full host environment during setup.  The key has
    # already been copied into argparse, so remove it before AndroidEnv starts
    # to keep credentials out of evaluation logs and tmux scrollback.
    os.environ.pop("API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    if sys.platform.startswith("linux"):
        os.environ.setdefault("ANDROID_ENV_A11Y_GRPC_HOST", "127.0.0.1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("m3a_baseline")

    # Mac FTS4 workaround
    _brew = "/opt/homebrew/opt/sqlite/lib"
    if os.path.isdir(_brew):
        os.environ.setdefault("DYLD_LIBRARY_PATH", "")
        if _brew not in os.environ["DYLD_LIBRARY_PATH"]:
            os.environ["DYLD_LIBRARY_PATH"] = f"{_brew}:{os.environ['DYLD_LIBRARY_PATH']}".rstrip(":")

    from android_world import checkpointer as checkpointer_lib
    from android_world import suite_utils
    from android_world.env import env_launcher
    from android_world.registry import TaskRegistry
    from android_world.agents import m3a

    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]

    # Setup emulator
    logger.info("Setting up environment (port=%d, grpc=%d)...", args.console_port, args.grpc_port)
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=False,
        freeze_datetime=False,
        adb_path=os.path.expanduser(args.adb_path),
        grpc_port=args.grpc_port,
    )
    logger.info("Env ready. Screen: %s", env.device_screen_size)

    # Create M3A with OpenRouter-backed LLM
    llm = OpenRouterGpt4Wrapper(
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    agent = m3a.M3A(env, llm, name="m3a_qwen")
    agent.transition_pause = None
    from aworld_agent.token_usage import token_usage

    task_usage_records = []
    task_usage_snapshot = None

    def _prepare_for_task(_task):
        nonlocal task_usage_snapshot
        task_usage_snapshot = token_usage.snapshot()

    def _on_task_end(task, result):
        nonlocal task_usage_snapshot
        if task_usage_snapshot is None:
            return
        record = token_usage.diff(task_usage_snapshot)
        if isinstance(result, dict):
            metadata = {
                "is_successful": result.get("is_successful"),
                "episode_length": result.get("episode_length"),
                "run_time": result.get("run_time"),
                "has_error": bool(result.get("exception_info")),
            }
        else:
            step_data = getattr(result, "step_data", {}) or {}
            metadata = {
                "is_successful": None,
                "episode_length": len(step_data.get("step_number", [])),
                "run_time": None,
                "has_error": False,
                "agent_done": bool(getattr(result, "done", False)),
            }
        record.update({
            "task": getattr(task, "name", type(task).__name__),
            **metadata,
        })
        task_usage_records.append(record)
        task_usage_snapshot = None

    agent.prepare_for_task = _prepare_for_task
    agent.on_task_end = _on_task_end
    registry = TaskRegistry()
    suite = suite_utils.create_suite(
        registry.get_registry(family='android_world'),
        n_task_combinations=args.n_task_combinations,
        seed=args.seed,
        tasks=task_names,
        use_identical_params=False,
    )
    suite.suite_family = 'android_world'
    logger.info(
        "Running %d AndroidWorld tasks%s",
        len(suite),
        f": {task_names}" if task_names is not None else " (full suite)",
    )

    # Checkpoint
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        ckpt_dir = checkpointer_lib.create_run_directory(args.output_path)

    # Run
    logger.info("Model: %s | Checkpoint: %s", args.model, ckpt_dir)
    results = suite_utils.run(
        suite, agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(ckpt_dir),
    )

    # Summarize
    success = sum(1 for r in results if r.get("is_successful", 0) > 0.5)
    # Align per-task token records to results BY TASK NAME, not by zip.
    # A task that fails before run_episode (e.g. setup error) never
    # triggers on_task_end, so task_usage_records is shorter than results
    # and zip() silently misaligns every subsequent record's
    # is_successful/episode_length (observed: TasksHighPriorityTasks,
    # SystemBrightnessMax reported 1.0 while the evaluator says 0.0).
    result_by_task = {}
    for r in results:
        result_by_task.setdefault(r.get("task_template"), r)
    for record in task_usage_records:
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
    # Append placeholder records for tasks that never produced an episode
    # (e.g. app-missing setup failures) so the JSONL covers all tasks.
    recorded = {rec.get("task") for rec in task_usage_records}
    for r in results:
        tname = r.get("task_template")
        if tname and tname not in recorded:
            logger.warning(
                "Task %s never produced a token record; appending placeholder",
                tname,
            )
            task_usage_records.append({
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
    logger.info("Done: %d/%d", success, len(results))
    print(f"\n=== M3A + {args.model} Results ===")
    for r in results:
        s = r.get("is_successful", 0)
        print(f"  [{'OK' if s > 0.5 else 'FAIL'}] {r.get('task_template', '?')}  steps={r.get('episode_length',0)}")
    print(f"\nTotal: {success}/{len(results)}")

    # Token report uses the same TokenUsage accumulator as Macro+Map so the
    # two methods can be compared with identical accounting fields.
    print(f"\n{token_usage.report()}")
    token_usage.save(os.path.join(ckpt_dir, "token_usage.json"))
    with open(os.path.join(ckpt_dir, "token_usage_per_task.jsonl"), "w") as f:
        for record in task_usage_records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
