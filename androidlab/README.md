# AndroidLab — PhoneCLI vs. pure-VLM baseline

AndroidLab benchmark harness (9 apps / 138 tasks, THUDM
[Android-Lab](https://github.com/THUDM/Android-Lab) task suite) with two agent
implementations side by side.

## The two variants

| Variant | Config | Task class | Agent implementation | Uses app map | Deterministic replay |
|---------|--------|-----------|----------------------|:--:|:--:|
| **PhoneCLI** (macro agent) | `configs/test_macro.yaml` | `MacroAgentTask_AutoTest` | `evaluation/macro_agent.py::MacroAgentTask` | ✅ | ✅ |
| **without PhoneCLI** (baseline) | `configs/test_screen_cloud.yaml` | `ScreenshotCloudTask_AutoTest` | `evaluation/evaluation.py::ScreenshotCloudTask` | ✗ | ✗ |

PhoneCLI works in two phases: **Round 1** maps the task instruction onto an
operation in the pre-built app map (LLM Phase 1 selection + Phase 2 target-page
verification), replays that navigation path over ADB, checks the landing screen
and hands the VLM a page description; **Round 2+** is the ordinary cloud-style
VLM loop with injected `STATE_ASSESSMENT` history. When no operation matches (or
the map is too small), it degrades gracefully to the pure-VLM path, so it can
never score below the baseline.

The baseline (`ScreenshotCloudTask`) is a single VLM call per round with history
injection, no app map and no replay — the "PhoneCLI removed" reference point.

Both configs are templates: `test_macro.yaml` points `app_map` at
`./app_maps/settings_android.yaml`. To run another app, copy the file and repoint
`app_map` at that app's map (nine maps ship in `app_maps/`); the baseline config
needs no change since it never reads a map.

## Layout

```text
eval.py                    Test runner (agent config + task config → logs/evaluation/<name>)
run_eval_judge.py          Judge: re-scores recorded traces (programmatic XML / LLM)
build_android_map.py       App-map builder CLI (clone AVD → BFS crawl → YAML)
generate_result.py         Result aggregation helpers
config_env.py              YAML loading with ${ENV_VAR} expansion
adb_client.py              ADB helper injected into Docker containers
requirements.txt           Python dependencies
agent/                     LLM/MLLM client wrappers (OpenAI-compatible, Qwen, GLM, Claude)
evaluation/                Android-Lab framework
├── macro_agent.py         PhoneCLI macro agent  ← the PhoneCLI implementation
├── evaluation.py          Task base classes incl. ScreenshotCloudTask (baseline)
├── app_map.py             App-map runtime: load / query / ops catalog / nav reference
├── build_map.py           BFS crawler + LLM element classification & enrichment
├── auto_test.py           Per-task orchestration, AVD lifecycle, token accounting
├── configs.py             AppConfig / TaskConfig
├── task.py, definition.py, utils.py, docker_utils.py, parallel.py
├── config/<app>.yaml      Task definitions: 138 tasks / 9 AndroidLab apps
│                          (plus Gmail, TikTok, Reddit → 156 tasks / 12 apps)
└── tasks/<app>/           Per-app judge implementations
phonecli/                  Only the PhoneCLI layer MacroAgent consumes
├── prompts.py             MACRO_PLAN / MACRO_VERIFY / VLM_VERIFY + the four
│                          map-builder prompts (classify / enrich element,
│                          screen, app)
├── llm_client.py          OpenRouter-compatible text/vision completion
└── token_usage.py         Per-label token accounting
templates/                 Cloud VLM system prompts (SYSTEM_PROMPT_ANDROID_MLLM_CLOUD_V0)
utils_mobile/              AndroidController (ADB), XML tree tooling
page_executor/             VLM action executors
recorder/, tools/          Trace recording and maintenance scripts
configs/                   The two agent configs (see table above)
app_maps/                  Pre-built maps for the 9 AndroidLab apps
```

## Setup

```bash
conda activate Android-Lab          # Python 3.11 environment
export OPENROUTER_API_KEY="sk-or-v1-..."   # all configs read the key from here
```

All configs are **secret-free**: `api_key` is written as
`"${OPENROUTER_API_KEY}"` and expanded by `config_env.load_config()` at load
time. An unset variable simply expands to an empty string.

An Android 13 (API 33) AVD named `Pixel_7_Pro_API_33` is expected under
`~/.android/avd/`; `eval.py` clones it per task and restores a clean snapshot.
See `docs/prepare_for_mac.md` / `docs/prepare_for_linux.md`.

## Running

```bash
# PhoneCLI (macro) agent — configs/test_macro.yaml, app_map = Settings
python eval.py -c configs/test_macro.yaml \
  -n macro_setting_v1 --task_config evaluation/config/setting.yaml

# Baseline (PhoneCLI removed) on the same app
python eval.py -c configs/test_screen_cloud.yaml \
  -n screen_cloud_setting_v1 --task_config evaluation/config/setting.yaml

# All apps (no --task_config = every file under evaluation/config), 4 workers
python eval.py -c configs/test_macro.yaml -n macro_v1 -p 4

# Single task (debug)
python eval.py -c configs/test_macro.yaml -n debug \
  --task_config evaluation/config/setting.yaml --task_id setting_0
```

To run another app with PhoneCLI, copy `configs/test_macro.yaml` and repoint
`task.args.app_map` at `./app_maps/<app>_android.yaml`.

## Judging

```bash
python run_eval_judge.py \
  --traces logs/evaluation/macro_setting_v1 \
  --config evaluation/config/setting.yaml \
  --test-config configs/test_macro.yaml
```

Judges are per-app: programmatic XML extraction for operation-style tasks and
an LLM check for query-style tasks (`evaluation/tasks/<app>/`). Traces land in
`logs/evaluation/<name>/<task>_<ts>/` and are intentionally **not** part of this
repository.

## Rebuilding an app map

```bash
# Auto mode: clone AVD → headless boot → BFS crawl → clean up
python build_android_map.py -p com.android.settings -a Settings \
  -o ./app_maps/settings_android.yaml

# Against a running device
python build_android_map.py -p com.android.settings -a Settings \
  -o ./app_maps/settings_android.yaml --device emulator-5554 \
  --max-screens 50 --max-depth 3 --scroll-pages 3
```

`--no-classify` / `--no-enrich` skip the two LLM post-processing stages
(`ELEMENT_CLASSIFY_PROMPT` / `ELEMENT_ENRICH_PROMPT` in `phonecli/prompts.py`)
and produce smaller, unenriched maps.
