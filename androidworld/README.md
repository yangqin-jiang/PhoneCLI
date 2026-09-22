# M3A + PhoneCLI on AndroidWorld

This repository compares an **M3A baseline** with an **M3A + PhoneCLI** agent on
the [AndroidWorld](https://github.com/google-research/android_world) benchmark.
The PhoneCLI variant keeps M3A's prompt / action space / SoM machinery intact and
prepends an app-map macro routing layer (deterministic ADB replay of the
navigation path to the target screen).

## The two variants

| Variant | Entry point | Agent class | Map layer |
|---------|-------------|-------------|-----------|
| **M3A baseline** | `run_m3a.py` | `android_world.agents.m3a.M3A` | none — pure SoM + VLM loop |
| **M3A + PhoneCLI** | `run.py` | `M3AMapAgent` (`m3a_map_agent.py`) | Round 1: app-map routing → macro replay → landing check → hand off to the unmodified M3A loop |

`M3AMapAgent` subclasses `m3a.M3A`; when no macro matches it falls straight
through to vanilla M3A, so the PhoneCLI layer can only add navigation shortcuts,
never change the base policy. `run.py` additionally exposes the full PhoneCLI
map-layer switches: `--force-macro-vlm`, `--skip-landing-check`,
`--landing-hint-level {0,1,2}`, `--cold-read-warmup`.

## Repository layout

```text
run.py                           M3A + PhoneCLI (macro + map) evaluation runner
run_m3a.py                       M3A baseline runner
m3a_map_agent.py                 M3AMapAgent: m3a.M3A + app-map macro routing
agent.py                         Legacy PhoneCLI SoM agent (superseded by M3AMapAgent)
app_map.py / build_map*.py       App-map runtime + BFS map builder
prompts.py / llm_client.py       Shared prompt templates + OpenRouter client
token_usage.py                   Per-label token accounting
and_controller.py, utils_adb.py, utils_mobile.py, xml_tool.py,
special_check.py, packages.py, docker_utils.py    Device/ADB/XML helpers
build_map.py                     Single-app map builder
scripts/build_all_maps.py        Parallel 22-app map builder
scripts/run_full_eval.py         Secret-safe evaluation launcher (--method m3a|macro)
scripts/summarize_checkpoints.py Checkpoint → CSV/JSON summaries
tests/test_regressions.py        Unit + regression tests for the agent layer
app_maps/androidworld_22/        Validated 22-app, 50-screen map set
patches/android_world.patch      Required AndroidWorld integration patch
pyproject.toml                   Editable `aworld_agent` package mapping
```

The `android_world/` benchmark checkout is **not** vendored here: `setup.sh`
clones the pinned revision and applies `patches/android_world.patch`.

## Setup

The project pins AndroidWorld to commit
`3e50888527ef9f29b9157ecd537e408008bb1c85`. `setup.sh` clones that revision,
applies `patches/android_world.patch`, and installs the editable dependency.

```bash
bash setup.sh
source env.sh
conda activate aworld
export OPENROUTER_API_KEY='your-key'
```

`config.yaml` is deliberately secret-free. Export the key instead (or keep a
local `config.local.yaml`, which is Git-ignored). All runners read
`OPENROUTER_API_KEY` / `API_KEY` from the environment.

The AndroidWorld patch provides two integration hooks:

- per-task map selection and token accounting callbacks;
- accessibility gRPC over `adb reverse`, so Wi-Fi tasks do not destroy the
  agent's observation channel when guest Wi-Fi is disabled.

## Build maps

Build one Settings map on an existing emulator:

```bash
python build_map.py \
  --package com.android.settings \
  --app Settings \
  --device emulator-5554 \
  --output app_maps/settings_m100_d3_s3.yaml \
  --config config.yaml \
  --max-screens 100 \
  --max-depth 3 \
  --scroll-pages 3
```

Build all 22 mapped apps with two emulators:

```bash
python scripts/build_all_maps.py \
  --devices emulator-5554 emulator-5556 \
  --output-dir app_maps/androidworld_22 \
  --config config.yaml \
  --adb "$ANDROID_HOME/platform-tools/adb" \
  --max-screens 50 \
  --max-depth 3 \
  --scroll-pages 3
```

`max_depth` is BFS navigation depth, not the agent's episode step budget. For
the Settings Wi-Fi, Bluetooth, and brightness tasks, depth 3 already reaches
the relevant pages; increasing it does not fix slider precision or stale
toggle-state observations.

## Run evaluations

Settings task set used in the recorded comparison:

```bash
TASKS='SystemBluetoothTurnOff,SystemBluetoothTurnOn,SystemBrightnessMax,SystemBrightnessMin,SystemWifiTurnOff,SystemWifiTurnOn'
```

Macro+Map:

```bash
python scripts/run_full_eval.py \
  --method macro \
  --config config.yaml \
  --app-maps app_maps/androidworld_22 \
  --checkpoint-dir artifacts/checkpoints/settings/macro_map \
  --console-port 5554 --grpc-port 8554 \
  --adb-path "$ANDROID_HOME/platform-tools/adb" \
  --tasks "$TASKS"
```

M3A baseline (use a different emulator when running concurrently):

```bash
python scripts/run_full_eval.py \
  --method m3a \
  --config config.yaml \
  --app-maps app_maps/androidworld_22 \
  --checkpoint-dir artifacts/checkpoints/settings/m3a \
  --console-port 5556 --grpc-port 8556 \
  --adb-path "$ANDROID_HOME/platform-tools/adb" \
  --tasks "$TASKS"
```

Omit `--tasks` for the full AndroidWorld suite. Raw checkpoints, screenshots,
and logs belong under `artifacts/` (Git-ignored); only small summaries belong
under `results/`. Neither directory is shipped in this repository — experiment
data lives outside it.

## Outputs and token accounting

Each completed run writes:

```text
<checkpoint-dir>/
├── <TaskName>_0.pkl.gz
├── token_usage.json
├── token_usage_per_task.jsonl
└── run_summary.json              # Macro+Map runner
```

Both runners now use the project's `TokenUsage` fields: prompt, completion,
total, cache-read, cache-write, total calls, and per-label usage. Historical
M3A runs made before this change have `0` in `token_usage.json`; that means the
wrapper did not record usage, not that the calls were free.

To turn raw checkpoint metadata into reviewable CSV/JSON:

```bash
python scripts/summarize_checkpoints.py \
  --run macro=artifacts/checkpoints/settings/macro_map \
  --run m3a=artifacts/checkpoints/settings/m3a \
  --output-dir results/settings_latest
```

## Tests

```bash
pytest tests/test_regressions.py
```

## Troubleshooting

- `device 'emulator-5554' not found`: start the matching AVD and verify it with
  `adb devices`; console and gRPC ports must match the runner.
- `Could not get a11y tree`: source `env.sh`, verify
  `ANDROID_ENV_A11Y_GRPC_HOST=127.0.0.1`, and rerun `setup.sh` so the
  AndroidWorld patch is applied.
- OpenRouter `invalid model ID`: the tested multimodal model is
  `qwen/qwen3.7-plus`.
- Empty OpenRouter response content: reasoning is disabled for this model in
  strict classifier/action calls and the request is retried once.
