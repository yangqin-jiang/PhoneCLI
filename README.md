# PhoneCLI

**Compiles an app's GUI navigation into callable commands** — with no app-internal
API, no runtime instrumentation, and no model training.

Mobile GUI agents run a perception–action loop: screenshot the device, invoke a
vision–language model (VLM), emit an action. It is slow, costly and brittle — yet
most of what those steps do is navigation, and everyday navigation is static,
ordered and endlessly repeated. PhoneCLI turns that repeated work into a compiled
artifact.

- **Offline**, PhoneCLI explores a target app from the outside and distills its
  screens, interactive elements and navigation edges into a semantically annotated
  **app map**. Every screen in that map yields one deterministic command: the replay
  sequence that reaches it.
- **Online**, the agent selects a command, verifies it against the task before
  execution, then executes it deterministically in sub-second time at **zero VLM
  cost**. Open-ended interaction — and every failure of the compiled path — falls
  back to the embedded VLM interpreter, which is exactly the pure VLM agent, so
  compilation can only help.

On AndroidLab, PhoneCLI improves the task success rate while reducing steps and
token consumption, and it transfers to AndroidWorld's official M3A agent with
consistent efficiency gains. What it compiles is the app's *navigation* rather than
one run, so it serves new tasks — not only repeated ones.

This repository contains the two benchmark harnesses used to evaluate it, each
paired with its no-PhoneCLI baseline:

| Benchmark | Baseline (no PhoneCLI) | PhoneCLI |
|---|---|---|
| **AndroidLab** — 9 apps / 138 tasks | `ScreenshotCloudTask` — pure VLM loop, no app map, no replay | `MacroAgentTask` — app-map routing + ADB macro replay + VLM fallback |
| **AndroidWorld** — 116 tasks | `m3a.M3A` — the official Set-of-Mark + VLM baseline | `M3AMapAgent` — `m3a.M3A` + app-map macro routing |

```
androidlab/     AndroidLab harness      → androidlab/README.md
androidworld/   AndroidWorld harness    → androidworld/README.md
```

Both directories are self-contained: `cd` into one before running anything.

---

## AndroidLab

`evaluation/macro_agent.py` (`MacroAgentTask`) is the PhoneCLI agent;
`evaluation/evaluation.py` (`ScreenshotCloudTask`) is the baseline. Nine pre-built
app maps ship in `app_maps/`.

```bash
cd androidlab
conda activate Android-Lab                    # Python 3.11
pip install -r requirements.txt               # first time only
export OPENROUTER_API_KEY="sk-or-v1-..."

# PhoneCLI (app_map points at Settings in this config)
python eval.py -c configs/test_macro.yaml \
  -n macro_v1 --task_config evaluation/config/setting.yaml

# Baseline — same task, no app map, no replay
python eval.py -c configs/test_screen_cloud.yaml \
  -n screen_cloud_v1 --task_config evaluation/config/setting.yaml

# Score the recorded traces
python run_eval_judge.py --traces logs/evaluation/macro_v1 \
  --config evaluation/config/setting.yaml --test-config configs/test_macro.yaml

# Rebuild an app map from scratch
python build_android_map.py -p com.android.settings -a Settings \
  -o ./app_maps/settings_android.yaml
```

Omit `--task_config` to run every app; add `-p 4` for four parallel workers.
To target another app with PhoneCLI, copy `configs/test_macro.yaml` and repoint
`task.args.app_map` at that app's map. Default AVD: `Pixel_7_Pro_API_33` (API 33).

## AndroidWorld

`m3a_map_agent.py` (`M3AMapAgent`) subclasses the official `m3a.M3A` and prepends
the map layer; `run.py` drives it, `run_m3a.py` drives the untouched M3A baseline.
`app_maps/androidworld_22/` holds 22 validated maps.

```bash
cd androidworld
bash setup.sh                 # clones the pinned AndroidWorld revision,
                              # applies patches/android_world.patch, pip install -e .
source env.sh
conda activate aworld
export OPENROUTER_API_KEY="sk-or-v1-..."

# M3A + PhoneCLI
python run.py --model qwen/qwen3.7-plus \
  --console-port 5554 --grpc-port 8554 \
  --app-maps ./app_maps/androidworld_22 --tasks SystemWifiTurnOn,SystemWifiTurnOff

# M3A baseline (use a different emulator when running both at once)
python run_m3a.py --model qwen/qwen3.7-plus \
  --console-port 5556 --grpc-port 8556 --tasks SystemWifiTurnOn,SystemWifiTurnOff
```

Drop `--tasks` for the full 116-task suite. `scripts/run_full_eval.py --method
m3a|macro` is a secret-safe launcher that keeps the API key out of argv, and
`scripts/summarize_checkpoints.py` turns raw checkpoints into CSV/JSON.
Default AVD: `AndroidWorldAvd` (Pixel 6, API 33).

