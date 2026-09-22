#!/bin/bash
# One-time setup for aworld_agent.
# Run: bash setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== aworld_agent setup ==="

# Configure the optional AWORLD_DATA_ROOT cache paths before creating the environment.
source "$REPO_ROOT/env.sh"

ANDROID_WORLD_COMMIT="3e50888527ef9f29b9157ecd537e408008bb1c85"
ANDROID_WORLD_PATCH="$REPO_ROOT/patches/android_world.patch"

# 1. Clone AndroidWorld benchmark framework
if [ ! -d "android_world" ]; then
    echo "Cloning android_world..."
    git clone https://github.com/google-research/android_world.git
else
    echo "android_world/ already exists"
fi

# Pin the dependency and apply the small integration patch used by this repo.
if git -C android_world apply --reverse --check "$ANDROID_WORLD_PATCH" >/dev/null 2>&1; then
    echo "AndroidWorld integration patch already applied"
else
    if ! git -C android_world diff --quiet || ! git -C android_world diff --cached --quiet; then
        echo "android_world/ has unrelated local changes; refusing to overwrite them"
        exit 1
    fi
    git -C android_world fetch origin "$ANDROID_WORLD_COMMIT"
    git -C android_world checkout --detach "$ANDROID_WORLD_COMMIT"
    git -C android_world apply "$ANDROID_WORLD_PATCH"
    echo "Applied AndroidWorld integration patch"
fi

# 2. Create conda environment
if conda env list | grep -q "^aworld "; then
    echo "conda env 'aworld' already exists"
else
    echo "Creating conda environment 'aworld' (Python 3.11)..."
    conda create -n aworld python=3.11 -y
fi

# 3. Install dependencies
echo "Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate aworld
pip install -e android_world/
pip install -e .
pip install backoff openai jsonlines lxml Pillow pyyaml python-dotenv \
    opencv-python numpy pandas fuzzywuzzy python-Levenshtein \
    termcolor tenacity pydub immutabledict IPython requests \
    tiktoken flask nltk pyshine colorama sounddevice

echo ""
echo "=== Setup complete ==="
echo "Run: conda activate aworld && source env.sh && python run.py --help"
