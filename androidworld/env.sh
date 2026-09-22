# AndroidWorld Agent environment setup.
# Source before activating/running:
#   source env.sh
#   conda activate aworld
#
# Required for OpenRouter calls: set OPENROUTER_API_KEY in your shell.
# API_KEY remains supported for compatibility.

# Linux layout. When the workspace lives on a large secondary volume, point
# AWORLD_DATA_ROOT at it so the Conda cache, environment, Android SDK and
# emulator data stay off a small /home. Defaults to the current home directory.
_AWORLD_DATA_ROOT="${AWORLD_DATA_ROOT:-$HOME}"
if [ "$(uname -s)" = "Linux" ] && [ -d "$_AWORLD_DATA_ROOT/android-sdk" ]; then
    export CONDA_PKGS_DIRS="$_AWORLD_DATA_ROOT/.conda/pkgs"
    export CONDA_ENVS_PATH="$_AWORLD_DATA_ROOT/.conda/envs"

    export ANDROID_HOME="$_AWORLD_DATA_ROOT/android-sdk"
    export ANDROID_SDK_ROOT="$ANDROID_HOME"
    export ANDROID_AVD_HOME="$_AWORLD_DATA_ROOT/.android/avd"
    # Route accessibility gRPC through `adb reverse`, so Wi-Fi tasks can turn
    # guest networking off without also losing the accessibility tree.
    export ANDROID_ENV_A11Y_GRPC_HOST="127.0.0.1"
    export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

fi
unset _AWORLD_DATA_ROOT

# Mac conda sqlite3 FTS4 workaround (Joplin tasks)
if [ -d /opt/homebrew/opt/sqlite/lib ]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/sqlite/lib:$DYLD_LIBRARY_PATH"
fi
