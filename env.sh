# Source this file (bash): `source env.sh`. Sets every path the project uses; no side effects besides exports.
CLVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$CLVLA_ROOT/.env" ]; then _envfile="$CLVLA_ROOT/.env"; else _envfile="$CLVLA_ROOT/.env.example"; fi
set -a
# shellcheck disable=SC1090
source <(sed -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]\+#[^"]*$//' -e '/^[[:space:]]*$/d' "$_envfile")
set +a
unset _envfile
export CLVLA_ROOT
export OPENPI_DIR="$CLVLA_ROOT/third_party/openpi"
export LIBERO_DIR="$OPENPI_DIR/third_party/libero"
export CLVLA_POLICY_PY="$OPENPI_DIR/.venv/bin/python"
export CLVLA_SIM_PY="$OPENPI_DIR/examples/libero/.venv/bin/python"
# Project-local LIBERO config (bddl/init/assets point into the pinned submodule); avoids ~/.libero from other projects.
export LIBERO_CONFIG_PATH="$CLVLA_ROOT/.libero"
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"
