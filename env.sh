# Source this file (bash): `source env.sh`. Sets every path the project uses; no side effects besides exports.
CLVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Defaults from .env.example, then local overrides from .env (so variables added later reach existing .env setups).
_caller_gpu="${CLVLA_GPU:-}"   # an explicit CLVLA_GPU from the caller wins over .env (per-process GPU pinning)
set -a
for _envfile in "$CLVLA_ROOT/.env.example" "$CLVLA_ROOT/.env"; do
  [ -f "$_envfile" ] || continue
  # shellcheck disable=SC1090
  source <(sed -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]\+#[^"]*$//' -e '/^[[:space:]]*$/d' "$_envfile")
done
set +a
unset _envfile
[ -n "$_caller_gpu" ] && CLVLA_GPU="$_caller_gpu"
unset _caller_gpu
export CLVLA_ROOT
export OPENPI_DIR="$CLVLA_ROOT/third_party/openpi"
export LIBERO_DIR="$OPENPI_DIR/third_party/libero"
export CLVLA_POLICY_PY="$OPENPI_DIR/.venv/bin/python"
export CLVLA_SIM_PY="$OPENPI_DIR/examples/libero/.venv/bin/python"
# Project-local LIBERO config (bddl/init/assets point into the pinned submodule); avoids ~/.libero from other projects.
export LIBERO_CONFIG_PATH="$CLVLA_ROOT/.libero"
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"
