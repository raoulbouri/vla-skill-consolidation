#!/usr/bin/env bash
# Reproduce both environments from scratch (idempotent; safe to rerun). No Docker needed.
#
#   policy env: third_party/openpi/.venv                (Python 3.11; OpenPI `uv sync` from its uv.lock)
#   sim env:    third_party/openpi/examples/libero/.venv (Python 3.8; OpenPI's documented non-Docker LIBERO setup)
#               + env/sim-extra-requirements.txt        (h5py, pytest: demo loading and tests in the sim env)
#
# Usage: bash scripts/setup_envs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

OPENPI_URL=https://github.com/Physical-Intelligence/openpi.git
OPENPI_SHA=215abfb217dbac7d5f1273282331b9b1866c0479   # pinned 2026-10-22 (main @ 2026-08-24)
LIBERO_SHA=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c   # OpenPI's third_party/libero submodule pin (official LIBERO)

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$OPENPI_DATA_HOME" "$CLVLA_OUTPUT_ROOT" "$ROOT/logs"

# 1. OpenPI + submodules at pinned revisions
if [ ! -d "$OPENPI_DIR/.git" ] && [ ! -f "$OPENPI_DIR/.git" ]; then
  if [ -f "$ROOT/.gitmodules" ] && git -C "$ROOT" rev-parse >/dev/null 2>&1; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT" submodule update --init third_party/openpi
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone "$OPENPI_URL" "$OPENPI_DIR"
  fi
fi
git -C "$OPENPI_DIR" checkout -q "$OPENPI_SHA"
GIT_LFS_SKIP_SMUDGE=1 git -C "$OPENPI_DIR" submodule update --init --recursive
test "$(git -C "$OPENPI_DIR" rev-parse HEAD)" = "$OPENPI_SHA"
test "$(git -C "$LIBERO_DIR" rev-parse HEAD)" = "$LIBERO_SHA"
test "$(git -C "$LIBERO_DIR" config --get remote.origin.url)" = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
echo "[setup] openpi $OPENPI_SHA, libero $LIBERO_SHA"

# 2. Policy env (OpenPI README: GIT_LFS_SKIP_SMUDGE=1 uv sync; uv.lock pins every package)
(cd "$OPENPI_DIR" && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen)

# 3. Simulator env (OpenPI examples/libero/README.md "Without Docker")
if [ ! -x "$CLVLA_SIM_PY" ]; then
  (cd "$OPENPI_DIR" && uv venv --python 3.8 examples/libero/.venv)
fi
(cd "$OPENPI_DIR" && export VIRTUAL_ENV="$OPENPI_DIR/examples/libero/.venv" &&
  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match &&
  uv pip install -e packages/openpi-client &&
  uv pip install -e third_party/libero &&
  uv pip install -r "$ROOT/env/sim-extra-requirements.txt")

# 4. Project-local LIBERO path config (LIBERO otherwise prompts interactively and writes ~/.libero)
L="$LIBERO_DIR/libero/libero"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $LIBERO_DATA_ROOT
assets: $L/assets
EOF

# 5. Record exact environments
(cd "$OPENPI_DIR" && VIRTUAL_ENV="$OPENPI_DIR/.venv" uv pip freeze) > "$ROOT/env/policy-env.freeze.txt"
(cd "$OPENPI_DIR" && VIRTUAL_ENV="$OPENPI_DIR/examples/libero/.venv" uv pip freeze) > "$ROOT/env/sim-env.freeze.txt"
cp "$OPENPI_DIR/uv.lock" "$ROOT/env/openpi-uv.lock"
"$CLVLA_POLICY_PY" --version; "$CLVLA_SIM_PY" --version
echo "[setup] done. Next: scripts/py-policy scripts/validate_assets.py"
