#!/usr/bin/env bash
# Launch start_wan22_service.py with a python that has BOTH telefuser and
# cacheseek importable (the service runs in one process needing both).
#
# Resolution order:
#   1. $PY_SVC if set (explicit override).
#   2. <telefuser_repo>/.venv for the chosen --preset — resolved by asking the
#      launcher itself (`--print-venv`, a stdlib-only path that does not import
#      telefuser/cacheseek). The venv lives inside the preset's TeleFuser repo.
#   3. Otherwise: error out with guidance — NO silent fall-through to a python
#      that lacks telefuser/cacheseek (that would ImportError and waste a run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/start_wan22_service.py"

PY="${PY_SVC:-}"
if [[ -z "${PY}" ]]; then
    # Pass "$@" so --preset is honored when resolving which venv to use.
    PY="$(python3 "${LAUNCHER}" --print-venv "$@" 2>/dev/null || true)"
fi

if [[ -z "${PY}" || ! -x "${PY}" ]]; then
    echo "[start_wan22_service] no service venv found." >&2
    echo "  Ensure the chosen preset's <telefuser_repo>/.venv exists, or set" >&2
    echo "  PY_SVC=/path/to/.venv/bin/python (the TeleFuser venv with cacheseek installed)." >&2
    exit 1
fi

exec "${PY}" "${LAUNCHER}" "$@"
