#!/bin/zsh
set -eu
TASK_APP_DIR="${0:A:h}"
cd "$TASK_APP_DIR"
if [[ -x "$TASK_APP_DIR/../../work/venv/bin/python" ]]; then
  TASK_PYTHON="$TASK_APP_DIR/../../work/venv/bin/python"
elif [[ -x "$TASK_APP_DIR/.venv/bin/python" ]]; then
  TASK_PYTHON="$TASK_APP_DIR/.venv/bin/python"
else
  TASK_BASE_PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3)"
  "$TASK_BASE_PYTHON" -m venv .venv
  TASK_PYTHON="$TASK_APP_DIR/.venv/bin/python"
  "$TASK_PYTHON" -m pip install -r requirements.txt
fi
exec "$TASK_PYTHON" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
