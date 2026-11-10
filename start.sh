#!/usr/bin/env bash
# HilbertFuse one-command start for Linux / macOS:  bash start.sh        (menu)
#                                                    bash start.sh predict --model models/golub --input my.csv --out out.csv
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/.hilbertfuse_ready ]; then
  echo "First run: setting up (5-10 minutes, internet needed)..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Darwin" ]; then .venv/bin/python -m pip install torch torchvision
  else .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi
  .venv/bin/python -m pip install -e ".[datasets]"
  touch .venv/.hilbertfuse_ready
fi
if [ $# -eq 0 ]; then .venv/bin/python -m hilbertfuse.cli menu; else .venv/bin/python -m hilbertfuse.cli "$@"; fi
