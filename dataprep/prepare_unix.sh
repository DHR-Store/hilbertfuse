#!/usr/bin/env bash
# Linux/macOS:  bash dataprep/prepare_unix.sh fit --input my_raw.csv --label-col label --out clean_data
cd "$(dirname "$0")/.."
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
exec $PY dataprep/prepare_data.py "$@"
