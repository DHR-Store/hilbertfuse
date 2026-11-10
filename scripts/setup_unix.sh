#!/usr/bin/env bash
# Linux / macOS setup + first real-data run:  bash scripts/setup_unix.sh
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # macOS: plain `pip install torch torchvision`
pip install -e ".[dev,datasets]"
pytest -q tests
hilbertfuse real-demo --dataset khan --out runs/real_demo
