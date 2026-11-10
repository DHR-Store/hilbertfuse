#!/usr/bin/env bash
# Reproduce everything: tests -> full benchmark on real data -> report. ~3-4 h on 1 CPU core, ~20 min on a GPU.
set -e
pip install -e ".[dev,datasets]"
pytest -q tests
hilbertfuse datasets                                    # list data (8 bundled in ./datasets)
hilbertfuse benchmark --datasets khan,alon,golub,shipp,su,singh,gordon \
    --methods lr_mi,svm_mi,lr_all,paper,hfnet+fuse --folds 5 --n-genes 1024 --epochs 50 --out bench
# optional ablations (what each design choice contributes):
hilbertfuse benchmark --datasets khan,alon,golub,shipp,su \
    --methods abl_rowmajor,abl_1ch,abl_mi,abl_noSE --folds 5 --out bench
python scripts/make_report.py bench/results.csv docs/RESULTS.md
