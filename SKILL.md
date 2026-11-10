---
name: pancancer-gene-image-classifier
description: Train, benchmark, explain and deploy gene-expression / omics-table classifiers with HilbertFuseNet (Hilbert-layout multi-channel gene images, MI + redundancy-pruned genes, SE attention DenseNet, optional ensemble and linear-blend hybrid) and Guided Grad-CAM gene ranking, extending the MI_DenseNetCAM paper (Wang et al. 2021). Use whenever the user wants a CNN or deep-learning classifier on RNA-seq or microarray tables, a leak-free comparison against logistic regression/SVM, candidate marker genes from a trained model, or to reproduce, critique or improve the Frontiers in Genetics pan-cancer paper, even if they do not name the method.
---

# HilbertFuseNet gene-image classifier (package `hilbertfuse`)

Docs: `docs/RESEARCH_HILBERTFUSE.md` (problem, design, measured results), `docs/PAPER_ANALYSIS.md`, `docs/COMMANDS.md`.

## Pipeline
`table → (log2) → Min-Max → MI top-N (+ redundancy pruning) → cluster order on a Hilbert curve → 3-channel image (expression | MI prior | rank) → HilbertFuseNet → probabilities → Guided Grad-CAM genes`.
Presets: `fuse` (default, CNN + linear), `fuse_ens`, `hfnet` (CNN only), `hfnet_ens`, `paper` (paper-style control).

## Workflow
1. Verify data layout (rows = samples, columns = genes, `label` column; `--transpose` for genes x samples); split by patient; keep test untouched.
2. Install: `pip install torch torchvision` (CPU wheel index is much smaller) then `pip install -e .`.
3. **Always benchmark against tuned classical baselines before recommending a deep model**: `hilbertfuse benchmark --datasets <names> --methods lr_mi,svm_mi,lr_all,paper,hfnet+fuse --folds 5 --out bench` (datasets: `hilbertfuse datasets`; own data: use `cv`). Report mean ± std over folds, macro-F1 and the paired-Wilcoxon lines in `bench/summary.md`.
4. Train the winner: `hilbertfuse train --train TRAIN --test TEST --preset fuse --n-genes 1024 --out runs/m1` (small data: `--epochs 80`; GPU: `--device cuda`).
5. Evaluate (`evaluate`), predict (`predict`), explain (`explain` → per-class and global gene CSVs).

## Rules
- Measured (7 real cancer datasets, 5-fold leak-free CV): `fuse` (HilbertFuse = CNN + tuned logistic regression) ~ best tuned linear baseline (mean acc .941 vs .947/.941, not significantly different) and +8.5 pts over the paper-style CNN; the CNN alone (`hfnet`) is worse than logistic regression (.860 vs .941) and unstable on some datasets. Never claim a deep model beats linear models without running `hilbertfuse benchmark` on the user's data. Details: `docs/RESEARCH_HILBERTFUSE.md`.
- Gene lists are model attributions, not proven biomarkers; validate with DE tests/enrichment (FDR) and external cohorts.
- Report macro-F1 and per-class accuracy, not accuracy alone. Research use only.

## Problems
| Symptom | Action |
|---|---|
| "labels not seen in training" | test has a class absent from train |
| slow on CPU | `--n-genes 576 --epochs 40`, fewer folds |
| pretrained download fails | `hilbertfusenet121` falls back to random init with a warning |
| CNN unstable on tiny data | use `--preset fuse` (linear member stabilises) or `lr_mi` |
