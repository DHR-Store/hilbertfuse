# HilbertFuse — cancer gene-expression classification you can clone and run

Pretrained models for 8 real cancer data sets, a quick way to train your own model on a CPU, and honest, leak-free scores. A leak-free, layout-aware CNN (**HilbertFuseNet**) fused with a tuned logistic regression (**HilbertFuse**), built on the ideas of *Wang et al., Front. Genet. 2021* (MI_DenseNetCAM).

**Research use only. The models recognise which known class an expression profile resembles. They do not find the cause of a cancer and must not be used to diagnose or treat patients.**

## Start in one step
```powershell
git clone https://github.com/<your-user>/hilbertfuse.git
cd hilbertfuse
.\start_windows.bat            # Windows (double-click works too)   |   Linux/macOS: bash start.sh
```
First run installs the packages (5-10 min, internet once, Python 3.10-3.12). Then a menu opens: **2 predict**, **3 evaluate**, **4 train my own model (choose quick / fast / full)**, **5 cross-validation**, **6 explain genes**, **7 show all scores**.
Predictions run offline on your computer. Keep patient files in `data_private/` (never committed). VS Code users: `docs/VSCODE.md`.

## Use a pretrained model on your cancer / patient data
Your CSV: one row per sample, one column per gene, the **same gene columns** the model was trained on (`models/<name>/required_genes.txt`). The tool refuses files with the wrong genes.
```powershell
hilbertfuse predict  --model models\golub --input my_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\golub --test my_labeled_samples.csv
```
Output per sample: predicted class, **confidence**, runner-up class and an **UNCERTAIN** flag below 0.70. Low confidence means "do not trust this answer".

## Train your own model quickly on CPU
```powershell
hilbertfuse train --train my_data.csv --label-col label --preset linear --out models\mymodel   # seconds
hilbertfuse train --train my_data.csv --label-col label --preset fast   --out models\mymodel   # ~1-3 min (CNN + linear)
hilbertfuse cv --data my_data.csv --folds 5 --preset linear                                    # honest accuracy first
```

## Scores (paper-style Table 2: mean over 8 real data sets, 5-fold cross-validation)
### TABLE 2 | The experimental results of the methods (mean over 8 real data sets, 5-fold CV).

| Method | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 89.29% | 90.33% | 89.29% | 89.02% |
| HilbertFuseNet (CNN only) | 81.48% | 85.79% | 81.48% | 81.17% |
| Paper-style DenseNet | 81.14% | 82.61% | 81.14% | 80.51% |
| MI_LogReg | 90.19% | 90.76% | 90.19% | 89.86% |
| LogReg (all genes) | 90.30% | 90.22% | 90.30% | 89.53% |
| MI_SVM | 88.09% | 90.41% | 88.09% | 88.26% |
| MI_KNN | 86.72% | 88.76% | 86.72% | 86.61% |

*Each data set counts equally; precision/recall/F1 are class-weighted as in the paper.*

Per pretrained model (accuracy, 5-fold CV):

| model | task | accuracy (5-fold CV) | macro-F1 | verdict |
|---|---|---|---|---|
| khan | 4 small round blue cell tumors (SRBCT) | 1.000 ± 0.000 | 1.000 | usable for research |
| gordon | lung: mesothelioma vs adenocarcinoma | 1.000 ± 0.000 | 1.000 | usable for research |
| su | 4 carcinoma tissue types | 0.952 ± 0.067 | 0.947 | usable for research |
| golub | ALL vs AML leukemia | 0.945 ± 0.056 | 0.940 | usable for research |
| singh | prostate tumor vs normal | 0.932 ± 0.064 | 0.932 | usable for research |
| shipp | DLBCL vs follicular lymphoma | 0.920 ± 0.119 | 0.898 | usable for research |
| alon | colon tumor vs normal | 0.794 ± 0.099 | 0.764 | usable for research |
| pomeroy | CNS embryonal tumor outcome (hard) | 0.600 ± 0.160 | 0.539 | **NOT useful** (no better than guessing) |

All tables (Tables 1-6 in the paper's format, plus one per model): `docs/PAPER_STYLE_TABLES.html`. What the numbers mean: `docs/RESEARCH_HILBERTFUSE.md`.

### Honest findings
* **HilbertFuse vs the best tuned classical method** (all-gene logistic regression): mean accuracy 89.3% vs 90.3%: -1.0 points, not significantly different (paired Wilcoxon p=0.49), better in 9 / worse in 9 of 40 folds.
* **HilbertFuse vs the paper-style DenseNet:** 89.3% vs 81.1%: +8.1 points, significantly better (p<0.001), better in 23 / worse in 2 of 40 folds.
* **CNN alone (HilbertFuseNet) vs MI logistic regression:** 81.5% vs 90.2%: -8.6 points, significantly worse (p=0.001), better in 6 / worse in 21 of 40 folds.
* **CNN alone vs the paper-style DenseNet (do the new design ideas help the CNN?):** 81.5% vs 81.1%: +0.3 points, not significantly different (paired Wilcoxon p=0.77), better in 18 / worse in 14 of 40 folds.
* The CNN alone beats MI logistic regression on khan, singh (of 8).
* The CNN alone is the least stable method: fold-to-fold standard deviation up to 23.3 points (on gordon).
* Models judged **not useful** (no better than guessing the most common class): `pomeroy` (CV accuracy 0.600 vs 0.650 for always guessing the common class).
* Number of genes (logistic regression, 100 to 4096 genes): accuracy varies only between 89.5% and 91.0%, far less than the 3.9-point swing the paper reports across image sizes with a CNN.
* Caveat: small cohorts (60-181 samples), one random seed, one platform per data set, networks trained from random weights; results do not automatically transfer to large cohorts such as TCGA.

## Prepare your own raw data (clean + normalise)
`dataprep/` cleans and normalises a raw expression file leak-free and makes a train/test pair: `python dataprep\\prepare_data.py fit --input my_raw.csv --label-col label --out clean_data` (see `dataprep/README.md`).

## Heat maps
`hilbertfuse heatmap --model models\\golub --data my_samples.csv --out heatmaps` draws the Guided Grad-CAM gene-image map, the top-gene expression heat map and class x gene importance (menu option 8). See `docs/HEATMAPS.md`.

## Documentation
`docs/TECH_AND_ALGORITHM_FOR_PPT.md` (algorithm details, slide by slide) · `docs/USE_MODELS.md` · `docs/TRAIN_ON_YOUR_PC.md` · `docs/COMMANDS.md` · `docs/GIT_AND_SHARING.md` · `docs/PAPER_ANALYSIS.md` · model cards in `models/*/MODEL_CARD.md`.

Data sets in `datasets/` are public research data (citations in `docs/RESEARCH_HILBERTFUSE.md`). MIT license.
