# Using the trained models (and sharing your own)

## 1. What is in `models\`
Eight models, each trained on 100 % of one **real** published cancer expression data set. Open `models\README.md` for the table and `models\<name>\MODEL_CARD.md` for each model's honest, cross-validated accuracy.

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

### See all scores any time
```powershell
hilbertfuse scores                       # table of every model's stored cross-validated scores (or menu option 7)
python scripts\make_report.py results\benchmark_results.csv docs\RESULTS.md   # full benchmark tables + significance tests
```
To re-measure a model's score yourself (takes minutes): `hilbertfuse benchmark --datasets golub --methods lr_mi,hfnet+fuse --folds 5 --out bench_check`.
Do **not** score a model on the data it was trained on (it will look near-perfect and means nothing); use new labeled samples with menu option 3 / `hilbertfuse evaluate`.

## 2. Give a model new data and get predictions
Easiest: double-click `start_windows.bat` (or `bash start.sh`) and choose 2 (predict) or 3 (evaluate). Or use the commands:
Your file: one row per sample, one column per gene (same column names as the model's `required_genes.txt`), optional `label` column.
```powershell
hilbertfuse predict  --model models\golub --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\golub --test your_labeled_samples.csv
```
* `predict` writes the predicted class and the probability of every class per sample.
* `evaluate` needs a `label` column and prints accuracy, macro-F1, per-class accuracy and the confusion matrix.
* Both first print **gene coverage** and **refuse** files that do not contain the model's genes (a wrong file would give meaningless answers). Do not evaluate on the samples the model was trained on: that score is meaningless.

## 3. The most important limit (read this)
A model only works on data that looks like its training data:
* `golub`, `gordon` (and `alon`, `pomeroy`) use Affymetrix probe IDs: they work on other data from the **same array platform with the same IDs and the same normalisation**.
* `khan`, `shipp`, `singh`, `su` have anonymous positional column names: they work **only** on files with exactly the same columns in the same order.
* Accuracy on another hospital, platform or processing pipeline is unknown and usually lower (batch effects). Only the classes listed in the card can be answered; any other tumor type is forced into one of them.

## 4. What these models can and cannot tell you
* They recognise **which known class an expression profile resembles**. They do **not** identify the *cause* of a cancer, and they must not be used to diagnose or make decisions about a patient. That would need large multi-hospital training data, prospective validation and regulatory approval.
* `hilbertfuse explain` lists genes the model relied on; these are candidates for follow-up (differential-expression tests, enrichment with FDR correction, independent cohorts), not proof of cause.

## 5. Train a final model on YOUR data and share it
```powershell
hilbertfuse cv --data all_samples.csv --folds 5 --preset fuse                       # honest accuracy first
hilbertfuse train --train all_samples.csv --label-col label --out models\mymodel     # then train on everything
Compress-Archive models\mymodel mymodel.zip                                          # send this to others
```
Others unzip it and run `hilbertfuse predict --model mymodel --input their.csv` (they need this package installed). Only load model folders from people you trust (they contain pickled files). Write down the accuracy from `cv` in a model card, as in `models\*\MODEL_CARD.md`.

## 6. Bigger real data: TCGA (5 tumor types, 801 samples)
Download `data.csv` and `labels.csv` from https://archive.ics.uci.edu/dataset/401 , then:
```powershell
python scripts\prepare_tcga_uci.py --data data.csv --labels labels.csv --out tcga5
hilbertfuse cv --data tcga5\all.csv.gz --folds 5 --preset fuse --n-genes 1024
hilbertfuse train --train tcga5\train.csv.gz --test tcga5\test.csv.gz --out models\tcga5
```
This could not be tested in my environment (no access to that site), so check the output carefully. The UCI file also uses anonymous gene columns; for a model other labs can use, start from data with real gene symbols (for example UCSC Xena / GDC TCGA tables) and keep the same normalisation.
