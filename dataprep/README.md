# dataprep — clean and normalise your own gene-expression data before training

One tool (`prepare_data.py`) turns a raw expression file into a clean, correctly normalised **train / test** pair that `hilbertfuse` can train on, plus a **report** of every problem it found, plus a saved **prep model** so new patient files get *exactly the same* cleaning later.

**Why it is safe:** everything that learns from the data (which genes to drop, the values used to fill gaps, the quantile reference, z-score statistics) is fitted on the **training part only** and then applied to the test part and to new files. The split is made first; test data never influences training. Samples of one patient (`--group-col`) always stay on the same side.

**What it cannot do (be honest with yourself):** cleaning removes artefacts; it cannot fix wrong labels, too few samples, or a batch/lab effect that is tied to the disease class. The report warns about these. A "better dataset" here means *cleaner and comparable*, not *more informative*.

## 1. Setup (once)
```powershell
cd C:\path\to\hilbertfuse                 # your project folder (the .venv from start_windows.bat already has everything)
.\.venv\Scripts\Activate.ps1
# standalone use without the project: pip install -r dataprep\requirements.txt
```

## 2. The three commands (PowerShell / VS Code terminal)
```powershell
# 1) look at your raw file first - changes nothing, prints problems
python dataprep\prepare_data.py check --input my_raw.csv --label-col label

# 2) clean + normalise + split  ->  clean_data\train.csv.gz, test.csv.gz, prep_report.md, prep_model.joblib
python dataprep\prepare_data.py fit --input my_raw.csv --label-col label --out clean_data

# 3) later: prepare NEW samples (patients) with the same cleaning
python dataprep\prepare_data.py apply --prep clean_data\prep_model.joblib --input new_raw.csv --out new_clean.csv
```
Double-click style: `dataprep\prepare_windows.bat fit --input my_raw.csv --label-col label --out clean_data` (Linux/macOS: `bash dataprep/prepare_unix.sh ...`).

**Try it now** on the included messy example (real Golub leukemia data with added mess):
```powershell
python dataprep\prepare_data.py check --input dataprep\example_messy_raw.csv --label-col label --id-col patient_id --batch-col batch
python dataprep\prepare_data.py fit   --input dataprep\example_messy_raw.csv --label-col label --id-col patient_id --group-col patient_id --batch-col batch --out clean_example
```

## 3. Then train with hilbertfuse
```powershell
hilbertfuse cv    --data clean_data\train.csv.gz --label-col label --folds 5 --preset linear      # honest accuracy first
hilbertfuse train --train clean_data\train.csv.gz --test clean_data\test.csv.gz --label-col label --preset linear --out models\mymodel
python dataprep\prepare_data.py apply --prep clean_data\prep_model.joblib --input new_patients_raw.csv --out new_clean.csv
hilbertfuse predict --model models\mymodel --input new_clean.csv --out predictions.csv
```
> **Important:** the 8 pretrained models in `models\` were trained on the *original* public files. Give them data in that original format, **not** prepared data. `dataprep` is for training **your own** models (and for preparing new samples for those models).

## 4. What it does, in order
| # | Step | Detail |
|---|---|---|
| 1 | Read | CSV / TSV / Parquet (.gz ok); `--transpose` if genes are rows. ID / group / batch columns are kept out of the genes; non-numeric columns are dropped and reported. |
| 2 | Gene names | trims spaces; optional `--strip-version` (ENSG..1.7 -> ENSG..1) and `--uppercase-genes`; **duplicated gene columns are averaged**. |
| 3 | Labels | trims spaces; merges spellings that differ only by case/spaces (`ALL`, ` all`) into the most common one (`--label-case`); optional `--label-map old,new` CSV; drops missing/unknown labels; drops classes with fewer than `--min-class-size` (5) samples. |
| 4 | Samples | drops samples with > 20 % missing values; removes **exact duplicates** (one copy kept); drops samples whose identical values carry **conflicting labels** (label error). |
| 5 | Split | stratified `--test-size` (0.2); by patient if `--group-col`; `--no-split` to only clean. |
| 6 | Fit on train | genes with > 20 % missing dropped; remaining gaps filled with the **training median**; constant and near-constant genes dropped; data type detected (below); normalisation fitted. |
| 7 | Apply | the same transformation is applied to test and to any new file. Test samples are never removed. |
| 8 | Report | what was removed, warnings (small classes, imbalance + majority-class baseline, p >> n, tiny test set, outlier samples, batch confounded with class), next commands. |

**Data types and normalisation** (`--data-type auto` detects; override if wrong):
| type | detected as | default normalisation |
|---|---|---|
| `counts` | non-negative integers (RNA-seq read counts) | CPM (removes sequencing depth) -> log2(CPM+1); genes expressed in < 20 % of samples dropped |
| `intensity` | microarray-like values | log2 if non-negative and heavy-tailed; **quantile normalisation** to the training reference (removes per-sample scale differences) |
| `log` | already log-scaled (small range) | none |
Choose another method with `--normalize none|cpm|median|quantile`; per-gene z-score with `--scale zscore` (hilbertfuse standardises internally, so leave it off for hilbertfuse).

## 5. Options you will actually use
`--label-col label` · `--id-col sample_id` · `--group-col patient_id` (avoid patient leakage) · `--batch-col lab` (confounding check) · `--transpose` · `--label-map map.csv` · `--test-size 0.2` · `--no-split` · `--top-var 5000` (keep the 5,000 most variable genes) · `--drop-outliers` · `--seed 0`. All options: `python dataprep\prepare_data.py fit -h`.

## 6. Output files (in `--out`)
`train.csv.gz`, `test.csv.gz` (genes + `label`, no index column, ready for `hilbertfuse`) · `prep_report.md/.json` · `prep_model.joblib` · `genes_kept.txt` · `samples_train.csv`, `samples_test.csv` (which sample went where). The `.joblib` holds training statistics derived from your data (gene medians, quantile reference): treat it like your data; load only files you trust.

## 7. Does it help? (measured)
Real data sets (golub, alon, khan, shipp), 5-fold cross-validation, same model in every column (tuned logistic regression, 1024 MI-selected genes, everything re-fitted inside each training fold). "messy" = synthetic mess added to the real file: every sample multiplied by a random factor (log-normal, sigma 0.5, like different depth/scaling), 3 % random missing values, 2 constant genes. Accuracy %:

| data set | clean file | clean + prep | messy file | messy + prep |
|---|---|---|---|---|
| alon | 82.7 | 81.0 | 81.0 | 85.8 |
| golub | 98.7 | 98.7 | 95.8 | 98.7 |
| khan | 96.8 | 96.8 | 96.9 | 96.9 |
| shipp | 92.0 | 92.0 | 92.1 | 92.0 |
| **mean** | **92.5** | **92.1** | **91.5** | **93.3** |

Reading it honestly:
* On files that were already clean, preparing changed accuracy by -0.4 points on average: no benefit, no real harm.
* With the added mess, accuracy fell by 1.0 point on average without preparation (mostly golub, 98.7 -> 95.8); with preparation it came back (golub 98.7; mean 93.3).
* These differences are small and within noise (4 data sets, 12-24 test samples per fold). The mess was mild, and a standardising logistic regression is already fairly robust to it.
* The bigger benefits are not accuracy numbers: removing duplicate samples and keeping each patient on one side of the split **lowers inflated scores to honest ones**, label/gene-name fixes stop silent errors, and the same cleaning is applied to new patient files. Reproduce the table: `python dataprep\validate_with_hilbertfuse.py` (needs the hilbertfuse project installed); raw folds in `validation.csv`.

## 8. Privacy
Everything runs locally; nothing is uploaded. Keep patient files and outputs in `data_private\` (ignored by git).

## 9. Tests
`pytest dataprep\test_prepare.py` (6 tests: no cross-sample dependence at apply time, count-depth invariance, quantile reference, missing/constant handling, cleaning + patient-wise split + exact reproduction by `apply`, refusal of wrong-gene files).
