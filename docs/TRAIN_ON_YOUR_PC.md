# Train and test on your own PC (step by step)

Everything runs locally. No GPU is required (an NVIDIA GPU makes it much faster). Commands below are for **Windows PowerShell**; Linux/macOS notes are at the end.

## 1. One-time setup (about 5–10 minutes)

1. Install **Python 3.10, 3.11 or 3.12** from python.org (tick *Add python.exe to PATH*). Check: `python --version`.
2. Unzip `hilbertfuse.zip`, open PowerShell in the unzipped `hilbertfuse` folder:
   ```powershell
   cd C:\path\to\hilbertfuse
   ```
3. **Automatic:** this creates a virtual environment, installs everything, runs the tests and trains once on a real dataset:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```
   **Manual (same thing):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
   python -m pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[dev,datasets]"
   pytest -q tests                        # expect: 9 passed
   ```
   NVIDIA GPU: install PyTorch with the command generated at https://pytorch.org instead of the CPU line, then add `--device cuda` to the commands below.

Each later session: open PowerShell in the folder and run `.\.venv\Scripts\Activate.ps1`.

## 2. Train + test on REAL data (one command)

```powershell
hilbertfuse real-demo --dataset khan
```
This splits the real Khan cancer data set 70/30 (stratified), **trains** on the 70 %, **tests** on the unseen 30 %, compares with a tuned logistic regression on the same split, and prints accuracy, macro-F1, per-class accuracy and the confusion matrix. Everything is saved under `runs\real_demo\khan\`:

| file | content |
|---|---|
| `train.csv.gz`, `test.csv.gz` | the exact split (so you can see the file format for your own data) |
| `metrics.json` | all test scores |
| `predictions.csv` | per-sample true label, prediction, class probabilities |
| `model\` | trained model (load it again with `hilbertfuse evaluate / predict / explain`) |

Other real data sets: `alon, golub, gordon, khan, pomeroy, shipp, singh, su`. **All eight in one go:**
```powershell
hilbertfuse real-demo --dataset all --out runs\all_real
```
(prints a summary table and saves `runs\all_real\summary.csv`).

Useful options: `--epochs 80` (more training), `--n-genes 1024`, `--test-size 0.3`, `--seed 1` (different split), `--preset fuse|hfnet|fuse_ens|hfnet_ens|paper`, `--device cuda`, `--quiet`.

If `hilbertfuse` is "not recognized": `python -m hilbertfuse.cli real-demo --dataset khan` does the same.

What you should see (from our run; see `docs/EXAMPLE_REAL_DATA_RUN.md`):

| dataset   |   train |   test |   genes |   accuracy |   macro_f1 |   logreg_baseline_acc |   minutes |
|:----------|--------:|-------:|--------:|-----------:|-----------:|----------------------:|----------:|
| alon      |      43 |     19 |    2000 |      0.895 |      0.878 |                 0.895 |       0.5 |
| golub     |      50 |     22 |    7129 |      0.909 |      0.902 |                 0.909 |       1.1 |
| gordon    |     126 |     55 |   12533 |      1     |      1     |                 0.982 |       2.1 |
| khan      |      44 |     19 |    2308 |      0.947 |      0.962 |                 0.947 |       0.6 |
| pomeroy   |      42 |     18 |    7128 |      0.778 |      0.75  |                 0.722 |       0.9 |
| shipp     |      53 |     24 |    7129 |      1     |      1     |                 0.958 |       1.1 |
| singh     |      71 |     31 |   12600 |      0.903 |      0.903 |                 0.903 |       1.5 |
| su        |      71 |     31 |    5565 |      1     |      1     |                 1     |       1.4 |

## 3. Train on YOUR data
Speed choices (`--preset`): **linear** = seconds, **fast** = about 1-3 minutes, **fuse** = full (slowest). In our benchmark the three give similar accuracy on small cohorts, so start with `linear`; the CNN adds gene-image explanations. The menu (`start_windows.bat`, option 4) asks you.

Put two files next to each other: **rows = samples, columns = genes, plus one `label` column** (CSV, TSV, or Parquet; `.gz` ok), e.g. `train.csv` and `test.csv`. (Genes-by-samples file? add `--transpose`.)
```powershell
hilbertfuse train --train train.csv --test test.csv --label-col label --out runs\mymodel
hilbertfuse evaluate --model runs\mymodel --test test.csv
hilbertfuse predict  --model runs\mymodel --input new_samples.csv --out predictions.csv
hilbertfuse explain  --model runs\mymodel --data test.csv --top-k 40 --out runs\mymodel\genes
```
(PowerShell breaks long lines with a backtick `` ` ``, not `\`. Keep each command on one line.)

Before trusting a model, compare it with strong baselines on **your** data with leak-free cross-validation:
```powershell
hilbertfuse cv --data all_samples.csv --folds 5 --preset fuse
hilbertfuse benchmark --datasets khan,alon,golub --methods lr_mi,svm_mi,lr_all,paper,hfnet+fuse --folds 5 --out bench
```

## 4. How long does it take?

Rough guide (measured on one slow CPU core in our build environment; a normal desktop is similar or faster): small data sets (60–110 samples) 0.5–2 minutes each for `real-demo`; all eight about 15–25 minutes; the full 7-data-set benchmark with 5 folds a few hours on CPU, much less on a GPU.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | reinstall Python with *Add to PATH*, or use `py -3.12 -m venv .venv` |
| pip cannot find torch (Python 3.13) | use Python 3.10–3.12 |
| script execution disabled | `Set-ExecutionPolicy -Scope Process Bypass` then activate the venv again |
| `labels not seen in training` | your test file has a class that is absent from the training file |
| out of memory | `--n-genes 576` and/or fewer folds |
| very slow | `--epochs 30 --n-genes 576`, or use an NVIDIA GPU with `--device cuda` |
| "Could not load pretrained weights" | only for the optional pretrained backbone; it falls back to random init |

## 6. Linux / macOS
```bash
bash scripts/setup_unix.sh                      # setup + first run
source .venv/bin/activate
hilbertfuse real-demo --dataset all --out runs/all_real
```

## 7. Read the results correctly
* One train/test split of 60–180 samples gives **noisy** scores (one wrong sample moves accuracy by 5 points). For trustworthy numbers use `cv`/`benchmark`.
* On these small data sets a tuned logistic regression is as good as the CNN (see `docs/RESEARCH_HILBERTFUSE.md`); `real-demo` prints it next to the model for exactly that reason.
* Research use only; not a diagnostic tool.
