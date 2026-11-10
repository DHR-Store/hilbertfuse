# Commands — install, train, evaluate, benchmark

## 0. One-command start
`start_windows.bat` (double-click) or `bash start.sh` -> installs on first run, then opens a menu. Without the launcher: `hilbertfuse menu`.

## 1. Install
```bash
unzip hilbertfuse.zip && cd hilbertfuse
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # GPU: pick the build at pytorch.org
pip install -e ".[dev,datasets]"
pytest -q tests                                             # 9 tests, ~1 min
```

## 1b. One-command train + test on real data
```bash
hilbertfuse real-demo --dataset khan            # or: alon golub gordon pomeroy shipp singh su | all
```
Windows PowerShell walkthrough: `docs/TRAIN_ON_YOUR_PC.md`.

## 1d. Heat maps
```bash
hilbertfuse heatmap --model models/golub --data my_samples.csv --out heatmaps   # gene-image map + top-gene expression + class x gene
```
Details: `docs/HEATMAPS.md`.

## 1c. Train quickly on a CPU
```bash
hilbertfuse train --train data.csv --label-col label --preset linear --out models/mine   # seconds: logistic regression only
hilbertfuse train --train data.csv --label-col label --preset fast   --out models/mine   # ~1-3 min: small CNN + linear
hilbertfuse train --train data.csv --label-col label --preset fuse   --out models/mine   # full CNN + linear
hilbertfuse scores                                                                       # cross-validated scores of all trained models
```
`predict` prints class, confidence, runner-up and an UNCERTAIN flag (`--min-confidence 0.70`).

## 2. Look at the real data
```bash
hilbertfuse datasets                                        # table of 11 real cancer datasets (8 bundled)
hilbertfuse datasets --fetch yeoh --out datasets            # download one more (needs internet + `rdata`)
```
Bundled (`datasets/*.csv.gz`): alon, golub, khan, shipp, singh, su, gordon, pomeroy. Format: rows = samples, columns = gene probes, last column `label`.

## 3. Train your own model (any table with a label column)
```bash
# recommended default (HilbertFuse = HilbertFuseNet CNN + tuned logistic regression)
hilbertfuse train --train train.csv --test test.csv --label-col label --n-genes 1024 --epochs 80 --out runs/m1
# CNN only / ensemble / paper-style control
hilbertfuse train --train train.csv --test test.csv --preset hfnet     --out runs/cnn
hilbertfuse train --train train.csv --test test.csv --preset fuse_ens  --out runs/ens     # 3 CNNs + linear
hilbertfuse train --train train.csv --test test.csv --preset paper     --out runs/paper
# genes x samples file (TCGA/GDAC style): add --transpose ; GPU: add --device cuda ; pretrained DenseNet-121 variant:
hilbertfuse train --train train.csv --test test.csv --preset hfnet --backbone hilbertfusenet121 --pretrained --out runs/pre
```
Outputs in `runs/m1/`: model, `test_metrics.json`, history, `mi_scores.csv`.

## 4. Use a trained model
```bash
hilbertfuse evaluate --model runs/m1 --test test.csv
hilbertfuse predict  --model runs/m1 --input new_samples.csv --out predictions.csv
hilbertfuse explain  --model runs/m1 --data test.csv --top-k 40 --out runs/m1/genes   # Guided Grad-CAM
```

## 5. Honest model selection (do this before trusting any model)
```bash
hilbertfuse cv --data all.csv --folds 5 --preset fuse        # leak-free CV for your own data
hilbertfuse benchmark --datasets khan,alon,golub --methods lr_mi,svm_mi,lr_all,paper,hfnet+fuse --folds 5 --out bench
python scripts/make_report.py bench/results.csv docs/RESULTS.md
```
Benchmark methods: `lr_mi svm_mi lr_all` (tuned classical), `paper`, `hfnet`, `hfnet_ens`, `fuse`, `fuse_ens`, `hfnet+fuse` (one network fit scored both ways), ablations `abl_rowmajor abl_1ch abl_mi abl_noSE`.

## 6. Python
```python
from hilbertfuse import preset, load_table
from hilbertfuse.datasets import load
X, y = load_table(load("khan"))
clf = preset("fuse", n_genes=1024, epochs=80).fit((X.iloc[:44], y.iloc[:44].values))
print(clf.evaluate((X.iloc[44:], y.iloc[44:].values))["accuracy"])
clf.explain((X.iloc[44:], y.iloc[44:].values), top_k=20)["global"]
```
## 7. Full reproduction
`bash scripts/run_all.sh`
