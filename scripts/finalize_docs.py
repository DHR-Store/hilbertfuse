"""Regenerate every results-dependent document from ONE benchmark results file, so all numbers match.
   python scripts/finalize_docs.py results/benchmark_results_v2.csv
Runs: model cards + scores, paper-style tables, figures, then rewrites README results, docs/USE_MODELS.md table,
docs/RESEARCH_HILBERTFUSE.md (sections 5-7), docs/RESULTS.md and docs/TECH_AND_ALGORITHM_FOR_PPT.md (placeholders)."""
import json, os, re, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src = sys.argv[1]
PY = sys.executable
env = dict(os.environ, PYTHONPATH="src")
for cmd in ([PY, "scripts/make_model_cards.py", src, "models"],
            [PY, "scripts/make_paper_tables.py", src, "docs/PAPER_STYLE_TABLES"],
            [PY, "scripts/make_figures.py", src],
            [PY, "scripts/make_report.py", src, "docs/RESULTS.md"]):
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)

res = pd.read_csv(src)
res["method"] = res.method.replace({"new": "hfnet", "new_lr": "fuse"})
md = open("docs/PAPER_STYLE_TABLES.md", encoding="utf-8").read()


def section(prefix):
    m = re.search(rf"### {re.escape(prefix)}.*?(?=\n### |\Z)", md, re.S)
    return m.group(0).strip() if m else "(not available)"


# ---------------------------------------------------------------- findings (computed, not hand-written)
piv = res.pivot_table(index=["dataset", "split"], columns="method", values="accuracy")
dsm = res.groupby(["method", "dataset"]).accuracy.mean().unstack()
mean = dsm.mean(axis=1) * 100
NAME = {"fuse": "HilbertFuse", "hfnet": "HilbertFuseNet (CNN alone)", "paper": "the paper-style DenseNet",
        "lr_mi": "MI logistic regression", "lr_all": "all-gene logistic regression", "svm_mi": "MI linear SVM", "knn_mi": "MI KNN"}


def cmp(a, b):
    d = (piv[a] - piv[b]).dropna()
    p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else float("nan")
    return d.mean() * 100, p, int((d > 0).sum()), int((d < 0).sum()), len(d)


def verdict(a, b):
    diff, p, w, l, n = cmp(a, b)
    if np.isnan(p) or p >= 0.05:
        v = f"not significantly different (paired Wilcoxon p={p:.2f})"
    else:
        v = f"significantly {'better' if diff > 0 else 'worse'} (" + ("p<0.001" if p < 0.001 else f"p={p:.3f}") + ")"
    return f"{diff:+.1f} points, {v}, better in {w} / worse in {l} of {n} folds"


best_cl = max(["lr_mi", "lr_all", "svm_mi", "knn_mi"], key=lambda m: mean[m])
F = []
F.append(f"**HilbertFuse vs the best tuned classical method** ({NAME[best_cl]}): mean accuracy {mean['fuse']:.1f}% vs {mean[best_cl]:.1f}%: {verdict('fuse', best_cl)}.")
F.append(f"**HilbertFuse vs the paper-style DenseNet:** {mean['fuse']:.1f}% vs {mean['paper']:.1f}%: {verdict('fuse', 'paper')}.")
F.append(f"**CNN alone (HilbertFuseNet) vs {NAME['lr_mi']}:** {mean['hfnet']:.1f}% vs {mean['lr_mi']:.1f}%: {verdict('hfnet', 'lr_mi')}.")
F.append(f"**CNN alone vs the paper-style DenseNet (do the new design ideas help the CNN?):** {mean['hfnet']:.1f}% vs {mean['paper']:.1f}%: {verdict('hfnet', 'paper')}.")
wins = [d for d in dsm.columns if dsm.loc["hfnet", d] > dsm.loc["lr_mi", d] + 1e-9]
F.append("The CNN alone beats MI logistic regression on " + (", ".join(wins) if wins else "no data set") + f" (of {dsm.shape[1]}).")
unstable = [(d, res[(res.method == 'hfnet') & (res.dataset == d)].accuracy.std()) for d in dsm.columns]
worst = max(unstable, key=lambda t: t[1])
F.append(f"The CNN alone is the least stable method: fold-to-fold standard deviation up to {worst[1]*100:.1f} points (on {worst[0]}).")
useless = []
for d in sorted(os.listdir("models")):
    f = os.path.join("models", d, "cv_scores.json")
    if os.path.exists(f):
        j = json.load(open(f))
        if j["not_useful"]:
            useless.append(f"`{d}` (CV accuracy {j['cv_accuracy']:.3f} vs {j['majority_baseline']:.3f} for always guessing the common class)")
F.append("Models judged **not useful** (no better than guessing the most common class): " + (", ".join(useless) if useless else "none") + ".")
sw = [res[res.method == ("lr_mi" if n == 1024 else f"lr_mi@{n}")].groupby("dataset").accuracy.mean().mean() * 100
      for n in (100, 256, 576, 1024, 2048, 4096) if len(res[res.method == ("lr_mi" if n == 1024 else f"lr_mi@{n}")])]
if sw:
    F.append(f"Number of genes (logistic regression, 100 to 4096 genes): accuracy varies only between {min(sw):.1f}% and {max(sw):.1f}%, far less than the 3.9-point swing the paper reports across image sizes with a CNN.")
F.append("Caveat: small cohorts (60-181 samples), one random seed, one platform per data set, networks trained from random weights; results do not automatically transfer to large cohorts such as TCGA.")
findings = "\n".join(f"* {x}" for x in F)

# ---------------------------------------------------------------- TECH doc
t = open("docs/TECH_AND_ALGORITHM_FOR_PPT.template.md", encoding="utf-8").read()
for ph, key in (("<<TABLE2>>", "TABLE 2"), ("<<TABLE5>>", "TABLE 5"), ("<<TABLE4>>", "TABLE 4")):
    t = t.replace(ph, section(key))
t = t.replace("<<TABLE6>>", section("TABLE 6 |") + "\n\n" + (section("TABLE 6b") if "TABLE 6b" in md else ""))
t = t.replace("<<FINDINGS>>", findings)
open("docs/TECH_AND_ALGORITHM_FOR_PPT.md", "w", encoding="utf-8").write(t)

# ---------------------------------------------------------------- RESEARCH doc (sections 5-7 rewritten)
r = open("docs/RESEARCH_HILBERTFUSE.md", encoding="utf-8").read()
r = r[:r.index("## 5. Measured results")]
r += "## 5. Measured results on 8 real cancer data sets (5-fold CV, leak-free)\n\n" + "\n\n".join(
    section(k) for k in ("TABLE 2", "TABLE 3", "TABLE 4", "TABLE 5", "TABLE 6 |")) + "\n\n"
r += ("Full tables incl. one table per pretrained model: `docs/PAPER_STYLE_TABLES.md` / `.html`. Raw per-fold numbers: `results/benchmark_results_v2.csv`.\n"
      "Settings: 1,024 genes, 50 epochs, batch 16, random initialisation (ImageNet weights could not be downloaded in the build environment), seed 0, one CPU core.\n\n"
      "## 6. What the numbers say (computed from the results file)\n\n" + findings + "\n\n"
      "## 7. Limitations\n\n* Small cohorts, 5 folds, one seed, 1,024 genes, 50 epochs, random initialisation; no external validation cohort.\n"
      "* Microarray data, not the paper's TCGA RNA-seq; to test TCGA, run `hilbertfuse benchmark` / `cv` on it yourself.\n"
      "* Gene rankings are model attributions, not validated biomarkers.\n* Research use only; not a diagnostic tool.\n")
open("docs/RESEARCH_HILBERTFUSE.md", "w", encoding="utf-8").write(r)

# ---------------------------------------------------------------- USE_MODELS table
rows = []
for d in sorted(os.listdir("models")):
    f = os.path.join("models", d, "cv_scores.json")
    if os.path.exists(f):
        j = json.load(open(f))
        rows.append((j["cv_accuracy"], f"| {d} | {j['task']} | {j['cv_accuracy']:.3f} ± {j['cv_accuracy_std']:.3f} | {j['cv_macro_f1']:.3f} | "
                     + ("**NOT useful** (no better than guessing)" if j["not_useful"] else "usable for research") + " |"))
tab = "| model | task | accuracy (5-fold CV) | macro-F1 | verdict |\n|---|---|---|---|---|\n" + "\n".join(x for _, x in sorted(rows, reverse=True))
u = open("docs/USE_MODELS.md", encoding="utf-8").read()
u = re.sub(r"\| model \| task \|.*?\n\n", tab + "\n\n", u, count=1, flags=re.S)
open("docs/USE_MODELS.md", "w", encoding="utf-8").write(u)

# ---------------------------------------------------------------- README
readme = f"""# HilbertFuse — cancer gene-expression classification you can clone and run

Pretrained models for 8 real cancer data sets, a quick way to train your own model on a CPU, and honest, leak-free scores. A leak-free, layout-aware CNN (**HilbertFuseNet**) fused with a tuned logistic regression (**HilbertFuse**), built on the ideas of *Wang et al., Front. Genet. 2021* (MI_DenseNetCAM).

**Research use only. The models recognise which known class an expression profile resembles. They do not find the cause of a cancer and must not be used to diagnose or treat patients.**

## Start in one step
```powershell
git clone https://github.com/<your-user>/hilbertfuse.git
cd hilbertfuse
.\\start_windows.bat            # Windows (double-click works too)   |   Linux/macOS: bash start.sh
```
First run installs the packages (5-10 min, internet once, Python 3.10-3.12). Then a menu opens: **2 predict**, **3 evaluate**, **4 train my own model (choose quick / fast / full)**, **5 cross-validation**, **6 explain genes**, **7 show all scores**.
Predictions run offline on your computer. Keep patient files in `data_private/` (never committed). VS Code users: `docs/VSCODE.md`.

## Use a pretrained model on your cancer / patient data
Your CSV: one row per sample, one column per gene, the **same gene columns** the model was trained on (`models/<name>/required_genes.txt`). The tool refuses files with the wrong genes.
```powershell
hilbertfuse predict  --model models\\golub --input my_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\\golub --test my_labeled_samples.csv
```
Output per sample: predicted class, **confidence**, runner-up class and an **UNCERTAIN** flag below 0.70. Low confidence means "do not trust this answer".

## Train your own model quickly on CPU
```powershell
hilbertfuse train --train my_data.csv --label-col label --preset linear --out models\\mymodel   # seconds
hilbertfuse train --train my_data.csv --label-col label --preset fast   --out models\\mymodel   # ~1-3 min (CNN + linear)
hilbertfuse cv --data my_data.csv --folds 5 --preset linear                                    # honest accuracy first
```

## Scores (paper-style Table 2: mean over 8 real data sets, 5-fold cross-validation)
{section('TABLE 2')}

Per pretrained model (accuracy, 5-fold CV):

{tab}

All tables (Tables 1-6 in the paper's format, plus one per model): `docs/PAPER_STYLE_TABLES.html`. What the numbers mean: `docs/RESEARCH_HILBERTFUSE.md`.

### Honest findings
{findings}

## Documentation
`docs/TECH_AND_ALGORITHM_FOR_PPT.md` (algorithm details, slide by slide) · `docs/USE_MODELS.md` · `docs/TRAIN_ON_YOUR_PC.md` · `docs/COMMANDS.md` · `docs/GIT_AND_SHARING.md` · `docs/PAPER_ANALYSIS.md` · model cards in `models/*/MODEL_CARD.md`.

Data sets in `datasets/` are public research data (citations in `docs/RESEARCH_HILBERTFUSE.md`). MIT license.
"""
open("README.md", "w", encoding="utf-8").write(readme)
print("finalized from", src, "|", len(res), "result rows,", res.dataset.nunique(), "data sets")
print(findings)
