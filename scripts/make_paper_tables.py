"""Build paper-style result tables (Accuracy / Precision / Recall / F1-Score in %) from a benchmark results.csv.
   python scripts/make_paper_tables.py results/benchmark_results_v2.csv docs/PAPER_STYLE_TABLES
Writes <out>.md, <out>.html and PNG images of the main tables into docs/figures/.
Precision/Recall/F1 are class-weighted averages, exactly as in the paper (so recall == accuracy)."""
import os, sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
res = pd.read_csv(src)
res["method"] = res.method.replace({"new": "hfnet", "new_lr": "fuse"})
NAMES = {"fuse": "HilbertFuse (CNN + linear)", "hfnet": "HilbertFuseNet (CNN only)", "paper": "Paper-style DenseNet",
         "lr_mi": "MI_LogReg", "lr_all": "LogReg (all genes)", "svm_mi": "MI_SVM", "knn_mi": "MI_KNN",
         "lr_var": "Var_LogReg", "lr_chi2": "Chi2_LogReg", "lr_f": "FTest_LogReg", "lr_mird": "MI+RD_LogReg"}
COLS = ["accuracy", "precision", "recall", "f1"]
HEAD = ["Accuracy", "Precision", "Recall", "F1-Score"]
DATASETS = sorted(res.dataset.unique())
TASK = {"alon": "colon tumor vs normal", "golub": "ALL vs AML leukemia", "gordon": "lung: mesothelioma vs adenocarcinoma",
        "khan": "4 small round blue cell tumors", "pomeroy": "brain tumor survival (hard)", "shipp": "DLBCL vs follicular lymphoma",
        "singh": "prostate tumor vs normal", "su": "4 carcinoma tissue types"}


def pooled(methods, datasets=None):
    """Mean over datasets of the per-dataset fold-mean (every data set counts equally)."""
    r = res[res.method.isin(methods)]
    if datasets is not None:
        r = r[r.dataset.isin(datasets)]
    g = r.groupby(["method", "dataset"])[COLS].mean().groupby("method").mean() * 100
    return g.reindex([m for m in methods if m in g.index])


def fmt(df, labels=None, extra=None):
    d = df.copy()
    d.index = [(labels or NAMES).get(i, i) for i in d.index]
    d = d.round(2).map(lambda v: f"{v:.2f}%")
    d.columns = HEAD
    if extra is not None:
        for k, v in extra.items():
            d[k] = v
    return d


def md(df, first="Method"):
    cols = [first] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, row in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


tables = []   # (title, df, first-col-name, note)
nd = res.dataset.nunique()
# Table 1 - parameter settings (static text from the code)
t1 = pd.DataFrame({"Parameter": [
    "genes kept = 1024 (image 32x32 Hilbert), MI + redundancy pruning (abs(r) > 0.9), 3 channels, AdamW lr=0.001, weight_decay=1e-4, cosine schedule, epochs=50 (benchmark) / 80 (saved models), batch_size=16, growth_rate=12, blocks=(4,6,8,6), init_features=24, bn_size=4, dropout=0.1/0.3, SE reduction=8, mixup alpha=0.3, label smoothing=0.1, gene dropout=0.1, noise std=0.02, class-weighted loss",
    "same CNN size and budget; row-major layout, plain MI, 1 channel, no SE / multi-scale head, no dropout, no mixup, no label smoothing, no class weights",
    "HilbertFuseNet probabilities averaged 50/50 with MI_LogReg probabilities",
    "MI top-1024 genes, standardised; LogisticRegressionCV, C in {1e-3, 1e-2, 0.1, 1, 10}, 3-fold inner CV",
    "MI top-1024 genes, standardised; LinearSVC, C in {1e-3, 1e-2, 0.1, 1}, 3-fold inner GridSearchCV",
    "n_neighbors = 5 on MI top-1024 genes scaled to [0, 1]"]},
    index=["HilbertFuseNet", "Paper-style DenseNet", "HilbertFuse", "MI_LogReg", "MI_SVM", "MI_KNN"])
tables.append(("TABLE 1 | Parameter settings.", t1, "Algorithm", "Same 1024-gene budget for every method. 5-fold stratified CV; all preprocessing re-fit inside each training fold."))
# Table 2
m2 = ["fuse", "hfnet", "paper", "lr_mi", "lr_all", "svm_mi", "knn_mi"]
tables.append((f"TABLE 2 | The experimental results of the methods (mean over {nd} real data sets, 5-fold CV).", fmt(pooled(m2)), "Method",
               "Each data set counts equally; precision/recall/F1 are class-weighted as in the paper."))
# Table 3 - per data set accuracy
t3 = res[res.method.isin(m2)].groupby(["dataset", "method"]).accuracy.mean().unstack()[[m for m in m2 if m in set(res.method)]] * 100
n_samp = {"alon": 62, "golub": 72, "gordon": 181, "khan": 63, "pomeroy": 60, "shipp": 77, "singh": 102, "su": 102}
t3.columns = [NAMES[c] for c in t3.columns]
t3 = t3.round(2).map(lambda v: f"{v:.2f}%")
t3.insert(0, "Samples", [n_samp.get(d, "") for d in t3.index])
t3.insert(0, "Task", [TASK.get(d, "") for d in t3.index])
tables.append(("TABLE 3 | Accuracy per real data set (5-fold CV).", t3, "Data set", ""))
# Table 4 - gene selection with LR
m4 = ["lr_var", "lr_chi2", "lr_f", "lr_mi", "lr_mird"]
tables.append(("TABLE 4 | The performance of different gene-selection strategies (same logistic-regression classifier, 1024 genes).",
               fmt(pooled(m4)), "Method", "Selection fitted on training folds only."))
# Table 5 - classifiers on MI genes
m5 = ["knn_mi", "svm_mi", "lr_mi", "paper", "hfnet", "fuse"]
l5 = {"knn_mi": "MI_KNN", "svm_mi": "MI_SVM", "lr_mi": "MI_LogReg", "paper": "MI_DenseNet (paper-style)",
      "hfnet": "MI+RD_HilbertFuseNet", "fuse": "HilbertFuse"}
tables.append(("TABLE 5 | The performance of different classifiers.", fmt(pooled(m5), l5), "Classifiers", ""))
# Table 6 - number of genes (image side^2) with LR
rows = []
for n, side in [(100, "10 x 10"), (256, "16 x 16"), (576, "24 x 24"), (1024, "32 x 32"), (2048, "45 x 45"), (4096, "64 x 64")]:
    m = "lr_mi" if n == 1024 else f"lr_mi@{n}"
    p = pooled([m])
    if len(p):
        rows.append((f"{side}  ({n} genes)", *p.iloc[0].tolist()))
if rows:
    t6 = pd.DataFrame(rows, columns=["Dimensions"] + HEAD).set_index("Dimensions")
    t6[HEAD] = t6[HEAD].round(2).map(lambda v: f"{v:.2f}%")
    tables.append(("TABLE 6 | The performance of different image dimensions / gene counts (MI_LogReg).", t6, "Dimensions",
                   "Image side^2 = number of genes. Logistic regression does not use the image; this shows the effect of the number of genes only."))
rows = []
for d in ["khan", "golub"]:
    for n, side in [(256, "16 x 16"), (576, "24 x 24 (32 x 32 Hilbert grid)"), (1024, "32 x 32")]:
        for m, lab in (("hfnet", "HilbertFuseNet"), ("fuse", "HilbertFuse")):
            mm = m if n == 1024 else f"{m}@{n}"
            r = res[(res.method == mm) & (res.dataset == d)]
            if len(r):
                rows.append((f"{d}: {lab}, {side} ({n} genes)", *(r[COLS].mean() * 100).tolist()))
if rows:
    t6b = pd.DataFrame(rows, columns=["Setting"] + HEAD).set_index("Setting")
    t6b[HEAD] = t6b[HEAD].round(2).map(lambda v: f"{v:.2f}%")
    tables.append(("TABLE 6b | CNN image-size check on two data sets.", t6b, "Setting", "Small check only (2 data sets, 5 folds)."))
# Per pretrained model tables
per_model = []
for d in DATASETS:
    r = res[(res.dataset == d) & res.method.isin(m2)]
    g = r.groupby("method")[COLS].mean() * 100
    sd = r.groupby("method").accuracy.std() * 100
    g = g.reindex([m for m in m2 if m in g.index])
    f = fmt(g)
    f["Accuracy std"] = [f"± {sd[m]:.2f}" for m in g.index]
    per_model.append((f"TABLE M-{d} | Pretrained model `{d}`: {TASK.get(d, '')} ({n_samp.get(d, '?')} samples, 5-fold CV).", f, "Classifier", ""))

# ---- write markdown + html
parts = ["# Paper-style result tables (real data, leak-free 5-fold cross-validation)\n",
         "Format follows Tables 1, 2, 4, 5, 6 of Wang et al. 2021. Precision/Recall/F1 are class-weighted averages. "
         "Data: 8 real published cancer expression data sets (60-181 samples each). Research use only.\n"]
for title, df, first, note in tables + per_model:
    parts += [f"### {title}\n", md(df, first), "", (f"*{note}*\n" if note else "")]
open(out + ".md", "w", encoding="utf-8").write("\n".join(parts))
css = """<style>body{font-family:Times,serif;max-width:980px;margin:30px auto;color:#111}h3{font-family:Arial;font-size:15px;margin:28px 0 6px}
table{border-collapse:collapse;width:100%;font-size:14px}th{border-top:2px solid #000;border-bottom:1px solid #000;text-align:left;padding:6px 10px;font-family:Arial}
td{padding:5px 10px}tr:last-child td{border-bottom:2px solid #000}.n{font-size:12px;color:#444;margin-top:4px}</style>"""
html = [f"<html><head><meta charset='utf-8'><title>Paper-style tables</title>{css}</head><body>"]
for title, df, first, note in tables + per_model:
    d = df.copy(); d.index.name = first
    html.append(f"<h3>{title}</h3>" + d.to_html(border=0, escape=True) + (f"<div class='n'>{note}</div>" if note else ""))
open(out + ".html", "w", encoding="utf-8").write("\n".join(html) + "</body></html>")

# ---- PNG images of main tables (for slides)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("docs/figures", exist_ok=True)
    for key, (title, df, first, note) in {"table2": tables[1], "table4": tables[3], "table5": tables[4]}.items():
        d = df.reset_index().rename(columns={"index": first})
        fig, ax = plt.subplots(figsize=(8.6, 0.55 + 0.42 * (len(d) + 1)))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        tb = ax.table(cellText=d.values, colLabels=d.columns, loc="upper left", cellLoc="left", colLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.5)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor("white"); cell.set_linewidth(0)
            if r == 0:
                cell.set_text_props(fontweight="bold"); cell.set_facecolor("#eeeeee")
        fig.savefig(f"docs/figures/{key}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
except Exception as e:
    print("table PNGs skipped:", e)
print(f"wrote {out}.md, {out}.html and PNGs ({len(tables)} tables + {len(per_model)} per-model tables)")
