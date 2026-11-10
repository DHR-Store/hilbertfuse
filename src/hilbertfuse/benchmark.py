"""Leak-free benchmark: classical baselines vs paper-style DenseNet vs HilbertFuseNet / HilbertFuse.

Every fold re-fits ALL preprocessing on the training part only. Each fit appends one row to
<out>/results.csv (accuracy, weighted precision/recall/F1 as in the paper, macro versions, seconds),
so partial runs are never lost and a re-run resumes where it stopped.

Method names:  lr_mi lr_all svm_mi knn_mi | lr_var lr_chi2 lr_f lr_mi lr_mird (gene-selection study) |
               paper hfnet fuse hfnet_ens fuse_ens  "hfnet+fuse" (one fit -> rows hfnet and fuse) |
               any of them with "@N" = use N genes (e.g. lr_mi@256, hfnet+fuse@576).
"""
from __future__ import annotations
import os
import time
import warnings
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import datasets as ds
from .pipeline import preset
from .preprocess import GeneImagePreprocessor

NN_METHODS = {
    "paper": dict(name="paper"),
    "hfnet": dict(name="hfnet"),
    "hfnet_ens": dict(name="hfnet_ens"),
    "fuse": dict(name="fuse"),
    "fuse_ens": dict(name="fuse_ens"),
    "abl_rowmajor": dict(name="hfnet", over=dict(layout="rowmajor")),
    "abl_1ch": dict(name="hfnet", over=dict(channels="expr")),
    "abl_mi": dict(name="hfnet", over=dict(selection="mi")),
    "abl_noSE": dict(name="hfnet", over=dict(model_kwargs=dict(se=False, multiscale=False))),
}
CLASSICAL = ("lr_mi", "svm_mi", "lr_all", "knn_mi", "lr_var", "lr_chi2", "lr_f", "lr_mird")


def metrics(y_true, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, pred), precision=p, recall=r, f1=f,
                precision_macro=pm, recall_macro=rm, f1_macro=fm)


_PRE_CACHE: dict = {}


def _select_features(sel, Xtr, yi, Xte, n_genes, seed, key=None):
    """Return (Ftr, Fte) feature matrices in [0,1] for the chosen gene-selection strategy.
    Plain MI ranking is computed once per training fold (key) and sliced to the top-n genes: identical
    to re-fitting with n_genes=n, but much faster when several methods/gene counts share a fold."""
    if sel == "mi" and key is not None:
        ck = (key, "mi")
        if ck not in _PRE_CACHE:
            _PRE_CACHE.clear()
            _PRE_CACHE[ck] = GeneImagePreprocessor(4096, selection="mi", layout="rowmajor", channels="expr",
                                                   random_state=seed).fit(Xtr, yi)
        pre = _PRE_CACHE[ck]
        pos = {g: i for i, g in enumerate(pre.gene_names_)}
        idx = [pos[g] for g in pre.mi_scores_.index[:n_genes]]
        f = lambda X: pre.transform(X)[:, 0].reshape(len(X), -1)[:, pre.pix_][:, idx]
        return f(Xtr), f(Xte)
    if sel in ("mi", "mird"):
        pre = GeneImagePreprocessor(n_genes, selection="mi" if sel == "mi" else "mi_rd", layout="rowmajor",
                                    channels="expr", random_state=seed).fit(Xtr, yi)
        f = lambda X: pre.transform(X)[:, 0].reshape(len(X), -1)[:, pre.pix_]
        return f(Xtr), f(Xte)
    A, B = Xtr.to_numpy(), Xte.reindex(columns=Xtr.columns).to_numpy()
    if sel != "all" and A.min() >= 0 and np.percentile(A, 99.9) > 50:
        A, B = np.log2(A + 1), np.log2(np.maximum(B, 0) + 1)
    mn, mx = A.min(0), A.max(0)
    keep = (mx - mn) > 1e-8
    A, B = (A[:, keep] - mn[keep]) / (mx[keep] - mn[keep]), np.clip((B[:, keep] - mn[keep]) / (mx[keep] - mn[keep]), 0, 1)
    if sel == "all":
        return A, B
    if sel == "var":
        score = A.var(0)
    elif sel == "chi2":
        score = np.nan_to_num(chi2(A, yi)[0])
    else:  # "f"
        score = np.nan_to_num(f_classif(A, yi)[0])
    idx = np.argsort(-score)[:n_genes]
    return A[:, idx], B[:, idx]


def _classical(kind, Xtr, ytr, Xte, n_genes, seed, key=None):
    clf_kind, _, sel = kind.partition("_")
    sel = {"mi": "mi", "all": "all", "var": "var", "chi2": "chi2", "f": "f", "mird": "mird"}[sel]
    if clf_kind == "knn":
        sel = "mi"
    classes = sorted(set(ytr))
    yi = np.array([classes.index(v) for v in ytr])
    Ftr, Fte = _select_features(sel, Xtr, yi, Xte, n_genes, seed, key)
    cv = max(2, int(min(3, np.bincount(yi).min())))
    if clf_kind == "knn":                                   # paper: n_neighbors = 5 on [0,1] features
        m = KNeighborsClassifier(n_neighbors=5).fit(Ftr, yi)
        return np.array(classes)[m.predict(Fte)]
    sc = StandardScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    if clf_kind == "svm":
        m = GridSearchCV(LinearSVC(max_iter=5000), {"C": [1e-3, 1e-2, 1e-1, 1]}, cv=cv).fit(Ftr, yi)
    else:
        m = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=cv, max_iter=3000).fit(Ftr, yi)
    return np.array(classes)[m.predict(Fte)]


def _append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_benchmark(datasets: Sequence[str], methods: Sequence[str], folds: int = 5, repeats: int = 1,
                  n_genes: int = 1024, epochs: int = 50, batch_size: int = 16, seed: int = 0,
                  out: str = "bench", device=None, data_dir=None) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    seen = set(zip(done.get("dataset", []), done.get("method", []), done.get("split", [])))
    for name in datasets:
        df = ds.load(name, data_dir)
        y = df.pop("label").astype(str).to_numpy()
        X = df.astype(np.float32)
        splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(X, y))
        for si, (tr, te) in enumerate(splits):
            for meth in methods:
                base, _, nsuf = meth.partition("@")
                ng = int(nsuf) if nsuf else n_genes
                tag = f"@{nsuf}" if nsuf else ""
                t0 = time.time()
                if base == "hfnet+fuse":
                    if (name, "hfnet" + tag, si) in seen and (name, "fuse" + tag, si) in seen:
                        continue
                    from .hybrid import LinearHead
                    clf = preset("hfnet", n_genes=ng, epochs=epochs, batch_size=batch_size, device=device, seed=seed + si)
                    clf.fit((X.iloc[tr], y[tr]))
                    P = clf.predict_proba((X.iloc[te], None))
                    lin = LinearHead(ng, seed + si).fit(X.iloc[tr], y[tr])
                    outs = {"hfnet" + tag: P, "fuse" + tag: 0.5 * P + 0.5 * lin.predict_proba(X.iloc[te])}
                    for mname, PP in outs.items():
                        if (name, mname, si) in seen:
                            continue
                        row = dict(dataset=name, method=mname, split=si,
                                   **metrics(y[te], np.array(clf.classes_)[PP.argmax(1)]), sec=round(time.time() - t0, 1))
                        _append(path, row)
                        print(f"{name:8s} {mname:14s} split {si}  acc={row['accuracy']:.3f}  f1={row['f1']:.3f}", flush=True)
                    continue
                if (name, meth, si) in seen:
                    continue
                if base in CLASSICAL:
                    pred = _classical(base, X.iloc[tr], y[tr], X.iloc[te], ng, seed, key=(name, si))
                else:
                    spec = NN_METHODS[base]
                    clf = preset(spec["name"], n_genes=ng, epochs=epochs, batch_size=batch_size,
                                 device=device, seed=seed + si, **spec.get("over", {}))
                    clf.fit((X.iloc[tr], y[tr]))
                    pred = clf.predict((X.iloc[te], None))
                row = dict(dataset=name, method=meth, split=si, **metrics(y[te], pred), sec=round(time.time() - t0, 1))
                _append(path, row)
                print(f"{name:8s} {meth:14s} split {si}  acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  ({row['sec']}s)", flush=True)
    res = pd.read_csv(path)
    summarize(res, out)
    return res


def summarize(res: pd.DataFrame, out: str, reference: str = "hfnet") -> str:
    """Write summary.md: mean±std per dataset, average rank, paired Wilcoxon vs `reference`."""
    lines = []
    for metric in ("accuracy", "f1_macro"):
        g = res.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).reset_index()
        tab = g.pivot(index="dataset", columns="method", values="mean")
        sd = g.pivot(index="dataset", columns="method", values="std")
        cell = tab.round(3).astype(str) + " ± " + sd.round(3).astype(str)
        cell.loc["**mean**"] = tab.mean().round(3).astype(str)
        cell.loc["avg rank"] = tab.rank(axis=1, ascending=False).mean().round(2).astype(str)
        lines += [f"### {metric} (mean ± std over CV folds)", "", cell.to_markdown(), ""]
    if reference in set(res.method):
        lines += [f"### Paired Wilcoxon (accuracy), `{reference}` vs others (over dataset x fold pairs)", ""]
        piv = res.pivot_table(index=["dataset", "split"], columns="method", values="accuracy")
        for m in piv.columns:
            if m == reference:
                continue
            d = (piv[reference] - piv[m]).dropna()
            if len(d) >= 6 and (d != 0).any():
                p = wilcoxon(d).pvalue
                lines.append(f"- vs **{m}**: mean diff {d.mean():+.3f}, wins {int((d>0).sum())} / ties {int((d==0).sum())} / losses {int((d<0).sum())}, p={p:.4f}")
            else:
                lines.append(f"- vs **{m}**: mean diff {d.mean():+.3f} (too few pairs for a test)")
    md = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md
