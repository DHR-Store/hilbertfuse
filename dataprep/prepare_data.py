#!/usr/bin/env python
"""prepare_data.py - clean + normalise a raw gene-expression table and make a train/test split for training.

Everything that LEARNS from data (which genes to drop, medians used to fill gaps, quantile reference, z-score
statistics) is fitted on the TRAINING part only and then applied to the test part and to any new file, so no test
information leaks into training.  Needs only: numpy, pandas, scikit-learn, scipy, joblib (no torch).

  python prepare_data.py check --input raw.csv --label-col label                 # report only, writes nothing
  python prepare_data.py fit   --input raw.csv --label-col label --out clean     # clean + split + report
  python prepare_data.py apply --prep clean/prep_model.joblib --input new.csv --out new_clean.csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

VERSION = "1.0"


# ------------------------------------------------------------------ reading
def read_table(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if p.endswith((".tsv", ".txt", ".tab", ".tsv.gz", ".txt.gz")):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    if df.shape[1] == 1:                                    # wrong delimiter guess (e.g. ';')
        df = pd.read_csv(path, sep=None, engine="python")
    return df


@dataclass
class Config:
    data_type: str = "auto"          # auto | counts | intensity | log
    normalize: str = "auto"          # auto | none | cpm | median | quantile
    scale: str = "none"              # none | zscore   (hilbertfuse/sklearn scale again themselves; keep none)
    max_missing_gene: float = 0.2
    max_missing_sample: float = 0.2
    min_expr_frac: float = 0.2       # counts only: keep genes with log2(CPM+1) > 1 in >= this fraction of samples
    near_const_frac: float = 0.95    # drop genes where >= this fraction of samples share (about) the same value
    top_var: int = 0                 # 0 = keep all; N = keep the N most variable genes
    min_class_size: int = 5
    label_case: str = "auto"         # auto (merge spellings that differ by case/space) | keep | lower | upper
    strip_version: bool = False      # ENSG00000123456.7 -> ENSG00000123456
    uppercase_genes: bool = False
    drop_outliers: bool = False


# ------------------------------------------------------------------ the fitted (leak-free) transformer
class DataPrep:
    """fit(train) learns every statistic; transform(any table) applies them."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _detect(A: np.ndarray) -> str:
        v = A[np.isfinite(A)]
        if v.size == 0:
            return "intensity"
        p999 = np.percentile(v, 99.9)
        if v.min() >= 0 and np.allclose(v[: 200000], np.round(v[: 200000]), atol=1e-6) and v.max() >= 20:
            return "counts"                                  # non-negative integers = read counts (log data is not integer)
        if p999 <= 30:
            return "log"
        return "intensity"

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        return X.reindex(columns=self.genes_in_).to_numpy(dtype=np.float64)

    def fit(self, X: pd.DataFrame) -> "DataPrep":
        cfg = self.cfg
        A = X.to_numpy(dtype=np.float64)
        miss = np.isnan(A).mean(0)
        keep0 = miss <= cfg.max_missing_gene                       # 1. too many gaps -> drop
        if keep0.sum() == 0:
            raise SystemExit("ERROR: every gene has too many missing values (raise --max-missing-gene).")
        self.genes_in_ = list(X.columns[keep0])
        A = A[:, keep0]
        self.med_raw_ = np.nanmedian(A, axis=0)                    # 2. fill gaps with the training median
        A = np.where(np.isnan(A), self.med_raw_[None, :], A)
        # constant genes must go BEFORE normalisation (quantile/CPM would make them vary slightly between samples)
        keepc = (A.std(0) > 1e-8) & ((np.round(A, 4) == np.round(np.median(A, 0), 4)).mean(0) < cfg.near_const_frac)
        if keepc.sum() == 0:
            raise SystemExit("ERROR: all genes are constant or near-constant.")
        self.genes_in_ = [g for g, k in zip(self.genes_in_, keepc) if k]
        A, self.med_raw_ = A[:, keepc], self.med_raw_[keepc]
        self.data_type_ = cfg.data_type if cfg.data_type != "auto" else self._detect(A)
        norm = cfg.normalize
        if norm == "auto":
            norm = {"counts": "cpm", "intensity": "quantile", "log": "none"}[self.data_type_]
        self.norm_ = norm
        self.log_ = False
        if self.data_type_ == "counts":                           # counts always need log after CPM
            self.log_ = True
        elif self.data_type_ == "intensity" and A.min() >= 0 and np.percentile(A, 99.9) > 50:
            self.log_ = True
        M = self._core(A, fit=True)
        keep = np.ones(M.shape[1], bool)                           # 3. gene filters on the normalised data
        if self.data_type_ == "counts":
            keep &= (M > 1.0).mean(0) >= cfg.min_expr_frac
        sd = M.std(0)
        keep &= sd > 1e-8
        keep &= (np.round(M, 4) == np.round(np.median(M, 0), 4)).mean(0) < cfg.near_const_frac
        if cfg.top_var and keep.sum() > cfg.top_var:
            var = np.where(keep, M.var(0), -1)
            idx = np.argsort(-var)[: cfg.top_var]
            k2 = np.zeros_like(keep); k2[idx] = True
            keep = k2
        if keep.sum() == 0:
            raise SystemExit("ERROR: no gene survived the filters; relax --min-expr-frac / --near-const-frac.")
        self.keep_ = keep
        self.genes_out_ = [g for g, k in zip(self.genes_in_, keep) if k]
        Mk = M[:, keep]
        self.mu_, self.sd_ = Mk.mean(0), np.maximum(Mk.std(0), 1e-8)
        return self

    def _core(self, A: np.ndarray, fit: bool = False) -> np.ndarray:
        """raw (gap-filled) matrix -> normalised matrix (all retained-by-missingness genes)."""
        if self.data_type_ == "counts":
            sums = np.maximum(np.maximum(A, 0).sum(1, keepdims=True), 1.0)
            A = np.log2(np.maximum(A, 0) / sums * 1e6 + 1.0)       # CPM then log2
        elif self.log_:
            A = np.log2(np.maximum(A, 0) + 1.0)
        if self.norm_ == "median":
            if fit:
                self.global_median_ = float(np.median(np.median(A, axis=1)))
            A = A - np.median(A, axis=1, keepdims=True) + self.global_median_
        elif self.norm_ == "quantile":
            if fit:
                self.q_ref_ = np.sort(A, axis=1).mean(0)
            order = np.argsort(A, axis=1, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[np.arange(A.shape[0])[:, None], order] = np.arange(A.shape[1])[None, :]
            A = self.q_ref_[ranks]
        return A

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        A = self._align(X)
        A = np.where(np.isnan(A), self.med_raw_[None, :], A)
        M = self._core(A)[:, self.keep_]
        if self.cfg.scale == "zscore":
            M = (M - self.mu_) / self.sd_
        return pd.DataFrame(M.astype(np.float32), index=X.index, columns=self.genes_out_)


# ------------------------------------------------------------------ cleaning helpers
def clean_gene_names(cols, cfg: Config):
    out = []
    for c in cols:
        c = str(c).strip()
        if cfg.strip_version and c.upper().startswith("ENS") and "." in c:
            c = c.split(".")[0]
        if cfg.uppercase_genes:
            c = c.upper()
        out.append(c)
    return out


def load_and_clean(args, cfg: Config, log: list, warn: list):
    """raw file -> (X genes DataFrame, y Series|None, meta DataFrame with group/batch/id)."""
    df = read_table(args.input)
    log.append(f"Read `{args.input}`: {df.shape[0]} rows x {df.shape[1]} columns.")
    if args.transpose:
        df = df.set_index(df.columns[0]).T.reset_index(drop=True)
        log.append("Transposed (file was genes x samples).")
    meta = pd.DataFrame(index=df.index)
    for name, col in (("id", args.id_col), ("group", args.group_col), ("batch", args.batch_col)):
        if col:
            if col not in df.columns:
                raise SystemExit(f"ERROR: column '{col}' not found. Columns start with: {list(df.columns[:6])}")
            meta[name] = df[col].astype(str).str.strip()
    y = None
    if args.label_col and args.label_col in df.columns:
        y = df[args.label_col].astype("string").str.strip()
    elif args.label_col and args.cmd == "fit":
        raise SystemExit(f"ERROR: label column '{args.label_col}' not found. Columns start with: {list(df.columns[:6])}")
    drop = [c for c in (args.label_col, args.id_col, args.group_col, args.batch_col) if c and c in df.columns]
    X = df.drop(columns=drop)
    X.columns = clean_gene_names(X.columns, cfg)
    num = X.apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    if num.columns.duplicated().any():                      # incl. names that only differ by spaces/case/version
        n_dup = int(num.columns.duplicated().sum())
        num = num.T.groupby(level=0, sort=False).mean().T
        log.append(f"Merged {n_dup} duplicated gene column(s) by averaging.")
    all_nan = [c for c in num.columns if num[c].isna().all()]
    if all_nan:
        warn.append(f"{len(all_nan)} non-numeric/empty column(s) were dropped (e.g. {all_nan[:3]}). If one is an ID, pass --id-col.")
        num = num.drop(columns=all_nan)
    return num.astype(np.float64), y, meta


def clean_labels(y: pd.Series, cfg: Config, label_map: Optional[str], log: list, warn: list):
    y = y.copy()
    if label_map:
        mp = pd.read_csv(label_map)
        mp = dict(zip(mp.iloc[:, 0].astype(str).str.strip(), mp.iloc[:, 1].astype(str).str.strip()))
        y = y.map(lambda v: mp.get(v, v) if pd.notna(v) else v)
        log.append(f"Applied label map `{label_map}` ({len(mp)} rules).")
    if cfg.label_case == "auto":
        norm = y.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
        best = {}
        for low, grp in norm[norm != ""].groupby(norm[norm != ""].str.lower()):
            best[low] = grp.value_counts().index[0]            # most common spelling wins
        merged = {k: sorted(set(norm[norm.str.lower() == k])) for k in best if norm[norm.str.lower() == k].nunique() > 1}
        y = norm.map(lambda v: best.get(v.lower(), v) if v else pd.NA).astype("string")
        if merged:
            log.append("Merged label spellings that differ only by case/spaces: " + "; ".join(f"{v} -> {best[k]}" for k, v in merged.items()))
    elif cfg.label_case != "keep":
        y = y.str.lower() if cfg.label_case == "lower" else y.str.upper()
    before = y.nunique()
    log.append(f"Labels after whitespace/case cleaning: {sorted(y.dropna().unique().tolist())}")
    if before > 20:
        warn.append(f"{before} different labels found. Is the label column right? Typos create fake classes (use --label-map).")
    return y


def sample_cleaning(X, y, meta, cfg, log, warn):
    keep = pd.Series(True, index=X.index)
    n0 = len(X)
    bad_lab = (y.isna() | (y.fillna("") == "") | y.fillna("").str.lower().isin(["nan", "na", "none", "null", "unknown"]))
    bad_lab = pd.Series(bad_lab.to_numpy(dtype=bool), index=X.index)
    keep &= ~bad_lab
    log.append(f"Dropped {int(bad_lab.sum())} sample(s) with missing/unknown label.")
    ms = X.isna().mean(1)
    hm = pd.Series((ms > cfg.max_missing_sample).to_numpy(dtype=bool), index=X.index)
    keep &= ~hm
    log.append(f"Dropped {int((hm & ~bad_lab).sum())} sample(s) with > {cfg.max_missing_sample:.0%} missing values.")
    # duplicates: identical values
    vals = X.round(6).astype(str)
    key = pd.util.hash_pandas_object(vals, index=False)
    d = pd.DataFrame({"key": key.values, "label": y.values, "keep": keep.values}, index=X.index)
    d = d[d.keep]
    conflict_keys = set(d.groupby("key").label.nunique()[lambda s: s > 1].index)
    is_conf = d.key.isin(conflict_keys)
    dup_same = d[~is_conf].duplicated("key", keep="first")
    drop_idx = set(d[is_conf].index) | set(d[~is_conf][dup_same].index)
    keep.loc[list(drop_idx)] = False
    log.append(f"Removed {int(dup_same.sum())} exact duplicate sample(s) (same values, same label)"
               f" and {int(is_conf.sum())} sample(s) whose identical values carry CONFLICTING labels.")
    if is_conf.sum():
        warn.append(f"{int(is_conf.sum())} samples had identical expression but different labels: label errors are likely in your file.")
    # rare classes
    cnt = y[keep].value_counts()
    rare = cnt[cnt < cfg.min_class_size].index.tolist()
    if rare:
        keep &= ~y.isin(rare)
        warn.append(f"Dropped class(es) {rare}: fewer than {cfg.min_class_size} samples (cannot be learned or tested reliably).")
    log.append(f"Samples kept after cleaning: {int(keep.sum())} of {n0}.")
    return X[keep], y[keep], meta[keep]


def split_data(X, y, meta, test_size, seed, log, warn):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    idx = np.arange(len(X))
    if "group" in meta.columns and meta.group.nunique() < len(X):
        k = max(2, int(round(1 / test_size)))
        sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        tr, te = next(sgk.split(idx, y.values, meta.group.values))
        log.append(f"Split by patient/group ({meta.group.nunique()} groups): no group appears in both train and test.")
    else:
        cnt = y.value_counts()
        strat = y.values if cnt.min() >= 2 else None
        if strat is None:
            warn.append("A class has a single sample: split is not stratified.")
        tr, te = train_test_split(idx, test_size=test_size, stratify=strat, random_state=seed)
        if "group" not in meta.columns:
            warn.append("No --group-col given: if one patient has several samples they may land in BOTH train and test and inflate the test score. Pass --group-col patient_id if you have it.")
    return tr, te


def outlier_scores(M: pd.DataFrame) -> pd.Series:
    """correlation of each sample with the median profile; robust z-score (negative = odd sample)."""
    med = M.median(0).to_numpy()
    C = np.array([np.corrcoef(r, med)[0, 1] for r in M.to_numpy()])
    C = np.nan_to_num(C, nan=0.0)
    mad = np.median(np.abs(C - np.median(C))) * 1.4826 + 1e-9
    return pd.Series((C - np.median(C)) / mad, index=M.index)


def cramers_v(a, b) -> float:
    from scipy.stats import chi2_contingency
    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def qc_warnings(X, y, meta, warn, n_test=None):
    n, g = X.shape
    cnt = y.value_counts()
    if g > 20 * n:
        warn.append(f"{g} genes for only {n} samples: models can over-fit; use cross-validation (hilbertfuse cv) before trusting any score.")
    if cnt.min() < 10:
        warn.append(f"Smallest class has {int(cnt.min())} samples: expect unstable accuracy.")
    if cnt.max() / cnt.min() > 5:
        warn.append(f"Class imbalance {cnt.max()}:{cnt.min()}. Report macro-F1 and per-class accuracy, not accuracy alone. Majority-class baseline = {cnt.max()/n:.3f}.")
    if n_test is not None and n_test < 20:
        warn.append(f"Test set has only {n_test} samples: one error moves accuracy by {100/n_test:.1f} points. Prefer cross-validation.")
    if "batch" in meta.columns:
        v = cramers_v(meta.batch, y)
        if v > 0.5:
            warn.append(f"Batch and class are strongly linked (Cramer's V = {v:.2f}): the model may learn the batch (lab/platform) instead of the disease. Normalisation cannot fix this; add samples so each batch contains several classes.")
        else:
            warn.append(f"Batch/class association: Cramer's V = {v:.2f} (<= 0.5 is acceptable). Consider batch correction (e.g. ComBat) outside this tool if batches differ visibly.") if v > 0.3 else None


# ------------------------------------------------------------------ commands
def report(path, title, log, warn, extra):
    md = [f"# {title}", "", "## What was done"] + [f"* {x}" for x in log] + ["", "## Warnings (read these)"]
    md += [f"* {w}" for w in warn] if warn else ["* none"]
    md += ["", "## Summary"] + [f"* {k}: {v}" for k, v in extra.items()]
    with open(path + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(dict(log=log, warnings=warn, summary=extra, version=VERSION), f, indent=2, default=str)
    print("\n".join(md))


def cfg_from(args) -> Config:
    return Config(data_type=args.data_type, normalize=args.normalize, scale=args.scale,
                  max_missing_gene=args.max_missing_gene, max_missing_sample=args.max_missing_sample,
                  min_expr_frac=args.min_expr_frac, near_const_frac=args.near_const_frac, top_var=args.top_var,
                  min_class_size=args.min_class_size, label_case=args.label_case, strip_version=args.strip_version,
                  uppercase_genes=args.uppercase_genes, drop_outliers=args.drop_outliers)


def cmd_fit(args):
    cfg = cfg_from(args)
    log, warn = [], []
    X, y, meta = load_and_clean(args, cfg, log, warn)
    if y is None:
        raise SystemExit("ERROR: a label column is required for `fit` (use --label-col).")
    y = clean_labels(y, cfg, args.label_map, log, warn)
    X, y, meta = sample_cleaning(X, y, meta, cfg, log, warn)
    if len(X) < 10:
        raise SystemExit(f"ERROR: only {len(X)} usable samples left.")
    os.makedirs(args.out, exist_ok=True)
    if args.no_split:
        tr, te = np.arange(len(X)), np.array([], dtype=int)
    else:
        tr, te = split_data(X, y, meta, args.test_size, args.seed, log, warn)
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    prep = DataPrep(cfg).fit(Xtr)
    Mtr = prep.transform(Xtr)
    osc = outlier_scores(Mtr)
    out_tr = osc < -4
    if out_tr.any():
        names = meta["id"][Xtr.index[out_tr.values]].tolist() if "id" in meta.columns else Xtr.index[out_tr.values].tolist()
        if cfg.drop_outliers:
            Xtr, ytr = Xtr[~out_tr.values], ytr[~out_tr.values]
            prep = DataPrep(cfg).fit(Xtr)
            Mtr = prep.transform(Xtr)
            log.append(f"Dropped {int(out_tr.sum())} outlier training sample(s) (correlation with the median profile far below the rest) and re-fitted.")
        else:
            warn.append(f"{int(out_tr.sum())} training sample(s) look like outliers (e.g. {names[:3]}); inspect them or use --drop-outliers.")
    log.append(f"Data type: {prep.data_type_}; normalisation: {prep.norm_}" + ("; log2 applied" if prep.log_ else "")
               + f"; gene scaling: {cfg.scale}.")
    log.append(f"Genes: {X.shape[1]} read -> {len(prep.genes_in_)} after missing-value and constant-gene filters -> {len(prep.genes_out_)} kept "
               f"(then removed near-constant{', low-expressed' if prep.data_type_ == 'counts' else ''} genes"
               f"{f', then top {cfg.top_var} by variance' if cfg.top_var else ''}). Missing values were filled with training medians.")
    Mtr = Mtr.copy()
    Mtr["label"] = ytr.values
    Mtr.to_csv(os.path.join(args.out, "train.csv.gz"), index=False, compression="gzip")
    pd.DataFrame({"sample": (meta["id"][Xtr.index] if "id" in meta.columns else Xtr.index), "label": ytr.values}).to_csv(
        os.path.join(args.out, "samples_train.csv"), index=False)
    n_te = len(te)
    if n_te:
        Xte, yte = X.iloc[te], y.iloc[te]
        Mte = prep.transform(Xte)
        o_te = outlier_scores(Mte) < -4
        if o_te.any():
            warn.append(f"{int(o_te.sum())} test sample(s) look like outliers (not removed: test data is never filtered).")
        Mte = Mte.copy(); Mte["label"] = yte.values
        Mte.to_csv(os.path.join(args.out, "test.csv.gz"), index=False, compression="gzip")
        pd.DataFrame({"sample": (meta["id"][Xte.index] if "id" in meta.columns else Xte.index), "label": yte.values}).to_csv(
            os.path.join(args.out, "samples_test.csv"), index=False)
    qc_warnings(Xtr, ytr, meta.loc[Xtr.index], warn, n_te if n_te else None)
    open(os.path.join(args.out, "genes_kept.txt"), "w").write("\n".join(prep.genes_out_))
    joblib.dump(dict(prep=prep, cfg=cfg, version=VERSION, label_col=args.label_col, gene_names_cfg=asdict(cfg)),
                os.path.join(args.out, "prep_model.joblib"))
    extra = {"train samples": len(Xtr), "test samples": n_te, "genes kept": len(prep.genes_out_),
             "train class counts": ytr.value_counts().to_dict(),
             "majority-class baseline (train)": round(ytr.value_counts().max() / len(ytr), 3)}
    if n_te:
        extra["test class counts"] = yte.value_counts().to_dict()
    nxt = [f"hilbertfuse cv --data {os.path.join(args.out, 'train.csv.gz')} --label-col label --folds 5 --preset linear",
           f"hilbertfuse train --train {os.path.join(args.out, 'train.csv.gz')}" + (f" --test {os.path.join(args.out, 'test.csv.gz')}" if n_te else "")
           + " --label-col label --preset linear --out models/mymodel",
           f"python dataprep/prepare_data.py apply --prep {os.path.join(args.out, 'prep_model.joblib')} --input new_raw.csv --out new_clean.csv   # same cleaning for new samples"]
    extra["next commands"] = " | ".join(nxt)
    report(os.path.join(args.out, "prep_report"), "Data preparation report", log, warn, extra)
    print(f"\nFiles in {args.out}: train.csv.gz" + (", test.csv.gz" if n_te else "") + ", prep_report.md, prep_model.joblib, genes_kept.txt, samples_*.csv")


def cmd_check(args):
    cfg = cfg_from(args)
    log, warn = [], []
    X, y, meta = load_and_clean(args, cfg, log, warn)
    if y is None:
        raise SystemExit("ERROR: --label-col not found.")
    y = clean_labels(y, cfg, args.label_map, log, warn)
    X, y, meta = sample_cleaning(X, y, meta, cfg, log, warn)
    A = X.to_numpy()
    log.append(f"Detected data type: {DataPrep._detect(np.where(np.isnan(A), np.nanmedian(A, 0), A))} "
               f"(value range {np.nanmin(A):.3g} to {np.nanmax(A):.3g}, {np.isnan(A).mean():.2%} missing).")
    qc_warnings(X, y, meta, warn)
    if "group" not in meta.columns:
        warn.append("No --group-col given (patient id). If one patient has several samples, add it to avoid train/test leakage.")
    os.makedirs(args.out or ".", exist_ok=True)
    report(os.path.join(args.out or ".", "check_report"), "Data check (nothing was changed)", log, warn,
           {"samples after cleaning": len(X), "genes": X.shape[1], "class counts": y.value_counts().to_dict()})


def cmd_apply(args):
    obj = joblib.load(args.prep)
    prep, cfg = obj["prep"], obj["cfg"]
    log, warn = [], []
    args.label_col = args.label_col or obj.get("label_col")
    X, y, meta = load_and_clean(args, cfg, log, warn)
    have = len(set(prep.genes_in_) & set(X.columns)) / len(prep.genes_in_)
    log.append(f"Gene coverage: {have:.1%} of the {len(prep.genes_in_)} genes used in training are present in your file.")
    if have < 0.5 and not args.force:
        raise SystemExit("ERROR: your file lacks most training genes (different gene names/platform?). Fix the gene names or use --force.")
    if have < 0.95:
        warn.append(f"{(1 - have):.0%} of training genes are missing in your file; they were filled with training medians (predictions less reliable).")
    M = prep.transform(X)
    if y is not None:
        M = M.copy(); M["label"] = clean_labels(y, cfg, None, [], []).values
    out = args.out if args.out.endswith((".csv", ".gz")) else args.out + ".csv"
    M.to_csv(out, index=False)
    log.append(f"Wrote {len(M)} cleaned sample(s) with {len(prep.genes_out_)} genes to `{out}` (same cleaning as the training data).")
    report(os.path.splitext(out)[0].replace(".csv", "") + "_apply_report", "Apply report", log, warn, {"samples": len(M)})


def build_parser():
    ap = argparse.ArgumentParser(prog="prepare_data.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, fit=True):
        p.add_argument("--input", required=True, help="raw CSV/TSV/Parquet: rows = samples, columns = genes (+ label)")
        p.add_argument("--label-col", default="label" if fit else None)
        p.add_argument("--id-col", help="column with sample ids (kept out of the genes)")
        p.add_argument("--group-col", help="patient id column: all samples of one patient stay on the same side of the split")
        p.add_argument("--batch-col", help="lab/platform/batch column: only used for the confounding check")
        p.add_argument("--transpose", action="store_true", help="file is genes x samples (first column = gene names)")
        p.add_argument("--data-type", choices=["auto", "counts", "intensity", "log"], default="auto",
                       help="counts = raw RNA-seq counts; intensity = microarray; log = already log-scaled")
        p.add_argument("--normalize", choices=["auto", "none", "cpm", "median", "quantile"], default="auto")
        p.add_argument("--scale", choices=["none", "zscore"], default="none")
        p.add_argument("--max-missing-gene", type=float, default=0.2)
        p.add_argument("--max-missing-sample", type=float, default=0.2)
        p.add_argument("--min-expr-frac", type=float, default=0.2)
        p.add_argument("--near-const-frac", type=float, default=0.95)
        p.add_argument("--top-var", type=int, default=0, help="keep only the N most variable genes (0 = all)")
        p.add_argument("--min-class-size", type=int, default=5)
        p.add_argument("--label-case", choices=["auto", "keep", "lower", "upper"], default="auto")
        p.add_argument("--label-map", help="CSV with two columns old,new to merge/rename labels")
        p.add_argument("--strip-version", action="store_true", help="ENSG00000123456.7 -> ENSG00000123456")
        p.add_argument("--uppercase-genes", action="store_true")
        p.add_argument("--drop-outliers", action="store_true")
    f = sub.add_parser("fit", help="clean + normalise + split; writes train/test files, report and a reusable prep model")
    common(f)
    f.add_argument("--out", default="clean_data")
    f.add_argument("--test-size", type=float, default=0.2)
    f.add_argument("--no-split", action="store_true", help="only clean; write everything to train.csv.gz")
    f.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("check", help="report problems in a raw file without changing anything")
    common(c)
    c.add_argument("--out", default=".")
    a = sub.add_parser("apply", help="apply a saved prep model to new samples (labels optional)")
    a.add_argument("--prep", required=True); a.add_argument("--input", required=True); a.add_argument("--out", default="new_clean.csv")
    a.add_argument("--label-col", default=None); a.add_argument("--id-col"); a.add_argument("--group-col"); a.add_argument("--batch-col")
    a.add_argument("--transpose", action="store_true"); a.add_argument("--force", action="store_true")
    return ap


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    {"fit": cmd_fit, "check": cmd_check, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
