"""Does preparation help?  Measured on REAL data sets with synthetic mess added (optional script; needs hilbertfuse).
   python dataprep/validate_with_hilbertfuse.py --datasets golub,alon,khan,shipp --out dataprep/validation.csv
Variants (same 5 folds, same model = tuned logistic regression on 1024 MI-selected genes, everything re-fit per fold):
  clean          original file, no preparation
  clean+prep     original file, prepared (fit on the training fold)
  messy          mess added (per-sample scale x lognormal(0,0.5), 3 % missing values, constant genes), no preparation
  messy+prep     same mess, prepared"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_data as pdp
from hilbertfuse import datasets as ds
from hilbertfuse.hybrid import LinearHead


def make_messy(X, seed):
    rng = np.random.default_rng(seed)
    M = X.mul(np.exp(rng.normal(0, 0.5, len(X))), axis=0)
    M = M.mask(rng.random(M.shape) < 0.03)
    M["CONST_1"], M["CONST_2"] = 7.0, 0.0
    return M


def run(Xtr, ytr, Xte, yte, prep):
    if prep:
        p = pdp.DataPrep(pdp.Config()).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    else:                                   # what hilbertfuse does with a raw file: missing -> 0
        Xtr, Xte = Xtr.fillna(0.0), Xte.fillna(0.0)
    head = LinearHead(1024, 0).fit(Xtr, ytr)
    pred = np.array(head.classes_)[head.predict_proba(Xte).argmax(1)]
    return float((pred == yte).mean())


ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="golub,alon,khan,shipp"); ap.add_argument("--out", default="validation.csv")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
rows = []
if os.path.exists(a.out):
    rows = pd.read_csv(a.out).to_dict("records")
done = {(r["dataset"], r["variant"], r["fold"]) for r in rows}
for name in a.datasets.split(","):
    df = ds.load(name)
    y = df.pop("label").astype(str).to_numpy()
    X = df.astype(float)
    Xm = make_messy(X, 1)
    for f, (tr, te) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=0).split(X, y)):
        for variant, D, prep in (("clean", X, False), ("clean+prep", X, True), ("messy", Xm, False), ("messy+prep", Xm, True)):
            if (name, variant, f) in done:
                continue
            acc = run(D.iloc[tr], y[tr], D.iloc[te], y[te], prep)
            rows.append(dict(dataset=name, variant=variant, fold=f, accuracy=acc))
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{name:8s} {variant:11s} fold {f} acc={acc:.3f}", flush=True)
r = pd.DataFrame(rows)
t = r.groupby(["dataset", "variant"]).accuracy.mean().unstack()[["clean", "clean+prep", "messy", "messy+prep"]]
t.loc["mean"] = t.mean()
print((t * 100).round(1).to_string())
