"""Build example_messy_raw.csv: a small, deliberately messy file made from the public Golub leukemia data
(40 samples x 600 genes) so you can try the tool. The mess is synthetic and documented below.
   python make_example.py            (needs ../datasets/golub.csv.gz)"""
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
g = pd.read_csv("../datasets/golub.csv.gz")
y = g.pop("label")
idx = np.concatenate([rng.choice(np.where(y == "ALL")[0], 24, replace=False), rng.choice(np.where(y == "AML")[0], 16, replace=False)])
genes = rng.choice(g.columns, 600, replace=False)
X = g.iloc[idx][genes].reset_index(drop=True).astype(float)
lab = y.iloc[idx].reset_index(drop=True)
X = X.mul(np.exp(rng.normal(0, 0.5, len(X))), axis=0)                 # 1) per-sample scale differences (library-size-like)
X = X.mask(rng.random(X.shape) < 0.03)                                # 2) 3 % random missing values
X.iloc[5, : int(0.45 * X.shape[1])] = np.nan                          # 3) one sample with 45 % missing
X["CONST_1"], X["CONST_2"] = 7.0, 0.0                                 # 4) constant genes
X[genes[0] + " "] = X[genes[0]] * 1.001                               # 5) duplicated gene (name differs by a space)
X = X.rename(columns={genes[1]: genes[1].lower()})                    # 6) inconsistent gene-name case
labels = [("  " + v.lower() if rng.random() < .15 else v + (" " if rng.random() < .15 else "")) for v in lab]  # 7) label typos
X["patient_id"] = [f"P{i:03d}" for i in range(len(X))]
X["batch"] = rng.choice(["A", "B"], len(X))
X["label"] = labels
dup = X.iloc[[3]].copy()                                              # 8) the same sample twice
X = pd.concat([X, dup], ignore_index=True)
X.round(3).to_csv("example_messy_raw.csv", index=False)
print("wrote example_messy_raw.csv", X.shape, "labels:", sorted(set(X.label)))
