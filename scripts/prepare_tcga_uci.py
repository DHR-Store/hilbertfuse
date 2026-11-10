"""Prepare the UCI 'gene expression cancer RNA-Seq' data (TCGA PANCAN HiSeq, 801 samples x 20,531 genes,
5 tumor types: BRCA KIRC COAD LUAD PRAD) for hilbertfuse.

1. Download and unzip from https://archive.ics.uci.edu/dataset/401  -> data.csv and labels.csv
2. python scripts/prepare_tcga_uci.py --data data.csv --labels labels.csv --out tcga5
   -> tcga5/train.csv.gz and tcga5/test.csv.gz (stratified 80/20) and tcga5/all.csv.gz
NOTE: written from the published file layout; it could not be tested on the real files in the build sandbox
(no internet access to that host). If a column name differs, the error message will say which.
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="tcga5"); ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

X = pd.read_csv(a.data, index_col=0)
lab = pd.read_csv(a.labels, index_col=0)
col = "Class" if "Class" in lab.columns else lab.columns[-1]
if len(X) != len(lab):
    raise SystemExit(f"data has {len(X)} rows but labels has {len(lab)}")
X["label"] = lab.loc[X.index, col].astype(str).values
os.makedirs(a.out, exist_ok=True)
tr, te = train_test_split(X, test_size=a.test_size, stratify=X.label, random_state=a.seed)
X.to_csv(os.path.join(a.out, "all.csv.gz"), compression="gzip")
tr.to_csv(os.path.join(a.out, "train.csv.gz"), index=False, compression="gzip")
te.to_csv(os.path.join(a.out, "test.csv.gz"), index=False, compression="gzip")
print(X.label.value_counts().to_string()); print("wrote", a.out)
