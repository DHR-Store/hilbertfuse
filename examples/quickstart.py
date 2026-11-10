"""End-to-end demo with HilbertFuseNet on a REAL cancer dataset (bundled): python examples/quickstart.py"""
import numpy as np
from sklearn.model_selection import train_test_split
from hilbertfuse import load_table, preset
from hilbertfuse.datasets import load

df = load("khan")                                   # 63 samples, 2308 genes, 4 tumor types
X, y = load_table(df)
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)

clf = preset("fuse", n_genes=1024, epochs=60)         # HilbertFuseNet: Hilbert layout + MI/RD + 3 channels
clf.fit((X.iloc[tr], y.iloc[tr].values))
m = clf.evaluate((X.iloc[te], y.iloc[te].values))
print("accuracy", round(m["accuracy"], 3), "macro-F1", round(m["f1_macro"], 3))
print(clf.explain((X.iloc[te], y.iloc[te].values), top_k=10)["global"])
clf.save("runs/khan_hilbertfusenet")
