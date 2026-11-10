"""HybridClassifier: HilbertFuseNet (or its ensemble) + tuned linear model, probabilities averaged.

Rationale: on small cohorts (<~200 samples) tuned linear models are very strong while CNNs on gene images
have high variance; their errors differ, so a blend is usually at least as good as the better one.
The linear member sees the same leak-free MI-selected genes (no layout needed).
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .pipeline import EnsembleClassifier, PanCancerClassifier, _score
from .preprocess import GeneImagePreprocessor


class LinearHead:
    """MI top-n genes -> standardise -> logistic regression with CV-tuned C."""

    def __init__(self, n_genes=1024, seed=0):
        self.n_genes, self.seed = n_genes, seed

    def fit(self, X: pd.DataFrame, y):
        self.classes_ = sorted(set(y))
        yi = np.array([self.classes_.index(v) for v in y])
        self.pre_ = GeneImagePreprocessor(self.n_genes, selection="mi", layout="rowmajor", channels="expr",
                                          random_state=self.seed).fit(X, yi)
        F = self._feat(X)
        self.sc_ = StandardScaler().fit(F)
        cv = max(2, int(min(3, np.bincount(yi).min())))
        self.lr_ = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], cv=cv, max_iter=3000).fit(self.sc_.transform(F), yi)
        return self

    def _feat(self, X):
        return self.pre_.transform(X)[:, 0].reshape(len(X), -1)[:, self.pre_.pix_]

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.sc_.transform(self._feat(X)))


class HybridClassifier:
    def __init__(self, network, n_genes: int = 1024, weight: float = 0.5, seed: int = 0):
        """network: PanCancerClassifier or EnsembleClassifier; weight = share of the network."""
        self.network, self.weight = network, weight
        self.linear = LinearHead(n_genes, seed)
        self.classes_: list = []

    @property
    def transpose(self):
        n = self.network
        return n.kwargs.get("transpose", False) if isinstance(n, EnsembleClassifier) else n.transpose

    def fit(self, train, label_col="label", val=None):
        X, y = PanCancerClassifier(transpose=self.transpose)._xy(train, label_col)
        self.network.fit((X, y), label_col, val=val)
        self.linear.fit(X, y)
        self.classes_ = self.network.classes_
        assert self.linear.classes_ == list(self.classes_)
        return self

    def predict_proba(self, data, label_col="label"):
        X, _ = PanCancerClassifier(transpose=self.transpose)._xy(data, label_col, need_y=False)
        return self.weight * self.network.predict_proba((X, None)) + (1 - self.weight) * self.linear.predict_proba(X)

    def predict(self, data, label_col="label"):
        return np.array(self.classes_)[self.predict_proba(data, label_col).argmax(1)]

    def evaluate(self, test, label_col="label"):
        X, y = PanCancerClassifier(transpose=self.transpose)._xy(test, label_col)
        idx = {c: i for i, c in enumerate(self.classes_)}
        return _score(self.predict_proba((X, None)), np.array([idx[v] for v in y]), self.classes_)

    def explain(self, *a, **k):
        return self.network.explain(*a, **k)       # attributions come from the CNN member

    def attribution_maps(self, *a, **k):
        return self.network.attribution_maps(*a, **k)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.network.save(os.path.join(path, "network"))
        joblib.dump((self.linear, self.weight), os.path.join(path, "linear.joblib"))
        import json
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(dict(kind="hybrid", weight=self.weight, classes=list(self.classes_)), f)
        return path

    @classmethod
    def load(cls, path, device=None):
        net = PanCancerClassifier.load(os.path.join(path, "network"), device)
        linear, weight = joblib.load(os.path.join(path, "linear.joblib"))
        obj = cls(net, weight=weight)
        obj.linear, obj.classes_ = linear, net.classes_
        return obj


class LinearClassifier:
    """Quick mode: only the tuned logistic regression on MI-selected genes (trains in seconds on any CPU).
    explain() ranks genes by the standardised regression coefficients."""

    def __init__(self, n_genes: int = 1024, seed: int = 0, transpose: bool = False):
        self.head = LinearHead(n_genes, seed)
        self.transpose = transpose
        self.classes_: list = []

    @property
    def pre_(self):
        return self.head.pre_

    def fit(self, train, label_col="label", val=None):
        X, y = PanCancerClassifier(transpose=self.transpose)._xy(train, label_col)
        self.head.fit(X, y.to_numpy())
        self.classes_ = self.head.classes_
        return self

    def predict_proba(self, data, label_col="label"):
        X, _ = PanCancerClassifier(transpose=self.transpose)._xy(data, label_col, need_y=False)
        return self.head.predict_proba(X)

    def predict(self, data, label_col="label"):
        return np.array(self.classes_)[self.predict_proba(data, label_col).argmax(1)]

    def evaluate(self, test, label_col="label"):
        X, y = PanCancerClassifier(transpose=self.transpose)._xy(test, label_col)
        idx = {c: i for i, c in enumerate(self.classes_)}
        return _score(self.predict_proba((X, None)), np.array([idx[v] for v in y]), self.classes_)

    def explain(self, data=None, label_col="label", top_k: int = 40, **kw):
        coef = self.head.lr_.coef_
        if coef.shape[0] == 1:                         # binary: one row -> class 1; class 0 is its mirror
            coef = np.vstack([-coef[0], coef[0]])
        genes = np.array(self.head.pre_.gene_names_)
        rows = []
        for c, cname in enumerate(self.classes_):
            others = np.delete(coef, c, axis=0).mean(0)
            for r, g in enumerate(np.argsort(-coef[c])[:top_k], 1):
                rows.append((cname, r, genes[g], float(coef[c, g]), float(coef[c, g] - others[g])))
        per_class = pd.DataFrame(rows, columns=["class", "rank", "gene", "score", "specificity"])
        gs = np.abs(coef).mean(0)
        gi = np.argsort(-gs)[:top_k]
        return {"per_class": per_class,
                "global": pd.DataFrame({"rank": range(1, len(gi) + 1), "gene": genes[gi], "score": gs[gi]})}

    def save(self, path):
        import json
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.head, os.path.join(path, "linear_only.joblib"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(dict(kind="linear", classes=list(self.classes_), transpose=self.transpose), f)
        return path

    @classmethod
    def load(cls, path, device=None):
        import json
        meta = json.load(open(os.path.join(path, "meta.json")))
        obj = cls(transpose=meta.get("transpose", False))
        obj.head = joblib.load(os.path.join(path, "linear_only.joblib"))
        obj.classes_ = meta["classes"]
        return obj
