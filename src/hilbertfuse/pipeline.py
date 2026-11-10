"""High-level API: PanCancerClassifier, EnsembleClassifier, cross_validate, presets."""
from __future__ import annotations
import dataclasses
import json
import os
from typing import Optional, Sequence
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .data import TableLike, load_table
from .explain import GuidedGradCAM, gene_importance
from .models import build_model
from .preprocess import GeneImagePreprocessor
from .train import TrainConfig, fit_model, metrics_from_preds, pick_device, predict_proba

_TV_PRETRAINED = {"hilbertfusenet121", "densenet121", "densenet169", "densenet201", "densenet161",
                  "resnet18", "resnet34", "resnet50"}


class PanCancerClassifier:
    """Train your own gene-image classifier.

    Defaults = HilbertFuseNet (compact, from scratch) + MI/redundancy-pruned genes + Hilbert layout +
    3-channel images.  Use `preset("paper")` to reproduce the paper-style configuration.
    """

    def __init__(self, backbone: str = "hilbertfusenet", pretrained: Optional[bool] = None, n_genes: int = 3600,
                 gene_annotation: Optional[TableLike] = None, input_size: int = 96,
                 mi_prefilter: Optional[int] = 8000, selection: str = "mi_rd", rd_threshold: float = 0.9,
                 layout: str = "hilbert", channels: str = "multi", transform: str = "auto",
                 dropout: float = 0.2, model_kwargs: Optional[dict] = None,
                 train_config: Optional[TrainConfig] = None, device: Optional[str] = None,
                 transpose: bool = False, random_state: int = 0):
        self.backbone = backbone
        self.pretrained = (backbone in _TV_PRETRAINED) if pretrained is None else pretrained
        self.n_genes, self.gene_annotation, self.input_size = n_genes, gene_annotation, input_size
        self.mi_prefilter, self.selection, self.rd_threshold = mi_prefilter, selection, rd_threshold
        self.layout, self.channels, self.transform = layout, channels, transform
        self.dropout, self.model_kwargs = dropout, dict(model_kwargs or {})
        self.cfg = train_config or TrainConfig(seed=random_state)
        self.device = pick_device(device)
        self.transpose, self.random_state = transpose, random_state
        self.pre_ = None
        self.model_ = None
        self.classes_: list = []
        self.history_: list = []

    # ------------------------------------------------------------------ helpers
    def _xy(self, data, label_col, need_y=True):
        if isinstance(data, tuple):
            X, y = data
            return X, (None if y is None else pd.Series(np.asarray(y)).astype(str))
        X, y = load_table(data, label_col=label_col, transpose=self.transpose)
        if need_y and y is None:
            raise ValueError(f"label column '{label_col}' not found")
        return X, y

    def _encode(self, y):
        idx = {c: i for i, c in enumerate(self.classes_)}
        unk = sorted(set(y) - set(idx))
        if unk:
            raise ValueError(f"labels not seen in training: {unk}")
        return np.array([idx[v] for v in y], dtype=np.int64)

    def _new_pre(self):
        return GeneImagePreprocessor(self.n_genes, self.gene_annotation, self.mi_prefilter,
                                     selection=self.selection, rd_threshold=self.rd_threshold,
                                     layout=self.layout, channels=self.channels,
                                     transform=self.transform, random_state=self.random_state)

    def _new_model(self, n_classes, in_channels):
        return build_model(self.backbone, n_classes, self.pretrained, self.input_size, self.dropout,
                           in_channels, **self.model_kwargs)

    # ---------------------------------------------------------------------- fit
    def fit(self, train, label_col: str = "label", val=None):
        X, y = self._xy(train, label_col)
        self.classes_ = sorted(set(y))
        yi = self._encode(y)
        self.pre_ = self._new_pre()
        imgs = self.pre_.fit_transform(X, yi)
        self.model_ = self._new_model(len(self.classes_), self.pre_.n_channels)
        cfg = self.cfg
        if not self.pretrained and cfg.warmup_epochs:      # nothing to warm up from random init
            cfg = dataclasses.replace(cfg, warmup_epochs=0)
        v = None
        if val is not None:
            Xv, yv = self._xy(val, label_col)
            v = (self.pre_.transform(Xv), self._encode(yv))
        self.history_ = fit_model(self.model_, imgs, yi, cfg, self.device, val=v)
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, data, label_col="label") -> np.ndarray:
        X, _ = self._xy(data, label_col, need_y=False)
        return predict_proba(self.model_, self.pre_.transform(X), self.device)

    def predict(self, data, label_col="label") -> np.ndarray:
        return np.array(self.classes_)[self.predict_proba(data, label_col).argmax(1)]

    def evaluate(self, test, label_col="label") -> dict:
        X, y = self._xy(test, label_col)
        return _score(self.predict_proba((X, None)), self._encode(y), self.classes_)

    # ------------------------------------------------------------------ explain
    def explain(self, data, label_col="label", top_k: int = 40, method: str = "guided_gradcam",
                only_correct: bool = True, max_per_class: int = 200) -> dict:
        X, y = self._xy(data, label_col)
        yi = self._encode(y)
        rng = np.random.default_rng(self.random_state)
        keep = np.concatenate([rng.permutation(np.where(yi == c)[0])[:max_per_class]
                               for c in range(len(self.classes_))]) if max_per_class else np.arange(len(yi))
        imgs = self.pre_.transform(X.iloc[keep])
        yk = yi[keep]
        pred = predict_proba(self.model_, imgs, self.device).argmax(1)
        attr = GuidedGradCAM(self.model_, self.device).attribute(imgs, yk, method=method)
        return gene_importance(attr, yk, pred, self.classes_, self.pre_.gene_names_, top_k,
                               only_correct, pixel_index=self.pre_.pix_)

    def attribution_maps(self, data, label_col="label", method="guided_gradcam", max_per_class=100,
                         use_predicted=False) -> dict:
        """Raw Guided Grad-CAM maps (N, S, S) for heat maps, plus everything needed to map pixels back to genes."""
        X, y = self._xy(data, label_col, need_y=False)
        imgs = self.pre_.transform(X)
        pred = predict_proba(self.model_, imgs, self.device).argmax(1)
        tgt = pred if (y is None or use_predicted) else self._encode(y)
        rng = np.random.default_rng(self.random_state)
        keep = np.concatenate([rng.permutation(np.where(tgt == c)[0])[:max_per_class]
                               for c in range(len(self.classes_))]).astype(int)
        keep.sort()
        attr = GuidedGradCAM(self.model_, self.device).attribute(imgs[keep], tgt[keep], method=method)
        return dict(attr=attr, target=tgt[keep], pred=pred[keep], index=keep, images=imgs[keep],
                    classes=list(self.classes_), pix=self.pre_.pix_, genes=list(self.pre_.gene_names_),
                    side=self.pre_.side_)

    # --------------------------------------------------------------------- I/O
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model_.state_dict(), os.path.join(path, "model.pt"))
        joblib.dump(self.pre_, os.path.join(path, "preprocessor.joblib"))
        meta = dict(kind="single", backbone=self.backbone, n_genes=self.n_genes, input_size=self.input_size,
                    dropout=self.dropout, classes=self.classes_, train_config=self.cfg.to_dict(),
                    transpose=self.transpose, random_state=self.random_state, channels=self.channels,
                    layout=self.layout, selection=self.selection, model_kwargs=self.model_kwargs)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        pd.DataFrame(self.history_).to_csv(os.path.join(path, "history.csv"), index=False)
        self.pre_.mi_scores_.rename("mutual_information").to_csv(os.path.join(path, "mi_scores.csv"))
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if meta.get("kind") == "ensemble":
            return EnsembleClassifier.load(path, device)
        if meta.get("kind") == "linear":
            from .hybrid import LinearClassifier
            return LinearClassifier.load(path, device)
        if meta.get("kind") == "hybrid":
            from .hybrid import HybridClassifier
            return HybridClassifier.load(path, device)
        obj = cls(backbone=meta["backbone"], pretrained=False, n_genes=meta["n_genes"],
                  input_size=meta["input_size"], dropout=meta["dropout"], device=device,
                  transpose=meta.get("transpose", False), random_state=meta.get("random_state", 0),
                  channels=meta.get("channels", "expr"), layout=meta.get("layout", "rowmajor"),
                  selection=meta.get("selection", "mi"), model_kwargs=meta.get("model_kwargs"))
        obj.classes_ = meta["classes"]
        obj.pre_ = joblib.load(os.path.join(path, "preprocessor.joblib"))
        obj.model_ = obj._new_model(len(obj.classes_), obj.pre_.n_channels)
        obj.model_.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu"))
        obj.model_.eval().to(obj.device)
        return obj


def _score(P, yi, classes) -> dict:
    m = metrics_from_preds(yi, P.argmax(1), len(classes))
    m["per_class_accuracy"] = dict(zip(classes, np.round(m["per_class_accuracy"], 4)))
    m["classes"] = list(classes)
    return m


class EnsembleClassifier:
    """Multi-view ensemble of HilbertFuseNet members.

    Each member differs in seed and in the number of selected genes (image size), so the average
    is far less sensitive to the single best image dimension (paper Table 6 fluctuates by 3.9 points).
    """

    def __init__(self, n_members: int = 3, gene_scales: Sequence[float] = (0.75, 1.0, 1.25),
                 n_genes: int = 1024, **kwargs):
        self.n_members, self.gene_scales, self.n_genes, self.kwargs = n_members, tuple(gene_scales), n_genes, kwargs
        self.members: list = []
        self.classes_: list = []

    def fit(self, train, label_col: str = "label", val=None):
        X, y = PanCancerClassifier(transpose=self.kwargs.get("transpose", False))._xy(train, label_col)
        self.classes_ = sorted(set(y))
        self.members = []
        base_seed = self.kwargs.get("random_state", 0)
        for i in range(self.n_members):
            kw = dict(self.kwargs)
            scale = self.gene_scales[i % len(self.gene_scales)]
            kw.update(n_genes=max(16, int(self.n_genes * scale)), random_state=base_seed + 101 * i)
            cfg = kw.get("train_config")
            if cfg is not None:
                kw["train_config"] = dataclasses.replace(cfg, seed=base_seed + 101 * i)
            self.members.append(PanCancerClassifier(**kw).fit((X, y), label_col, val=val))
        return self

    def predict_proba(self, data, label_col="label"):
        P = [m.predict_proba(data, label_col) for m in self.members]
        return np.mean(P, axis=0)

    def predict(self, data, label_col="label"):
        return np.array(self.classes_)[self.predict_proba(data, label_col).argmax(1)]

    def evaluate(self, test, label_col="label") -> dict:
        X, y = self.members[0]._xy(test, label_col)
        return _score(self.predict_proba((X, None)), self.members[0]._encode(y), self.classes_)

    def attribution_maps(self, *a, **k):
        return self.members[0].attribution_maps(*a, **k)          # maps of the first member (members use different layouts)

    def explain(self, data, label_col="label", top_k: int = 40, **kw) -> dict:
        res = [m.explain(data, label_col, top_k=10 ** 6, **kw) for m in self.members]
        g = pd.concat([r["global"].set_index("gene")["score"].rename(i) for i, r in enumerate(res)], axis=1).fillna(0)
        s = g.mean(axis=1).sort_values(ascending=False).head(top_k)
        glob = pd.DataFrame({"rank": range(1, len(s) + 1), "gene": s.index, "score": s.values})
        pc = pd.concat([r["per_class"] for r in res])
        pc = pc.groupby(["class", "gene"], as_index=False)[["score", "specificity"]].mean()
        pc["rank"] = pc.groupby("class")["score"].rank(ascending=False, method="first").astype(int)
        return {"global": glob, "per_class": pc[pc["rank"] <= top_k].sort_values(["class", "rank"])}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(os.path.join(path, f"member_{i}"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(dict(kind="ensemble", n_members=len(self.members), classes=self.classes_), f)
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        obj = cls(n_members=meta["n_members"])
        obj.classes_ = meta["classes"]
        obj.members = [PanCancerClassifier.load(os.path.join(path, f"member_{i}"), device)
                       for i in range(meta["n_members"])]
        return obj


# --------------------------------------------------------------------------- presets
def preset(name: str, n_genes: int = 1024, epochs: int = 80, batch_size: int = 16, device=None, seed: int = 0, **over):
    """Ready-made configurations (same network size and training budget for `paper` and `hfnet`).

    paper     : row-major layout, plain MI, 1 channel, plain compact DenseNet, no regularisers (control)
    hfnet     : HilbertFuseNet CNN only (Hilbert layout + MI/redundancy pruning + 3 channels + SE/multi-scale + mixup)
    hfnet_ens : 3-member multi-view ensemble of hfnet
    fuse      : HilbertFuse = hfnet averaged with a tuned logistic regression  (recommended default)
    fuse_ens  : hfnet_ens averaged with the linear model
    fast      : fuse with 512 genes / 30 epochs / batch 32 (quick CNN + linear)
    linear    : logistic regression only (seconds; no CNN, no gene image)
    """
    base = dict(n_genes=n_genes, device=device, random_state=seed)
    if name == "linear":                      # quick mode: seconds on any CPU
        from .hybrid import LinearClassifier
        return LinearClassifier(n_genes, seed)
    if name == "fast":                        # fast CNN + linear: 512 genes, 30 epochs, batch 32 (~3x faster than fuse)
        return preset("fuse", n_genes=512, epochs=30, batch_size=32, device=device, seed=seed, **over)
    if name == "paper":
        cfg = TrainConfig(epochs=epochs, batch_size=batch_size, label_smoothing=0.0, class_balance="none",
                          gene_dropout=0.0, noise_std=0.0, mixup_alpha=0.0, seed=seed, verbose=False, amp=False)
        kw = dict(backbone="densenet_paper", selection="mi", layout="rowmajor", channels="expr", train_config=cfg)
        return PanCancerClassifier(**{**base, **kw, **over})
    if name in ("hfnet", "hfnet_ens", "fuse", "fuse_ens"):
        cfg = TrainConfig(epochs=epochs, batch_size=batch_size, label_smoothing=0.1, class_balance="loss",
                          gene_dropout=0.1, noise_std=0.02, mixup_alpha=0.3, seed=seed, verbose=False, amp=False)
        kw = dict(backbone="hilbertfusenet", selection="mi_rd", layout="hilbert", channels="multi", train_config=cfg)
        net = (PanCancerClassifier(**{**base, **kw, **over}) if name in ("hfnet", "fuse")
               else EnsembleClassifier(n_members=3, **{**base, **kw, **over}))
        if name.startswith("fuse"):
            from .hybrid import HybridClassifier
            return HybridClassifier(net, n_genes=n_genes, seed=seed)
        return net
    raise ValueError(f"unknown preset {name}")


def cross_validate(data, label_col: str = "label", k: int = 5, transpose: bool = False,
                   preset_name: Optional[str] = None, **clf_kwargs) -> pd.DataFrame:
    """Leak-free k-fold CV: preprocessing is re-fit inside every fold."""
    X, y = load_table(data, label_col=label_col, transpose=transpose)
    y = y.to_numpy()
    rows = []
    seed = clf_kwargs.get("random_state", 0)
    for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y), 1):
        clf = preset(preset_name, **clf_kwargs) if preset_name else PanCancerClassifier(transpose=transpose, **clf_kwargs)
        clf.fit((X.iloc[tr], y[tr]), label_col)
        m = clf.evaluate((X.iloc[te], y[te]), label_col)
        rows.append(dict(fold=f, accuracy=m["accuracy"], precision_macro=m["precision_macro"],
                         recall_macro=m["recall_macro"], f1_macro=m["f1_macro"]))
        print(f"fold {f}/{k}: acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean()
    return df
