"""CLI: hilbertfuse {train,evaluate,predict,explain,cv,benchmark,datasets,demo-data}"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd

from .pipeline import EnsembleClassifier, PanCancerClassifier, cross_validate, preset
from .synthetic import make_synthetic
from .train import TrainConfig


def _add_model_args(p):
    p.add_argument("--label-col", default="label")
    p.add_argument("--preset", choices=["fuse", "fuse_ens", "hfnet", "hfnet_ens", "paper", "fast", "linear"], default="fuse",
                   help="fuse = HilbertFuse (CNN + linear, default) | fast = quick CNN+linear | linear = logistic regression only (seconds) | hfnet = CNN only | *_ens = 3-model ensemble | paper = paper-style control")
    p.add_argument("--backbone", help="override: hilbertfusenet|hilbertfusenet121|densenet_paper|densenet121|resnet18...")
    p.add_argument("--pretrained", action="store_true", help="use ImageNet weights (hilbertfusenet121 / torchvision backbones)")
    p.add_argument("--n-genes", type=int, default=1024)
    p.add_argument("--gene-annotation", help="CSV/TSV gene,chromosome[,start] (row-major layout only)")
    p.add_argument("--layout", choices=["rowmajor", "cluster", "hilbert"])
    p.add_argument("--selection", choices=["mi", "mi_rd"])
    p.add_argument("--channels", choices=["expr", "multi"])
    p.add_argument("--transpose", action="store_true", help="input files are genes x samples")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)


def _make(a):
    over = {k: v for k, v in dict(backbone=a.backbone, layout=a.layout, selection=a.selection,
                                  channels=a.channels, gene_annotation=a.gene_annotation).items() if v}
    if a.pretrained:
        over["pretrained"] = True
        if not a.backbone:
            over["backbone"] = "hilbertfusenet121"
    clf = preset(a.preset, n_genes=a.n_genes, epochs=a.epochs, batch_size=a.batch_size,
                 device=a.device, seed=a.seed, **over)
    if hasattr(clf, "head"):                      # linear (quick) mode: nothing to configure
        clf.transpose = a.transpose
        return clf
    net = clf.network if hasattr(clf, "network") else clf
    cfg = TrainConfig(**{**(net.cfg if hasattr(net, "cfg") else net.kwargs["train_config"]).to_dict(), "verbose": not (getattr(a, "quiet", False) or getattr(a, "dataset", None) == "all")})
    if hasattr(net, "cfg"):
        net.cfg, net.transpose = cfg, a.transpose
    else:
        net.kwargs["train_config"], net.kwargs["transpose"] = cfg, a.transpose
    return clf


def _show(m):
    print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()
                      if k != "confusion_matrix"}, indent=2, default=float))

BUNDLED = ["alon", "golub", "gordon", "khan", "pomeroy", "shipp", "singh", "su"]


def _real_demo(a):
    """Split a bundled real dataset, train, test, compare with logistic regression, save everything."""
    import time
    from sklearn.model_selection import train_test_split
    from . import datasets as ds
    from .data import load_table
    from .hybrid import LinearHead
    names = BUNDLED if a.dataset == "all" else [a.dataset]
    rows = []
    for n in names:
        t0 = time.time()
        df = ds.load(n)
        tr, te = train_test_split(df, test_size=a.test_size, stratify=df.label, random_state=a.seed)
        out = os.path.join(a.out, n)
        os.makedirs(out, exist_ok=True)
        ptr, pte = os.path.join(out, "train.csv.gz"), os.path.join(out, "test.csv.gz")
        tr.to_csv(ptr, index=False)
        te.to_csv(pte, index=False)
        print(f"\n=== {n}: {len(tr)} train / {len(te)} test samples, {df.shape[1] - 1} genes, "
              f"classes {sorted(df.label.unique())}", flush=True)
        clf = _make(a)
        clf.fit(ptr, a.label_col)
        m = clf.evaluate(pte, a.label_col)
        Xtr, ytr = load_table(ptr, a.label_col)
        Xte, yte = load_table(pte, a.label_col)
        lin = LinearHead(a.n_genes, a.seed).fit(Xtr, ytr.to_numpy())
        base = float((np.array(lin.classes_)[lin.predict_proba(Xte).argmax(1)] == yte.to_numpy()).mean())
        P = clf.predict_proba((Xte, None))
        cls = list(clf.classes_)
        pred = pd.DataFrame(P, index=Xte.index, columns=[f"p_{c}" for c in cls])
        pred.insert(0, "prediction", np.array(cls)[P.argmax(1)])
        pred.insert(0, "true", yte.values)
        pred.to_csv(os.path.join(out, "predictions.csv"))
        cm = pd.DataFrame(m["confusion_matrix"], index=[f"true_{c}" for c in cls], columns=[f"pred_{c}" for c in cls])
        print(f"TEST accuracy {m['accuracy']:.3f} | macro-F1 {m['f1_macro']:.3f} | "
              f"macro-precision {m['precision_macro']:.3f} | macro-recall {m['recall_macro']:.3f}")
        print(f"(tuned logistic-regression baseline on the same split: accuracy {base:.3f})")
        print("per-class accuracy:", {k: round(float(v), 3) for k, v in m["per_class_accuracy"].items()})
        print(cm.to_string())
        clf.save(os.path.join(out, "model"))
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}, f, indent=2, default=float)
        rows.append(dict(dataset=n, train=len(tr), test=len(te), genes=df.shape[1] - 1,
                         accuracy=round(m["accuracy"], 3), macro_f1=round(m["f1_macro"], 3),
                         logreg_baseline_acc=round(base, 3), minutes=round((time.time() - t0) / 60, 1)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\n==================== SUMMARY (one train/test split per dataset) ====================")
    print(summ.to_string(index=False))
    print(f"\nSaved models, splits, predictions and metrics under: {a.out}")


def _first_net(clf):
    n = clf.network if hasattr(clf, "network") else clf
    return n.members[0] if isinstance(n, EnsembleClassifier) else n


def _coverage_check(clf, X, force=False):
    """Refuse (unless --force) to predict when the file does not contain the genes the model was trained on."""
    genes = _first_net(clf).pre_.input_genes_
    cov = len(set(genes) & set(map(str, X.columns))) / len(genes)
    print(f"gene coverage: {cov:.1%} of the {len(genes)} genes this model was trained on are present in your file")
    if cov < 0.5 and not force:
        raise SystemExit("ERROR: your file does not contain the genes/columns this model was trained on, so the "
                         "predictions would be meaningless. The model only works on data with the SAME gene columns "
                         "(see required_genes.txt in the model folder) and the same normalisation. "
                         "Use --force only if you know the columns are equivalent.")


def _train_final(a):
    """Train on 100 % of a bundled real dataset and save a ready-to-share model folder."""
    from . import datasets as ds
    from .data import load_table
    names = BUNDLED if a.dataset == "all" else [a.dataset]
    for n in names:
        df = ds.load(n)
        X, y = load_table(df, a.label_col)
        path = os.path.join(a.out, n)
        os.makedirs(path, exist_ok=True)
        print(f"\n=== training final model on ALL {len(X)} samples of {n} ({X.shape[1]} genes, classes {sorted(set(y))})", flush=True)
        clf = _make(a)
        clf.fit((X, y.values))
        clf.save(path)
        with open(os.path.join(path, "required_genes.txt"), "w") as f:
            f.write("\n".join(_first_net(clf).pre_.input_genes_))
        with open(os.path.join(path, "train_info.json"), "w") as f:
            json.dump(dict(dataset=n, n_samples=len(X), n_genes_in_file=X.shape[1], classes=sorted(set(y)),
                           class_counts=y.value_counts().to_dict(), preset=a.preset, epochs=a.epochs,
                           n_genes_used=a.n_genes, seed=a.seed, task=ds.INFO[n][2], citation=ds.INFO[n][3]), f, indent=2)
        print(f"saved -> {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hilbertfuse")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on --train, optionally evaluate on --test")
    t.add_argument("--train", required=True); t.add_argument("--val"); t.add_argument("--test")
    t.add_argument("--out", default="runs/model"); _add_model_args(t)

    e = sub.add_parser("evaluate"); e.add_argument("--model", required=True); e.add_argument("--test", required=True)
    e.add_argument("--label-col", default="label"); e.add_argument("--device")
    e.add_argument("--transpose", action="store_true"); e.add_argument("--force", action="store_true")

    r = sub.add_parser("predict"); r.add_argument("--model", required=True); r.add_argument("--input", required=True)
    r.add_argument("--out", default="predictions.csv"); r.add_argument("--device")
    r.add_argument("--force", action="store_true", help="predict even if gene columns do not match (unreliable)")
    r.add_argument("--min-confidence", type=float, default=0.70,
                   help="predictions with a lower top probability are flagged UNCERTAIN (default 0.70)")

    x = sub.add_parser("explain", help="Guided Grad-CAM gene ranking")
    x.add_argument("--model", required=True); x.add_argument("--data", required=True)
    x.add_argument("--label-col", default="label"); x.add_argument("--top-k", type=int, default=40)
    x.add_argument("--out", default="explain"); x.add_argument("--device")

    c = sub.add_parser("cv", help="leak-free k-fold cross-validation")
    c.add_argument("--data", required=True); c.add_argument("--folds", type=int, default=5); _add_model_args(c)

    b = sub.add_parser("benchmark", help="classical vs paper-style vs HilbertFuseNet on real datasets")
    b.add_argument("--datasets", default="khan,alon,golub,shipp,su,singh,gordon",
                   help="comma list; see `hilbertfuse datasets`")
    b.add_argument("--methods", default="lr_mi,svm_mi,lr_all,paper,hfnet+fuse")
    b.add_argument("--folds", type=int, default=5); b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--n-genes", type=int, default=1024); b.add_argument("--epochs", type=int, default=50)
    b.add_argument("--out", default="bench"); b.add_argument("--device"); b.add_argument("--data-dir")

    tf = sub.add_parser("train-final", help="train on 100%% of a bundled real dataset (or 'all') and save a shareable model")
    tf.add_argument("--dataset", default="khan"); tf.add_argument("--out", default="models"); tf.add_argument("--quiet", action="store_true")
    _add_model_args(tf)

    rd = sub.add_parser("real-demo", help="train + test on a bundled REAL cancer dataset (or 'all') in one command")
    rd.add_argument("--dataset", default="khan", help="alon|golub|gordon|khan|pomeroy|shipp|singh|su|all")
    rd.add_argument("--test-size", type=float, default=0.3)
    rd.add_argument("--out", default="runs/real_demo"); rd.add_argument("--quiet", action="store_true")
    _add_model_args(rd)

    hm = sub.add_parser("heatmap", help="heat maps: Guided Grad-CAM gene image, top-gene expression, class x gene importance")
    hm.add_argument("--model", required=True); hm.add_argument("--data", required=True)
    hm.add_argument("--label-col", default="label", help="if absent, predicted classes are used")
    hm.add_argument("--top-k", type=int, default=40); hm.add_argument("--out", default="heatmaps")
    hm.add_argument("--sample", type=int, help="also draw the maps for this one sample (row number, 0-based)")
    hm.add_argument("--max-per-class", type=int, default=100); hm.add_argument("--device")
    hm.add_argument("--transpose", action="store_true"); hm.add_argument("--force", action="store_true")

    sub.add_parser("scores", help="show the stored cross-validated scores of all trained models")
    sub.add_parser("menu", help="interactive menu (list models, predict, evaluate, train, explain)")

    d0 = sub.add_parser("datasets", help="list / download real benchmark datasets")
    d0.add_argument("--fetch", help="name or 'all': download into --out")
    d0.add_argument("--out", default="datasets")

    d = sub.add_parser("demo-data", help="write synthetic train/test CSVs")
    d.add_argument("--out", default="demo_data"); d.add_argument("--classes", type=int, default=5)

    a = ap.parse_args(argv)

    if a.cmd == "train":
        clf = _make(a).fit(a.train, a.label_col, val=a.val)
        clf.save(a.out); print(f"saved model -> {a.out}")
        if a.test:
            m = clf.evaluate(a.test, a.label_col); _show(m)
            with open(os.path.join(a.out, "test_metrics.json"), "w") as f:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}, f, indent=2, default=float)
    elif a.cmd == "evaluate":
        from .data import load_table
        clf = PanCancerClassifier.load(a.model, a.device)
        Xc, yc = load_table(a.test, a.label_col, transpose=a.transpose)
        _coverage_check(clf, Xc, a.force)
        m = clf.evaluate((Xc, yc.values))
        absent = [c for c in m["classes"] if c not in set(yc.values)]
        if absent:
            print(f"NOTE: your file contains no samples of class(es) {absent}; scores only cover the classes present.")
        print(f"\nACCURACY {m['accuracy']:.3f} | macro-F1 {m['f1_macro']:.3f} on {len(Xc)} samples")
        print("per-class accuracy:", {k: round(float(v), 3) for k, v in m["per_class_accuracy"].items()})
        cls = list(m["classes"])
        print(pd.DataFrame(m["confusion_matrix"], index=[f"true_{c}" for c in cls], columns=[f"pred_{c}" for c in cls]).to_string())
    elif a.cmd == "predict":
        from .data import load_table
        clf = PanCancerClassifier.load(a.model, a.device)
        first = clf.members[0] if isinstance(clf, EnsembleClassifier) else clf
        X, _ = load_table(a.input, label_col=None, transpose=getattr(first, "transpose", False))
        _coverage_check(clf, X, a.force)
        P = clf.predict_proba((X, None))
        cls = np.array(clf.classes_)
        order = np.argsort(-P, axis=1)
        rows = np.arange(len(P))
        conf = P[rows, order[:, 0]]
        runner = cls[order[:, 1]] if P.shape[1] > 1 else np.array([""] * len(P))
        runner_p = P[rows, order[:, 1]] if P.shape[1] > 1 else np.zeros(len(P))
        out = pd.DataFrame(P, index=X.index, columns=[f"p_{c}" for c in clf.classes_])
        out.insert(0, "uncertain", conf < a.min_confidence)
        out.insert(0, "runner_up_prob", runner_p.round(3))
        out.insert(0, "runner_up", runner)
        out.insert(0, "confidence", conf.round(3))
        out.insert(0, "prediction", cls[order[:, 0]])
        show = out[["prediction", "confidence", "runner_up", "runner_up_prob", "uncertain"]].head(25)
        print("\nPREDICTIONS (confidence = the model's probability for its answer):")
        print(show.to_string())
        if len(out) > 25:
            print(f"... {len(out) - 25} more rows in the output file")
        print("\nSummary:", {k: int(v) for k, v in out.prediction.value_counts().items()}, f"| uncertain (confidence < {a.min_confidence}): {int(out.uncertain.sum())} of {len(out)}")
        print("Low confidence means: do not trust that answer. High confidence is NOT proof of correctness.")
        out.to_csv(a.out); print(f"wrote {a.out}")
    elif a.cmd == "explain":
        clf = PanCancerClassifier.load(a.model, a.device)
        res = clf.explain(a.data, a.label_col, top_k=a.top_k)
        os.makedirs(a.out, exist_ok=True)
        res["per_class"].to_csv(os.path.join(a.out, "genes_per_class.csv"), index=False)
        res["global"].to_csv(os.path.join(a.out, "genes_global.csv"), index=False)
        print(res["global"].to_string(index=False))
    elif a.cmd == "cv":
        print(cross_validate(a.data, a.label_col, a.folds, a.transpose, preset_name=a.preset,
                             n_genes=a.n_genes, epochs=a.epochs, batch_size=a.batch_size,
                             device=a.device, seed=a.seed).round(4))
    elif a.cmd == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(a.datasets.split(","), a.methods.split(","), a.folds, a.repeats, a.n_genes,
                      a.epochs, out=a.out, device=a.device, data_dir=a.data_dir)
        print(open(os.path.join(a.out, "summary.md"), encoding="utf-8").read())
    elif a.cmd == "heatmap":
        from .data import load_table
        from .heatmap import make_heatmaps
        clf = PanCancerClassifier.load(a.model, a.device)
        Xh, yh = load_table(a.data, a.label_col, transpose=a.transpose)
        _coverage_check(clf, Xh, a.force)
        files = make_heatmaps(clf, Xh, None if yh is None else yh.values, a.out, a.top_k, a.sample, a.max_per_class)
        print(f"\nHeat maps written to {a.out}:")
        for f in files:
            print("  ", f)
    elif a.cmd == "scores":
        from .menu import show_scores
        show_scores()
    elif a.cmd == "menu":
        from .menu import run_menu
        run_menu()
    elif a.cmd == "train-final":
        _train_final(a)
    elif a.cmd == "real-demo":
        _real_demo(a)
    elif a.cmd == "datasets":
        from . import datasets as ds
        if a.fetch:
            for n in (ds.INFO if a.fetch == "all" else [a.fetch]):
                ds.fetch(n, a.out); print("downloaded", n)
        else:
            print(ds.list_datasets().to_string(index=False))
    elif a.cmd == "demo-data":
        os.makedirs(a.out, exist_ok=True)
        df = make_synthetic(n_classes=a.classes, seed=0)
        df.iloc[:480].to_csv(os.path.join(a.out, "train.csv"), index=False)
        df.iloc[480:].to_csv(os.path.join(a.out, "test.csv"), index=False)
        print(f"wrote {a.out}/train.csv and test.csv")


if __name__ == "__main__":
    main()
