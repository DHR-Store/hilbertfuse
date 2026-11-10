import numpy as np
import pandas as pd
import torch
from hilbertfuse import (HilbertFuseNet, EnsembleClassifier, GeneImagePreprocessor, PanCancerClassifier,
                            make_synthetic)
from hilbertfuse.data import load_table, order_genes
from hilbertfuse.datasets import load as load_ds
from hilbertfuse.preprocess import hilbert_d2xy
from hilbertfuse.train import TrainConfig


def _data(n=240, g=400, c=3):
    df = make_synthetic(n_samples=n, n_genes=g, n_classes=c, n_informative=30, seed=1)
    k = int(n * 0.75)
    return df.iloc[:k], df.iloc[k:]


def _cfg(**kw):
    return TrainConfig(epochs=3, batch_size=32, verbose=False, amp=False, **kw)


def test_hilbert_is_continuous():
    pts = [hilbert_d2xy(16, i) for i in range(256)]
    assert len(set(pts)) == 256
    assert max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:])) == 1


def test_preprocessor_variants_and_no_leak():
    tr, te = _data()
    X, y = load_table(tr)
    yi = pd.factorize(y)[0]
    pre = GeneImagePreprocessor(n_genes=100, mi_prefilter=200).fit(X, yi)
    img = pre.transform(X)
    assert img.shape == (len(X), 1, 10, 10) and 0 <= img.min() and img.max() <= 1
    pre2 = GeneImagePreprocessor(n_genes=100, mi_prefilter=200, selection="mi_rd", layout="hilbert",
                                 channels="multi").fit(X, yi)
    im2 = pre2.transform(load_table(te)[0])
    assert im2.shape[1:] == (3, 16, 16) and len(set(pre2.gene_names_)) == 100
    assert np.mean([int(g[1:]) < 30 for g in pre2.mi_scores_.index[:20]]) > 0.7


def test_redundancy_pruning_drops_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 20))
    y = (base[:, 0] > 0).astype(int)
    dup = base[:, :1] + 0.01 * rng.normal(size=(120, 10))          # 10 near-copies of gene 0
    X = pd.DataFrame(np.hstack([base, dup]), columns=[f"g{i}" for i in range(30)])
    kept = GeneImagePreprocessor(n_genes=5, selection="mi_rd", mi_prefilter=None).fit(X, y).gene_names_
    assert sum(g in ["g0"] + [f"g{i}" for i in range(20, 30)] for g in kept) <= 2


def test_gene_order():
    ann = pd.DataFrame({"gene": ["A", "B", "C"], "chromosome": ["2", "1", "X"], "start": [5, 9, 1]})
    assert order_genes(["C", "A", "B", "Z"], ann) == ["B", "A", "C", "Z"]


def test_hilbertfusenet_shapes():
    for kw, size in [(dict(variant="compact"), 32), (dict(variant="pretrained121", input_size=64), 32)]:
        m = HilbertFuseNet(4, in_channels=3, pretrained=False, **kw).eval()
        assert m(torch.rand(2, 3, size, size)).shape == (2, 4)
    # SE gates start as identity
    from hilbertfuse.hilbertfusenet import SE
    x = torch.rand(2, 8, 4, 4)
    assert torch.allclose(SE(8)(x), x)


def test_train_predict_explain_save_load(tmp_path):
    tr, te = _data()
    clf = PanCancerClassifier(n_genes=100, mi_prefilter=200, device="cpu",
                              train_config=TrainConfig(epochs=30, batch_size=32, verbose=False, amp=False, mixup_alpha=0.3))
    clf.fit(tr)
    assert clf.evaluate(te)["accuracy"] > 0.5
    res = clf.explain(te, top_k=10, max_per_class=10)
    assert len(res["global"]) == 10 and {"class", "gene", "score"} <= set(res["per_class"].columns)
    clf.save(tmp_path / "m")
    clf2 = PanCancerClassifier.load(str(tmp_path / "m"), device="cpu")
    np.testing.assert_allclose(clf.predict_proba(te), clf2.predict_proba(te), atol=1e-5)


def test_ensemble_and_torchvision_backbone(tmp_path):
    tr, te = _data(180)
    ens = EnsembleClassifier(n_members=2, gene_scales=(0.5, 1.0), n_genes=100, mi_prefilter=200,
                             train_config=_cfg(), device="cpu").fit(tr)
    assert ens.predict_proba(te).shape == (len(te), 3)
    assert len(ens.explain(te, top_k=5, max_per_class=5)["global"]) == 5
    ens.save(tmp_path / "e")
    again = PanCancerClassifier.load(str(tmp_path / "e"), device="cpu")
    np.testing.assert_allclose(ens.predict_proba(te), again.predict_proba(te), atol=1e-5)
    r = PanCancerClassifier(backbone="resnet18", pretrained=False, n_genes=64, mi_prefilter=200,
                            input_size=32, train_config=_cfg(), device="cpu").fit(tr)
    assert r.predict(te).shape[0] == len(te)


def test_bundled_real_dataset():
    df = load_ds("khan")
    assert df.shape == (63, 2309) and set(df.label) == {"EWS", "RMS", "NB", "BL"}


def test_fuse_preset_roundtrip(tmp_path):
    from hilbertfuse import preset
    tr, te = _data(160)
    clf = preset("fuse", n_genes=64, epochs=3, batch_size=32, device="cpu").fit(tr)
    m = clf.evaluate(te)
    assert m["accuracy"] > 0.5 and "f1_macro" in m
    clf.save(tmp_path / "f")
    again = PanCancerClassifier.load(str(tmp_path / "f"), device="cpu")
    np.testing.assert_allclose(clf.predict_proba(te), again.predict_proba(te), atol=1e-5)
    assert len(again.explain(te, top_k=5, max_per_class=5)["global"]) == 5


def test_menu_lists_models_and_exits(monkeypatch, capsys):
    from hilbertfuse import menu
    answers = iter(["1", "", "0"])                       # list models, Enter, exit
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    menu.run_menu()
    out = capsys.readouterr().out
    assert "HilbertFuse" in out and "Bye." in out


def test_linear_quick_mode_and_explain(tmp_path):
    from hilbertfuse import preset
    tr, te = _data(200)
    clf = preset("linear", n_genes=100).fit(tr)
    assert clf.evaluate(te)["accuracy"] > 0.8
    ex = clf.explain(te, top_k=5)
    assert len(ex["global"]) == 5 and set(ex["per_class"]["class"]) == set(clf.classes_)
    clf.save(tmp_path / "lin")
    again = PanCancerClassifier.load(str(tmp_path / "lin"))
    np.testing.assert_allclose(clf.predict_proba(te), again.predict_proba(te), atol=1e-9)


def test_cli_predict_flags_uncertain(tmp_path, capsys):
    from hilbertfuse import preset
    from hilbertfuse.cli import main
    tr, te = _data(200)
    preset("linear", n_genes=100).fit(tr).save(tmp_path / "m")
    te.to_csv(tmp_path / "new.csv", index=False)
    main(["predict", "--model", str(tmp_path / "m"), "--input", str(tmp_path / "new.csv"),
          "--out", str(tmp_path / "p.csv"), "--min-confidence", "0.99"])
    out = pd.read_csv(tmp_path / "p.csv", index_col=0)
    assert {"prediction", "confidence", "runner_up", "uncertain"} <= set(out.columns)
    assert "PREDICTIONS" in capsys.readouterr().out


def test_heatmaps_for_cnn_and_linear_models(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from hilbertfuse import preset
    from hilbertfuse.heatmap import make_heatmaps
    tr, te = _data(160)
    Xte, yte = load_table(te)
    fuse = preset("fuse", n_genes=64, epochs=3, batch_size=32, device="cpu").fit(tr)
    f1 = make_heatmaps(fuse, Xte, yte.values, str(tmp_path / "cnn"), top_k=10, sample=0, max_per_class=8)
    assert {"heatmap_top_genes_expression.png", "heatmap_class_gene_scores.png",
            "heatmap_gene_image_all_classes.png", "heatmap_sample_0.png"} <= set(f1)
    f2 = make_heatmaps(fuse, Xte, None, str(tmp_path / "cnn_unlabeled"), top_k=10, max_per_class=8)   # predicted classes
    assert "heatmap_gene_image_all_classes.png" in f2
    lin = preset("linear", n_genes=64).fit(tr)
    f3 = make_heatmaps(lin, Xte, yte.values, str(tmp_path / "lin"), top_k=10)
    assert "heatmap_top_genes_expression.png" in f3 and "heatmap_gene_image_all_classes.png" not in f3
    for f in f1 + f3:
        assert (tmp_path / ("cnn" if f in f1 else "lin") / f).exists()
