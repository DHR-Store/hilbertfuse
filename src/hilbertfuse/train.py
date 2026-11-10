"""Training loop with fine-tuning schedule, class balancing, augmentation, early stopping."""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: Optional[float] = None     # None -> 3e-4 pretrained fine-tuning / 1e-3 from scratch
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    class_balance: str = "loss"    # "loss" | "sampler" | "none"
    val_fraction: Optional[float] = None  # None -> 0.1 if n>=200 else 0 (tiny sets: fixed schedule)
    mixup_alpha: float = 0.0       # >0: mixup between samples (regulariser for small n)
    patience: int = 8
    gene_dropout: float = 0.05     # augmentation: randomly zero pixels (genes)
    noise_std: float = 0.01        # augmentation: gaussian noise on [0,1] pixels
    warmup_epochs: int = 1         # train head only first (backbone frozen)
    amp: bool = True               # mixed precision on CUDA
    num_workers: int = 0
    seed: int = 0
    verbose: bool = True

    def to_dict(self):
        return asdict(self)


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split(images, y, frac, seed):
    if frac <= 0:
        return images, y, None, None
    counts = np.bincount(y)
    strat = y if counts[counts > 0].min() >= 2 and int(len(y) * frac) >= len(counts) else None
    tr, va = train_test_split(np.arange(len(y)), test_size=frac, stratify=strat, random_state=seed)
    return images[tr], y[tr], images[va], y[va]


@torch.no_grad()
def predict_proba(model, images: np.ndarray, device, batch_size=128) -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(images), batch_size):
        xb = torch.from_numpy(images[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def metrics_from_preds(y_true, y_pred, n_classes) -> dict:
    labels = list(range(n_classes))
    present = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))  # macro over classes that occur
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = np.divide(cm.diagonal(), cm.sum(1), out=np.full(n_classes, np.nan), where=cm.sum(1) > 0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision_macro=p, recall_macro=r, f1_macro=f,
                precision_weighted=pw, recall_weighted=rw, f1_weighted=fw,
                per_class_accuracy=per_class, confusion_matrix=cm)


def fit_model(model, images, y, cfg: TrainConfig, device, val=None):
    """Train `model` in-place; returns history. `val`=(images,y) overrides the internal split."""
    set_seed(cfg.seed)
    n_classes = model.classifier.out_features
    if val is None:
        frac = cfg.val_fraction if cfg.val_fraction is not None else (0.1 if len(y) >= 200 else 0.0)
        Xtr, ytr, Xva, yva = _split(images, y, frac, cfg.seed)
    else:
        Xtr, ytr, (Xva, yva) = images, y, val
    model.to(device)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    loss_w = torch.tensor(w, dtype=torch.float32, device=device) if cfg.class_balance == "loss" else None
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=cfg.label_smoothing)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long())
    sampler = None
    if cfg.class_balance == "sampler":
        sampler = WeightedRandomSampler(torch.tensor(w[ytr], dtype=torch.double), len(ytr), replacement=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
                    num_workers=cfg.num_workers, drop_last=len(ds) > cfg.batch_size)

    lr = cfg.lr if cfg.lr is not None else (3e-4 if cfg.warmup_epochs > 0 else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        t0 = time.time()
        for p in backbone_params:
            p.requires_grad_(ep >= cfg.warmup_epochs)
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.gene_dropout > 0:
                xb = xb * (torch.rand_like(xb) > cfg.gene_dropout)
            if cfg.noise_std > 0:
                xb = (xb + cfg.noise_std * torch.randn_like(xb)).clamp(0, 1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if cfg.mixup_alpha > 0 and len(xb) > 1:
                    lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(len(xb), device=device)
                    out = model(lam * xb + (1 - lam) * xb[perm])
                    loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[perm])
                else:
                    loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        rec = dict(epoch=ep + 1, train_loss=tot / max(nb, 1), sec=time.time() - t0)
        if Xva is not None and len(yva):
            m = metrics_from_preds(yva, predict_proba(model, Xva, device).argmax(1), n_classes)
            rec.update(val_acc=m["accuracy"], val_f1=m["f1_macro"])
            score = m["f1_macro"]
        else:
            score = -rec["train_loss"]
        hist.append(rec)
        if cfg.verbose:
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if score > best + 1e-6:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if Xva is not None and bad >= cfg.patience:
                if cfg.verbose:
                    print(f"early stopping at epoch {ep + 1}")
                break
    for p in backbone_params:
        p.requires_grad_(True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
