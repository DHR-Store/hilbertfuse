"""Guided Grad-CAM (Selvaraju et al. 2017) re-implemented with plain PyTorch hooks,
and conversion of pixel importance back to gene names.

Guided Grad-CAM = Grad-CAM (class-discriminative, coarse) x Guided Backprop (pixel-sharp).
The paper averaged heat maps over all samples and read genes from the brightest pixels;
here we do it PER CLASS (true-class target, correctly-classified samples) and also give a
global ranking and a class-specificity score.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _guided_relu(model: nn.Module):
    """Guided backprop: clamp the gradient arriving at every ReLU output to >= 0.
    (ReLU's own backward then applies the forward mask -> exactly guided backprop.)"""
    handles = []

    def fwd(_, __, out):
        if out.requires_grad:
            out.register_hook(lambda g: g.clamp(min=0))
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(fwd))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


class GuidedGradCAM:
    def __init__(self, model: nn.Module, device):
        self.model, self.device = model.eval().to(device), device

    def _cam(self, x, target):
        store = {}

        def fwd(_, __, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))
        h = self.model.target_layer.register_forward_hook(fwd)
        try:
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()
        finally:
            h.remove()
        w = store["g"].mean(dim=(2, 3), keepdim=True)            # GAP of gradients
        cam = F.relu((w * store["a"]).sum(1, keepdim=True))      # weighted sum + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.detach()
        return cam / cam.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

    def _guided(self, x, target):
        x = x.clone().requires_grad_(True)
        with _guided_relu(self.model):
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()
        return x.grad.detach()

    def attribute(self, images: np.ndarray, targets: np.ndarray, batch_size: int = 32,
                  method: str = "guided_gradcam") -> np.ndarray:
        """Return (N, S, S) non-negative importance maps, each normalised to max 1."""
        assert method in ("guided_gradcam", "gradcam", "guided_backprop")
        outs = []
        for i in range(0, len(images), batch_size):
            xb = torch.from_numpy(images[i:i + batch_size]).to(self.device)
            tb = torch.from_numpy(np.asarray(targets[i:i + batch_size])).long().to(self.device)
            cam = self._cam(xb, tb) if method != "guided_backprop" else None
            gb = self._guided(xb, tb).abs().sum(1, keepdim=True) if method != "gradcam" else None
            m = cam if gb is None else (gb if cam is None else cam * gb)
            m = m / m.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            outs.append(m[:, 0].detach().cpu().numpy())
        return np.concatenate(outs)


def gene_importance(attr: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: Sequence[str], gene_names: Sequence[str], top_k: int = 40,
                    only_correct: bool = True, pixel_index: Optional[np.ndarray] = None) -> dict:
    """Aggregate per-sample maps into per-class and global gene rankings."""
    n_genes = len(gene_names)
    flat = attr.reshape(len(attr), -1)
    flat = flat[:, pixel_index] if pixel_index is not None else flat[:, :n_genes]  # drop padding pixels
    C = len(class_names)
    mat = np.full((C, n_genes), np.nan)
    for c in range(C):
        idx = np.where(y_true == c)[0]
        if only_correct:
            good = idx[y_pred[idx] == c]
            idx = good if len(good) >= 3 else idx
        if len(idx):
            mat[c] = flat[idx].mean(0)
    present = ~np.isnan(mat[:, 0])
    denom = np.nansum(mat, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0                                  # all-zero maps (untrained net) -> scores 0, not NaN
    norm = np.nan_to_num(mat / denom)                        # each class sums to 1
    rows = []
    for c in np.where(present)[0]:
        others = np.nanmean(norm[np.setdiff1d(np.where(present)[0], [c])], axis=0) if present.sum() > 1 else 0 * norm[c]
        spec = norm[c] - others
        for r, g in enumerate(np.argsort(-norm[c])[:top_k], 1):
            rows.append((class_names[c], r, gene_names[g], norm[c, g], spec[g]))
    per_class = pd.DataFrame(rows, columns=["class", "rank", "gene", "score", "specificity"])
    gscore = np.nanmean(norm, axis=0)
    gi = np.argsort(-gscore)[:top_k]
    global_df = pd.DataFrame({"rank": range(1, len(gi) + 1), "gene": [gene_names[i] for i in gi],
                              "score": gscore[gi]})
    return {"per_class": per_class, "global": global_df}
