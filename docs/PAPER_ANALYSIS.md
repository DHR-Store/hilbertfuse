# Analysis of the MI_DenseNetCAM paper (background)

> This file keeps the paper analysis (sections 1–5). For the current package, commands and measured results see `README.md`, `docs/COMMANDS.md` and `docs/RESEARCH_HILBERTFUSE.md`.

Analysis of *Wang et al. 2021, "MI_DenseNetCAM: A Novel Pan-Cancer Classification and Prediction Method Based on Mutual Information and Deep Learning Model", Front. Genet. 12:670232* (doi:10.3389/fgene.2021.670232), and the background for the `hilbertfuse` package.

> **Research use only.** Nothing here is a validated diagnostic tool.

---

## 1. What the paper does (in one paragraph)

TCGA RNA-seq expression (33 tumor types, ~20,531 genes per sample) is high-dimensional with few samples. The authors (1) Min-Max normalise each gene, (2) rank genes by **mutual information (MI)** with the tumor label and keep the top **N = 3600**, (3) write each sample's 3600 values row-by-row into a **60×60 grayscale image**, (4) classify the images with a **DenseNet** trained with Adam + cross-entropy + 10-fold CV, and (5) run **Guided Grad-CAM** to see which pixels (= genes) drive each decision, mapping bright pixels back to gene names to propose biomarkers. Reported: **96.81 % accuracy** over 33 classes (vs 95.59 % for the previous CNN baseline *Var-CNN*).

## 2. Pipeline, step by step

```
 expression table (samples x ~20k genes) + label
        │
 [1] Min-Max per gene ............ y = (x - min) / (max - min)                  (Eq. 1)
        │
 [2] Mutual information ranking .. I(X;Y) = Σ P(x,y) log P(x,y)/(P(x)P(y))      (Eq. 2)
        │   keep top N=3600 genes (N chosen empirically)
        │
 [3] Gene ordering ............... by chromosome position (neighbouring genes may interact)
        │
 [4] 1-D -> 2-D .................. 3600 values -> 60x60 image, row-major, scaled to [0,255]
        │
 [5] DenseNet classifier ......... 4 dense blocks (BN-ReLU-1x1conv-BN-ReLU-3x3conv),
        │                           transitions (BN-ReLU-1x1conv-AvgPool), GAP, Linear(33)
        │                           growth_rate=16, compression=0.5, Adam lr=1e-4,
        │                           200 epochs, batch 32, 10-fold CV
        │
 [6] Guided Grad-CAM ............. heat-map = Grad-CAM(activation maps) ⊙ GuidedBackprop(gradient maps)
        │   Grad-CAM: weights = mean gradient of class score per feature map;
        │             activation map = ReLU(Σ weights·feature maps)
        │   average heat-maps over samples; argmax pixel -> index -> gene name
        ▼
 top-40 genes -> DAVID/KEGG enrichment + GeneCards literature check
```

### Why each design choice
| Choice | Reason given | Practical effect |
|---|---|---|
| MI filter | removes noise/redundant genes cheaply (filter method) | +1.4–3.1 pts accuracy over no-preprocessing / variance / Chi² / F-test (Table 4) |
| Image embedding | lets a CNN be used on tabular omics | enables CAM-style explanation |
| DenseNet | feature reuse, fewer parameters, regularising dense connections for small-sample classes | beats KNN (92.6), SVM (91.5), plain CNN (94.3) in Table 5 |
| 60×60 | best of {30…130} in Table 6 | 96.81 % vs 92.9–95.2 % for others |
| Guided Grad-CAM | pixel-sharp *and* class-discriminative | gene shortlist; 83.2 % accuracy with only 20 genes (Fig. 3) |

## 3. Reported results (paper)

* Overall (10-fold CV): Acc 96.81 / Prec 96.89 / Rec 96.81 / F1 96.85 %.
* Baselines: Var-CNN 95.59, MI-KNN 92.61, ET-SVM 90.73, rL-GenSVM 87.29.
* Small classes: ACC 100 %, CHOL 75 %, MESO 99 %. Failure: **READ = 0 %** (all baselines also ≈0–35 %).
* Gene sets: with only 25 genes it reaches 86 % (ET-SVM needs 70, Var-CNN 85, MI-KNN 130).
* 40 genes analysed (GFAP→GBM/LGG, LGALS4→COAD/READ, SFTPA2→LUSC, MUC1→OV, PGC→STAD, S100A6→THCA, …); 31 KEGG pathways with P<0.01.
* Hardware: i7-10700, RTX 2080 Super, Python.

## 4. Critical review — weaknesses we found and how the package responds

| # | Issue in the paper | Why it matters | Package fix |
|---|---|---|---|
| 1 | **Data leakage.** Min-Max and MI selection use *all* samples, then 10-fold CV is run on the selected genes (Sec. 3.2, Fig. 1A). | Gene selection has seen the test labels → optimistic accuracy. | `GeneImagePreprocessor` is **fit on training data only**; `cross_validate()` refits it inside each fold. |
| 2 | READ ≈ 0 %. READ (rectum) and COAD (colon) are biologically near-identical in expression. | Overall accuracy hides complete failure on a class; the paper does not discuss it. | Reports **per-class accuracy, macro-F1 and confusion matrix**; `class_balance="loss"` by default; advice to merge COAD/READ (COADREAD) if appropriate. |
| 3 | Accuracy is the headline metric. | With imbalanced classes (36 – 1,093 samples) accuracy ≈ weighted recall. | Macro-averaged metrics reported; early stopping on macro-F1. |
| 4 | Spatial layout is arbitrary. Row-major fill puts a gene's horizontal neighbour 1 gene away but its vertical neighbour 60 genes away; ordering is applied *before* MI so selected genes are not contiguous anyway. | CNN locality assumptions are only weakly met; gains may come from the MI filter, not from "image-ness". | Genes are ordered **after** selection (`gene_annotation`); the claim is testable via ablation (`--no-pretrained`, `resnet18`, or shuffled order). |
| 5 | Gene-importance is read from the *argmax pixel of a heat-map averaged over all classes*. | Mixes class-specific signals; the 40-gene list is dominated by generally variable genes (collagens, keratins, MALAT1). | **Per-class** Guided Grad-CAM on correctly-classified samples, plus a global score and a **class-specificity** score. |
| 6 | Validation of genes is anecdotal (GeneCards look-ups; 11 of 40 genes "unclear"). Enrichment run on a handful of genes without a background set or multiple-testing correction. | Weak evidence for "biomarkers". | Output is labelled *candidates*; recommended validation steps in §10. |
| 7 | Reproducibility: parameter count (13.9 M) cannot be matched to a stock DenseNet (≈8 M for DenseNet-121) from the text; no code/seed released; Fig. 3 max accuracy (~93 % at 200 genes) differs from some text claims (86 % ceiling for rL-GenSVM vs ~88 % in the plot). | Hard to reproduce exactly. | Package is fully seeded, stores config + history; do not expect bit-identical numbers. |
| 8 | Training 200 epochs from scratch. | Slow, over-fit risk on small classes. | Fine-tune **pretrained** backbone, cosine LR, early stopping, augmentation. |

## 5. Enhancements implemented

1. **Leak-free preprocessing** (fit/transform API, train-only statistics; values outside the training range are clipped at inference).
2. **Pretrained, existing models** (torchvision, ImageNet weights): `densenet121/169/201/161`, `resnet18/34/50`. The 3-channel stem is collapsed to 1 channel by summing the pretrained filters; images are standardised and resized (default 96×96) *inside* the model so Guided Grad-CAM still returns gradients w.r.t. original gene pixels. Falls back to random init with a warning if weights can't be downloaded.
3. **Fine-tuning recipe**: head-only warm-up epoch, AdamW, cosine schedule, label smoothing, class-weighted loss (or weighted sampler), stratified internal validation split, early stopping on macro-F1, mixed precision on CUDA.
4. **Augmentation for tabular images**: random gene dropout + Gaussian noise.
5. **Speed**: optional ANOVA prefilter (default top 8,000) before kNN-MI so 20k genes finish in minutes.
6. **Genomic ordering from any annotation file** (`gene,chromosome[,start]`).
7. **Per-class Guided Grad-CAM** implemented with plain PyTorch hooks (no extra dependency), global ranking, class-specificity.
8. **Robust inference**: aligns columns by gene name, warns/imputes for missing genes, rejects unseen labels.
9. **Packaging**: `pip install`, CLI, Python API, saved artifact (`model.pt`, `preprocessor.joblib`, `meta.json`, `history.csv`, `mi_scores.csv`), tests.

