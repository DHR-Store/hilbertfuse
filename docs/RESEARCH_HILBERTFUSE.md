# Research problem, solution and measured results: HilbertFuseNet / HilbertFuse

Based on Tables 4, 5 and 6 of *Wang et al. 2021 (MI_DenseNetCAM)* — preprocessing strategies, classifiers and image dimensions — plus the rest of the paper.

## 1. What the three tables say, and what they hide

| Table | Reported | What it hides |
|---|---|---|
| **4** preprocessing | Var 94.46 · Chi² 95.42 · F-test 95.03 · **MI 96.81** | All four are *univariate filters* fit on the **whole data set** before cross-validation (selection has seen the test labels). Ranking genes one by one ignores redundancy, so 3,600 slots are partly filled with near-copies of the same signal. One run, no variance. |
| **5** classifiers | KNN 92.61 · SVM 91.53 · CNN 94.30 · **DenseNet 96.81** | The paper's own related work quotes a linear SVM at **95.8 %** (Hsu & Si 2018) — 4 points above the SVM here — so the classical baselines look under-tuned. "Deep beats classical" is not established without tuned baselines and repeated CV. |
| **6** image size | 30²: 93.60 · 50²: 95.03 · **60²: 96.81** · 70²: 95.22 · 90²: 94.17 · 110²: 92.93 · 130²: 93.41 | Image side² *is* the number of genes (900 … 16,900), so "dimension" confounds gene count with image size. Accuracy swings **3.9 points**, peaks sharply at 60 and falls by ~1.6 on either side: the model is **size/layout-fragile**. The best size was also picked on the same CV that reports it (selection bias). |

## 2. The research problem

> **Image-embedding of omics data is layout- and size-fragile, selects redundant genes, is evaluated with leaky, single-run, weakly-baselined protocols — so the headline gain of "DenseNet on gene images" is not reliable.**

Concretely:
1. **Arbitrary topology.** Row-major fill puts a gene's right neighbour 1 position away but its lower neighbour 60 positions away; convolutions assume local 2-D structure that does not exist.
2. **Redundant univariate selection** (Table 4).
3. **Fragility to image size** (Table 6).
4. **Weak evidence**: leakage, one run, untuned baselines (Tables 4–5).

## 3. Solution: HilbertFuseNet + a leak-free protocol

| Problem | Component | Idea |
|---|---|---|
| 1 topology | **Cluster + Hilbert layout** | Order selected genes by hierarchical clustering of co-expression, then place them along a Hilbert curve (consecutive genes are always adjacent pixels; verified in tests). Co-expressed genes become spatial neighbours, so 3×3 convolutions model gene modules. |
| 2 redundancy | **MI + redundancy pruning** (`selection="mi_rd"`) | Keep genes in MI order but skip any with \|r\| > 0.9 to an already kept gene (mRMR-like). |
| 2/3 information | **3-channel images** | channel 0 expression · channel 1 MI prior (class relevance of the gene, same for all samples) · channel 2 per-sample rank (scale/batch robust). |
| 3 fragility | **SE gates + multi-scale head** | Zero-initialised Squeeze-Excitation after each dense block (starts as identity) and a head that concatenates pooled features from blocks 2, 3 and 4. |
| 3 fragility | **Multi-view ensemble** (`hfnet_ens`) | Members differ in seed and gene count (0.75×, 1×, 1.25×) and are averaged — the average does not depend on the single "lucky" image size. |
| small n | **Mixup, gene-dropout, label smoothing, class-weighted loss** | Regularisers for 60–200 samples. |
| 4 evidence | **Leak-free repeated CV + tuned classical baselines + paired Wilcoxon test + ablations** | `hilbertfuse benchmark`. The CNN plus a tuned linear model is the **fused model `fuse` = HilbertFuse** (see §5). |

### HilbertFuseNet architecture
```
 input (B, 3, S, S)  -> BatchNorm (learnable input norm)
 stem: Conv3x3(stride 1) - BN - ReLU
 [DenseBlock1 -> SE] -> Transition -> [DenseBlock2 -> SE] -> Transition ->
 [DenseBlock3 -> SE] -> Transition -> [DenseBlock4 -> SE] -> BN - ReLU
 head: concat( GAP(final), GAP(BN-ReLU(block3)), GAP(BN-ReLU(block2)) ) -> Dropout -> Linear(C)
 compact default: growth 12, blocks (4,6,8,6), 24 stem features  (~0.4 M parameters)
 variant "hilbertfusenet121": same SE + multi-scale head on a pretrained torchvision DenseNet-121
```
Backbones: `hilbertfusenet` (from scratch, compact), `hilbertfusenet121` (ImageNet-pretrained DenseNet-121 + new parts), `densenet_paper` (same compact net *without* SE/multi-scale/dropout — the control), plus torchvision DenseNet/ResNet.

## 4. Datasets (multiple real cancer data sets)

Public microarray collection (`datamicroarray`; original papers cited). Eight are bundled in `datasets/`, three more download on demand: `hilbertfuse datasets` lists them, `hilbertfuse datasets --fetch yeoh --out datasets` downloads one.

| name | samples | genes | task |
|---|---|---|---|
| alon | 62 | 2,000 | colon tumor vs normal |
| golub | 72 | 7,129 | ALL vs AML |
| khan | 63 | 2,308 | 4 small round blue cell tumors |
| shipp | 77 | 7,129 | DLBCL vs follicular lymphoma |
| singh | 102 | 12,600 | prostate tumor vs normal |
| su | 102 | 5,565 | 4 carcinoma tissue types |
| gordon | 181 | 12,533 | mesothelioma vs lung adenocarcinoma |
| pomeroy | 60 | 7,128 | CNS tumor outcome (hard) |
| chiaretti / burczynski / yeoh | 128 / 127 / 248 | 12,625 / 22,283 / 12,625 | ALL subtypes / non-cancer IBD / 6 ALL subtypes (download) |

**Important:** these are *small* cohorts (60–250 samples), not the 10,000-sample TCGA pan-cancer set of the paper. Conclusions about HilbertFuseNet on small data do not automatically transfer to TCGA; for that, download TCGA RNA-seq yourself (see README) and run the same `benchmark`/`cv` commands.

## 5. Measured results on 8 real cancer data sets (5-fold CV, leak-free)

### TABLE 2 | The experimental results of the methods (mean over 8 real data sets, 5-fold CV).

| Method | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 89.29% | 90.33% | 89.29% | 89.02% |
| HilbertFuseNet (CNN only) | 81.48% | 85.79% | 81.48% | 81.17% |
| Paper-style DenseNet | 81.14% | 82.61% | 81.14% | 80.51% |
| MI_LogReg | 90.19% | 90.76% | 90.19% | 89.86% |
| LogReg (all genes) | 90.30% | 90.22% | 90.30% | 89.53% |
| MI_SVM | 88.09% | 90.41% | 88.09% | 88.26% |
| MI_KNN | 86.72% | 88.76% | 86.72% | 86.61% |

*Each data set counts equally; precision/recall/F1 are class-weighted as in the paper.*

### TABLE 3 | Accuracy per real data set (5-fold CV).

| Data set | Task | Samples | HilbertFuse (CNN + linear) | HilbertFuseNet (CNN only) | Paper-style DenseNet | MI_LogReg | LogReg (all genes) | MI_SVM | MI_KNN |
|---|---|---|---|---|---|---|---|---|---|
| alon | colon tumor vs normal | 62 | 79.36% | 72.95% | 71.15% | 82.69% | 76.28% | 80.90% | 79.23% |
| golub | ALL vs AML leukemia | 72 | 94.48% | 87.33% | 86.10% | 98.67% | 98.67% | 95.90% | 97.24% |
| gordon | lung: mesothelioma vs adenocarcinoma | 181 | 100.00% | 69.26% | 98.89% | 99.44% | 99.44% | 99.44% | 98.89% |
| khan | 4 small round blue cell tumors | 63 | 100.00% | 100.00% | 70.26% | 97.71% | 98.90% | 97.80% | 92.22% |
| pomeroy | brain tumor survival (hard) | 60 | 60.00% | 53.33% | 50.00% | 61.67% | 65.00% | 58.33% | 48.33% |
| shipp | DLBCL vs follicular lymphoma | 77 | 92.00% | 85.33% | 89.33% | 92.00% | 94.75% | 83.00% | 89.50% |
| singh | prostate tumor vs normal | 102 | 93.24% | 92.29% | 88.29% | 91.24% | 91.29% | 91.24% | 90.24% |
| su | 4 carcinoma tissue types | 102 | 95.24% | 91.38% | 95.14% | 98.10% | 98.10% | 98.10% | 98.10% |

### TABLE 4 | The performance of different gene-selection strategies (same logistic-regression classifier, 1024 genes).

| Method | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Var_LogReg | 90.86% | 90.03% | 90.86% | 89.73% |
| Chi2_LogReg | 90.65% | 91.34% | 90.65% | 90.20% |
| FTest_LogReg | 90.89% | 91.51% | 90.89% | 90.44% |
| MI_LogReg | 90.19% | 90.76% | 90.19% | 89.86% |
| MI+RD_LogReg | 90.26% | 90.80% | 90.26% | 89.93% |

*Selection fitted on training folds only.*

### TABLE 5 | The performance of different classifiers.

| Classifiers | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| MI_KNN | 86.72% | 88.76% | 86.72% | 86.61% |
| MI_SVM | 88.09% | 90.41% | 88.09% | 88.26% |
| MI_LogReg | 90.19% | 90.76% | 90.19% | 89.86% |
| MI_DenseNet (paper-style) | 81.14% | 82.61% | 81.14% | 80.51% |
| MI+RD_HilbertFuseNet | 81.48% | 85.79% | 81.48% | 81.17% |
| HilbertFuse | 89.29% | 90.33% | 89.29% | 89.02% |

### TABLE 6 | The performance of different image dimensions / gene counts (MI_LogReg).

| Dimensions | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 10 x 10  (100 genes) | 89.54% | 89.77% | 89.54% | 88.86% |
| 16 x 16  (256 genes) | 91.03% | 91.32% | 91.03% | 90.39% |
| 24 x 24  (576 genes) | 90.53% | 91.67% | 90.53% | 90.29% |
| 32 x 32  (1024 genes) | 90.19% | 90.76% | 90.19% | 89.86% |
| 45 x 45  (2048 genes) | 90.95% | 91.53% | 90.95% | 90.46% |
| 64 x 64  (4096 genes) | 90.86% | 90.25% | 90.86% | 90.03% |

*Image side^2 = number of genes. Logistic regression does not use the image; this shows the effect of the number of genes only.*

Full tables incl. one table per pretrained model: `docs/PAPER_STYLE_TABLES.md` / `.html`. Raw per-fold numbers: `results/benchmark_results_v2.csv`.
Settings: 1,024 genes, 50 epochs, batch 16, random initialisation (ImageNet weights could not be downloaded in the build environment), seed 0, one CPU core.

## 6. What the numbers say (computed from the results file)

* **HilbertFuse vs the best tuned classical method** (all-gene logistic regression): mean accuracy 89.3% vs 90.3%: -1.0 points, not significantly different (paired Wilcoxon p=0.49), better in 9 / worse in 9 of 40 folds.
* **HilbertFuse vs the paper-style DenseNet:** 89.3% vs 81.1%: +8.1 points, significantly better (p<0.001), better in 23 / worse in 2 of 40 folds.
* **CNN alone (HilbertFuseNet) vs MI logistic regression:** 81.5% vs 90.2%: -8.6 points, significantly worse (p=0.001), better in 6 / worse in 21 of 40 folds.
* **CNN alone vs the paper-style DenseNet (do the new design ideas help the CNN?):** 81.5% vs 81.1%: +0.3 points, not significantly different (paired Wilcoxon p=0.77), better in 18 / worse in 14 of 40 folds.
* The CNN alone beats MI logistic regression on khan, singh (of 8).
* The CNN alone is the least stable method: fold-to-fold standard deviation up to 23.3 points (on gordon).
* Models judged **not useful** (no better than guessing the most common class): `pomeroy` (CV accuracy 0.600 vs 0.650 for always guessing the common class).
* Number of genes (logistic regression, 100 to 4096 genes): accuracy varies only between 89.5% and 91.0%, far less than the 3.9-point swing the paper reports across image sizes with a CNN.
* Caveat: small cohorts (60-181 samples), one random seed, one platform per data set, networks trained from random weights; results do not automatically transfer to large cohorts such as TCGA.

## 7. Limitations

* Small cohorts, 5 folds, one seed, 1,024 genes, 50 epochs, random initialisation; no external validation cohort.
* Microarray data, not the paper's TCGA RNA-seq; to test TCGA, run `hilbertfuse benchmark` / `cv` on it yourself.
* Gene rankings are model attributions, not validated biomarkers.
* Research use only; not a diagnostic tool.
