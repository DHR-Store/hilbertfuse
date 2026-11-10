# HilbertFuse — technology and algorithm details (slide-by-slide material for a PPT)

Each block = one slide: **Title**, **On the slide** (short bullets), **Visual** (file in `docs/figures/`), **Speaker notes** (what to say). All numbers come from the benchmark in this repository (real data, leak-free 5-fold cross-validation).

---
### Slide 1 — Title
**HilbertFuse: a leak-free, layout-aware CNN fused with a linear model for cancer gene-expression classification**
Speaker notes: This work starts from the Frontiers in Genetics 2021 paper "MI_DenseNetCAM" (pan-cancer classification with mutual information and DenseNet), checks its weak points on real data, and builds a usable tool around the result. Research use only.

### Slide 2 — The problem
* Cancer tissue can be told apart by which genes are active (RNA/microarray expression).
* Data: thousands of genes (columns) but only dozens to hundreds of samples (rows) → easy to over-fit.
* Goal: given a new expression profile, predict which known tumor class it resembles, with an honest accuracy estimate.
Speaker notes: "p ≫ n" problem. Our system does NOT find the cause of a cancer and is NOT a diagnostic device; it recognises patterns.

### Slide 3 — What the base paper did (and what we found)
* Paper: Min-Max → mutual-information (MI) top 3600 genes → 60×60 gene image (row-major) → DenseNet → Guided Grad-CAM; reported 96.81 % on 33 TCGA tumor types.
* Weak points we found: (1) gene selection used all samples before cross-validation (**leakage**); (2) arbitrary pixel layout; (3) redundant genes; (4) weak classical baselines; (5) accuracy swings by 3.9 points with image size (Table 6); (6) one run, no variance.
Speaker notes: These come from reading Tables 4, 5 and 6 of the paper: preprocessing, classifiers and image-size results.

### Slide 4 — Real data used
* 8 published cancer data sets, 60–181 samples each, 2,000–12,600 genes: alon (colon), golub (leukemia), gordon (lung), khan (4 childhood tumors), pomeroy (brain tumor survival), shipp (lymphoma), singh (prostate), su (4 carcinoma types).
* All results use stratified 5-fold cross-validation; every preprocessing step is re-fitted inside each training fold.
Speaker notes: small cohorts are realistic for many labs; they are not the 10,000-sample TCGA set of the paper, so conclusions are for small-data settings.

### Slide 5 — System overview
**Visual:** `fig_workflow.png`
* Clean → select genes → order genes → gene image → CNN ‖ logistic regression → fuse → prediction + confidence + explanation.
Speaker notes: walk left to right; the two branches at the bottom are trained independently and their probabilities are averaged.

### Slide 6 — Step 1: leak-free cleaning
* If all values ≥ 0 and heavy-tailed (99.9th percentile > 50): x ← log2(x + 1).
* Min-Max per gene using **training-fold** minimum/maximum only: y = (x − min) / (max − min); new data is clipped to [0, 1].
* Constant genes are removed.
Speaker notes: the paper normalised and selected genes on all samples; that lets test information leak into training and inflates accuracy.

### Slide 7 — Step 2: gene selection (MI + redundancy pruning)
* Prefilter by ANOVA F-score to the top ~8,000 genes (speed).
* **Mutual information** with the class label: I(X;Y) = Σ p(x,y) · log[ p(x,y) / (p(x)p(y)) ] (k-nearest-neighbour estimator, k = 3).
* **Redundancy pruning:** walk genes in decreasing MI; skip a gene if |Pearson r| > 0.9 with a gene already kept → 1024 genes that are informative and not copies of each other.
Speaker notes: filter methods are fast; pruning avoids filling the budget with near-duplicate genes (Table 4 of the paper compares filters but ignores redundancy).

### Slide 8 — Step 3: gene image with a Hilbert curve
**Visual:** `fig_layout.png`
* Cluster the selected genes by co-expression (average linkage, 1 − correlation distance); the leaf order puts similar genes next to each other.
* Place the ordered list on a **Hilbert space-filling curve** on a 2^k × 2^k grid (1024 genes → 32×32): consecutive genes are always neighbouring pixels.
* Row-major fill (paper): gene i+1 is the right neighbour but gene i+32 is the lower neighbour → convolutions see arbitrary neighbours.
Speaker notes: a CNN assumes nearby pixels are related. The Hilbert curve preserves 1-D closeness in 2-D, so co-expressed gene modules become local patterns.

### Slide 9 — Step 3b: three channels
* Channel 0: expression (Min-Max).
* Channel 1: MI prior = MI(gene) / max MI — tells the network which genes matter (same for all samples).
* Channel 2: per-sample rank percentile among the selected genes — robust to scale and batch effects.
Speaker notes: the rank channel is invariant to monotone rescaling of a sample.

### Slide 10 — HilbertFuseNet (the CNN)
**Visual:** `fig_architecture.png`
* BatchNorm → 3×3 conv stem → 4 **dense blocks** (4, 6, 8, 6 layers, growth rate 12): layer ℓ receives the concatenation of all earlier outputs x_ℓ = H_ℓ([x_0, …, x_{ℓ−1}]).
* Transition layers: BN-ReLU-1×1 conv (compress ×0.5) - average pool.
* **SE gate** after each block: g = 2·σ(W₂ ReLU(W₁ GAP(x))); x ← x·g; W₂ starts at 0 so the gate starts as identity.
* **Multi-scale head:** global-average features of blocks 2, 3 and the last block are concatenated → dropout 0.3 → linear → softmax. About 0.4 M parameters.
Speaker notes: dense connections reuse features and have a regularising effect; SE lets the net re-weight feature maps; the multi-scale head reduces sensitivity to image size.

### Slide 11 — Training recipe (small-data regularisation)
* AdamW (lr 1e-3, weight decay 1e-4), cosine learning-rate schedule, batch 16, 50–80 epochs; runs on CPU.
* Loss: cross-entropy with label smoothing 0.1 and class weights w_c = N / (C · N_c).
* Augmentation: random gene dropout (10 % of pixels), Gaussian noise σ = 0.02, **mixup**: x = λ·x_i + (1−λ)·x_j, loss = λ·L(y_i) + (1−λ)·L(y_j), λ ~ Beta(0.3, 0.3).
Speaker notes: with 40–150 training samples a CNN over-fits easily; these regularisers are the reason the CNN is usable at all.

### Slide 12 — Fusion with a linear model
* Logistic regression on the same selected genes (standardised, L2, C chosen by inner 3-fold CV from {1e-3 … 10}); trains in seconds.
* **HilbertFuse** prediction: p = 0.5 · p_CNN + 0.5 · p_LogReg.
* Why: on small cohorts linear models are very strong, CNNs are higher-variance; averaging stabilises. The CNN adds gene-level explanation.
Speaker notes: this is the honest finding of the benchmark — see the results slides.

### Slide 13 — How the model gives a prediction
* Input: one row of gene values per sample (same gene columns as training).
* Output per sample: predicted class, **confidence** (its probability), runner-up class, and an **UNCERTAIN** flag when confidence < 0.70.
```
  prediction  confidence runner_up  runner_up_prob  uncertain
0        ALL       0.971       AML           0.029      False
1        ALL       0.914       AML           0.086      False
```
(Real output of the `golub` quick model on rows of its own training file — shown only to illustrate the format.)
* Safety checks: refuses files that lack the model's genes (coverage < 50 %).
Speaker notes: confidence is a model probability, not a guarantee; low confidence means "do not trust this answer".

### Slide 14 — Explainability
* **Guided Grad-CAM:** Grad-CAM weights α_k = mean over pixels of ∂y_c/∂A_k; map L = ReLU(Σ_k α_k A_k), upsampled; guided back-propagation keeps only positive gradients at ReLUs; importance = L ⊙ |guided gradient|.
* Averaged per class over correctly classified samples; pixel → gene via the Hilbert index → ranked gene list. Quick (linear) mode ranks genes by standardised coefficients.
Speaker notes: gene lists are leads for follow-up (differential expression, enrichment with FDR, independent cohorts); they are not proof of cause.

### Slide 15 — Evaluation protocol
* Stratified 5-fold CV; all preprocessing re-fitted per fold; same 1024-gene budget for all methods.
* Metrics (class-weighted, as in the paper): Accuracy = (TP+TN)/all; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R). Also macro-F1.
* Always compare with the majority-class rate and with tuned logistic regression / SVM / KNN; paired Wilcoxon test over data set × fold pairs.
Speaker notes: with 18–55 test samples a single split is noisy (one error = 2–5 points), so cross-validation is the number to trust.

### Slide 16 — Results: methods (paper Table 2 style)
<<TABLE2>>
**Visual:** `fig_results.png` (accuracy per data set)

### Slide 17 — Results: classifiers (paper Table 5 style)
<<TABLE5>>

### Slide 18 — Results: gene-selection strategies (paper Table 4 style)
<<TABLE4>>

### Slide 19 — Results: number of genes / image dimension (paper Table 6 style)
<<TABLE6>>

### Slide 20 — Honest findings
<<FINDINGS>>

### Slide 21 — Per-model scores
* One table per pretrained model (8 models) in `docs/PAPER_STYLE_TABLES.md` / `.html`; menu option 7 or `hilbertfuse scores` prints the cross-validated accuracy of every model.
* Model `pomeroy` (survival prediction) is **not useful**: no better than always answering the most common class.

### Slide 22 — Use it: clone, run, predict
* `git clone <repo>` → double-click `start_windows.bat` (first run installs) → menu: **2 Predict**, **3 Evaluate**, **7 Scores**.
* Give your cancer/patient expression CSV (rows = samples, same gene columns as the model). Runs offline on your computer.
Speaker notes: keep patient files out of git (`data_private/`).

### Slide 23 — Train your own model fast on CPU
* Menu option 4 or `hilbertfuse train --train data.csv --preset linear|fast|fuse`:
  **linear** ≈ seconds (logistic regression, gene coefficients as explanation) · **fast** ≈ 1–3 min (small CNN + linear) · **fuse** ≈ several minutes (full).
* `hilbertfuse cv` gives an honest cross-validated accuracy before you trust the model.

### Slide 24 — Limitations and ethics
* Small cohorts, one platform each; accuracy on other hospitals/platforms unknown (batch effects).
* Models only answer with the classes they were trained on.
* They do **not** identify the cause of a cancer and must **not** be used to diagnose or treat patients; clinical use would need large multi-site data, prospective validation and regulatory approval.

### Slide 25 — Future work
* Large multi-site cohorts with real gene symbols (e.g. TCGA tables), external validation, calibrated probabilities, pretrained backbones, stronger multi-seed statistics.

### Slide 26 — References
* Wang J. et al. (2021) MI_DenseNetCAM, Front. Genet. 12:670232 · Huang G. et al. (2017) Densely connected convolutional networks · Hu J. et al. (2018) Squeeze-and-excitation networks · Selvaraju R. et al. (2017) Grad-CAM · Springenberg J. et al. (2015) Guided backpropagation · Zhang H. et al. (2018) mixup · Hilbert D. (1891) space-filling curve · Kraskov A. et al. (2004) Estimating mutual information · data sets: Alon 1999, Golub 1999, Khan 2001, Su 2001, Gordon 2002, Pomeroy 2002, Shipp 2002, Singh 2002.
