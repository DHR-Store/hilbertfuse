# Heat maps for HilbertFuse (paper Figure 1B and Figure 4 style)

## Generate them (VS Code / PowerShell, environment activated)
```powershell
pip install matplotlib                     # only if your .venv was created before this update
hilbertfuse heatmap --model models\golub --data my_samples.csv --out heatmaps
hilbertfuse heatmap --model models\golub --data my_labeled.csv --label-col label --top-k 40 --sample 0 --out heatmaps
```
* **Menu:** `start_windows.bat` -> option **8**. **VS Code:** Terminal > Run Task > *HilbertFuse: heat maps*.
* `--data` must contain the model's genes (same check as `predict`). With a `label` column the **true** classes are used; without it (new patient file) the **predicted** classes are used and the figures say so.
* `--top-k 40` genes in the expression map · `--sample 0` also draws sample number 0 (first row) · `--max-per-class 100` limits samples used for the maps (speed).
* Works for `fuse`, `hfnet`, `*_ens` and the `linear` quick models (the CNN gene-image maps are skipped for `linear` because there is no CNN).

## What you get (in `heatmaps\`)
| file | shows | paper |
|---|---|---|
| `heatmap_gene_image_all_classes.png`, `heatmap_gene_image_<class>.png` | **Guided Grad-CAM on the Hilbert gene image**, mean over the samples of each class (correctly classified ones when at least 3 exist). Bright pixel = gene the CNN relied on; circles name the top genes. Grey = no gene on that pixel. | Fig. 1B / heat map used for gene selection |
| `heatmap_top_genes_expression.png` | expression (rows = top genes, columns = samples grouped by class, colour bar on top); z-score per gene, blue low / red high. Genes are ordered by the class where they are highest, so a block pattern appears when the genes separate the classes. | Fig. 4 |
| `heatmap_class_gene_scores.png` | importance of each top gene for each class (class x gene). Linear models: signed coefficients. | - |
| `heatmap_sample_<i>.png` (with `--sample i`) | one sample: its gene image, the map for its **predicted** class (with confidence) and an overlay: *why this sample got this answer*. | - |
| `heatmap_gene_scores.csv` | the numbers behind all figures (global and per class). | Table 7 |

Examples (real pretrained models): `docs/figures/heatmaps/` — `golub_*.png` (ALL vs AML) and `khan_gene_image.png` (4 tumor types).

## How the maps are computed
1. The sample is turned into the 3-channel Hilbert gene image exactly as in training (training min-max statistics).
2. **Grad-CAM** for class c: weights alpha_k = mean over pixels of dy_c/dA_k for each feature map A_k of the last block; map L = ReLU(sum_k alpha_k * A_k), up-sampled to the image size (coarse, class-specific).
3. **Guided back-propagation**: gradient of y_c w.r.t. the input pixels with negative gradients removed at every ReLU (pixel-sharp).
4. **Guided Grad-CAM** = L x |guided gradient| (summed over the 3 channels), normalised to 0-1 per sample.
5. Averaged over the samples of the class; each pixel is mapped back to its gene through the Hilbert index (`pix_`) so every bright pixel has a gene name.
In fused models (`fuse`) the maps come from the **CNN member**; the linear member contributes coefficients only in the quick `linear` models.

## How to read them (and what not to conclude)
* They show **what the model used**, not what causes the disease. Treat top genes as leads: check them with a differential-expression test, enrichment analysis with FDR correction, literature, and an independent cohort.
* On the real Golub data (`models\golub`) the top genes include probe IDs such as `M23197_at` (to my knowledge CD33, a known AML marker), which is the kind of sanity check you want; but the probe IDs are not gene symbols. To get symbols, map them with the array's annotation file. Models `khan`, `su`, `shipp`, `singh` have anonymous column names (GENE123) and cannot be mapped.
* Heat maps drawn on the **training** samples describe the model, not its accuracy. Never use them as evidence that the model generalises; use cross-validation (`hilbertfuse cv`) for that.
* Maps depend on the random seed and the training run. For stable conclusions train 3 or more models and keep genes that recur.
* Low confidence + a diffuse map = the model had no clear pattern for that sample: treat the prediction as uncertain.

## Heat maps for your own model
```powershell
hilbertfuse train --train data.csv --label-col label --preset fast --out models\mymodel     # CNN + linear -> gene-image maps available
hilbertfuse heatmap --model models\mymodel --data data.csv --out heatmaps_mine --sample 0
```
(`--preset linear` trains in seconds but gives only the gene heat maps.)
