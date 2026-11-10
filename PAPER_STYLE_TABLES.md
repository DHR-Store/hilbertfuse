# Paper-style result tables (real data, leak-free 5-fold cross-validation)

Format follows Tables 1, 2, 4, 5, 6 of Wang et al. 2021. Precision/Recall/F1 are class-weighted averages. Data: 8 real published cancer expression data sets (60-181 samples each). Research use only.

### TABLE 1 | Parameter settings.

| Algorithm | Parameter |
|---|---|
| HilbertFuseNet | genes kept = 1024 (image 32x32 Hilbert), MI + redundancy pruning (abs(r) > 0.9), 3 channels, AdamW lr=0.001, weight_decay=1e-4, cosine schedule, epochs=50 (benchmark) / 80 (saved models), batch_size=16, growth_rate=12, blocks=(4,6,8,6), init_features=24, bn_size=4, dropout=0.1/0.3, SE reduction=8, mixup alpha=0.3, label smoothing=0.1, gene dropout=0.1, noise std=0.02, class-weighted loss |
| Paper-style DenseNet | same CNN size and budget; row-major layout, plain MI, 1 channel, no SE / multi-scale head, no dropout, no mixup, no label smoothing, no class weights |
| HilbertFuse | HilbertFuseNet probabilities averaged 50/50 with MI_LogReg probabilities |
| MI_LogReg | MI top-1024 genes, standardised; LogisticRegressionCV, C in {1e-3, 1e-2, 0.1, 1, 10}, 3-fold inner CV |
| MI_SVM | MI top-1024 genes, standardised; LinearSVC, C in {1e-3, 1e-2, 0.1, 1}, 3-fold inner GridSearchCV |
| MI_KNN | n_neighbors = 5 on MI top-1024 genes scaled to [0, 1] |

*Same 1024-gene budget for every method. 5-fold stratified CV; all preprocessing re-fit inside each training fold.*

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

### TABLE 6b | CNN image-size check on two data sets.

| Setting | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| khan: HilbertFuseNet, 16 x 16 (256 genes) | 95.38% | 97.23% | 95.38% | 95.38% |
| khan: HilbertFuse, 16 x 16 (256 genes) | 98.46% | 98.77% | 98.46% | 98.46% |
| khan: HilbertFuseNet, 24 x 24 (32 x 32 Hilbert grid) (576 genes) | 96.79% | 97.64% | 96.79% | 96.79% |
| khan: HilbertFuse, 24 x 24 (32 x 32 Hilbert grid) (576 genes) | 100.00% | 100.00% | 100.00% | 100.00% |
| khan: HilbertFuseNet, 32 x 32 (1024 genes) | 100.00% | 100.00% | 100.00% | 100.00% |
| khan: HilbertFuse, 32 x 32 (1024 genes) | 100.00% | 100.00% | 100.00% | 100.00% |
| golub: HilbertFuseNet, 16 x 16 (256 genes) | 97.33% | 97.33% | 97.33% | 97.33% |
| golub: HilbertFuse, 16 x 16 (256 genes) | 97.33% | 97.33% | 97.33% | 97.33% |
| golub: HilbertFuseNet, 24 x 24 (32 x 32 Hilbert grid) (576 genes) | 88.86% | 92.65% | 88.86% | 88.70% |
| golub: HilbertFuse, 24 x 24 (32 x 32 Hilbert grid) (576 genes) | 95.90% | 96.14% | 95.90% | 95.93% |
| golub: HilbertFuseNet, 32 x 32 (1024 genes) | 87.33% | 92.25% | 87.33% | 86.82% |
| golub: HilbertFuse, 32 x 32 (1024 genes) | 94.48% | 94.95% | 94.48% | 94.52% |

*Small check only (2 data sets, 5 folds).*

### TABLE M-alon | Pretrained model `alon`: colon tumor vs normal (62 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 79.36% | 81.22% | 79.36% | 78.59% | ± 9.85 |
| HilbertFuseNet (CNN only) | 72.95% | 76.72% | 72.95% | 71.25% | ± 12.23 |
| Paper-style DenseNet | 71.15% | 74.22% | 71.15% | 69.74% | ± 5.93 |
| MI_LogReg | 82.69% | 83.89% | 82.69% | 82.29% | ± 13.63 |
| LogReg (all genes) | 76.28% | 76.83% | 76.28% | 74.89% | ± 13.88 |
| MI_SVM | 80.90% | 84.47% | 80.90% | 81.03% | ± 10.23 |
| MI_KNN | 79.23% | 81.87% | 79.23% | 78.23% | ± 11.96 |


### TABLE M-golub | Pretrained model `golub`: ALL vs AML leukemia (72 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 94.48% | 94.95% | 94.48% | 94.52% | ± 5.64 |
| HilbertFuseNet (CNN only) | 87.33% | 92.25% | 87.33% | 86.82% | ± 17.77 |
| Paper-style DenseNet | 86.10% | 88.27% | 86.10% | 85.01% | ± 5.08 |
| MI_LogReg | 98.67% | 98.79% | 98.67% | 98.62% | ± 2.98 |
| LogReg (all genes) | 98.67% | 98.79% | 98.67% | 98.62% | ± 2.98 |
| MI_SVM | 95.90% | 96.14% | 95.90% | 95.93% | ± 6.02 |
| MI_KNN | 97.24% | 97.50% | 97.24% | 97.15% | ± 3.79 |


### TABLE M-gordon | Pretrained model `gordon`: lung: mesothelioma vs adenocarcinoma (181 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 100.00% | 100.00% | 100.00% | 100.00% | ± 0.00 |
| HilbertFuseNet (CNN only) | 69.26% | 90.01% | 69.26% | 70.72% | ± 23.34 |
| Paper-style DenseNet | 98.89% | 99.17% | 98.89% | 98.95% | ± 2.48 |
| MI_LogReg | 99.44% | 99.46% | 99.44% | 99.42% | ± 1.24 |
| LogReg (all genes) | 99.44% | 99.46% | 99.44% | 99.42% | ± 1.24 |
| MI_SVM | 99.44% | 99.52% | 99.44% | 99.46% | ± 1.24 |
| MI_KNN | 98.89% | 98.92% | 98.89% | 98.85% | ± 1.52 |


### TABLE M-khan | Pretrained model `khan`: 4 small round blue cell tumors (63 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 100.00% | 100.00% | 100.00% | 100.00% | ± 0.00 |
| HilbertFuseNet (CNN only) | 100.00% | 100.00% | 100.00% | 100.00% | ± 0.00 |
| Paper-style DenseNet | 70.26% | 71.68% | 70.26% | 68.78% | ± 15.21 |
| MI_LogReg | 97.71% | 98.17% | 97.71% | 97.66% | ± 3.91 |
| LogReg (all genes) | 98.90% | 99.12% | 98.90% | 98.90% | ± 2.91 |
| MI_SVM | 97.80% | 98.39% | 97.80% | 97.70% | ± 5.81 |
| MI_KNN | 92.22% | 94.55% | 92.22% | 92.14% | ± 4.45 |


### TABLE M-pomeroy | Pretrained model `pomeroy`: brain tumor survival (hard) (60 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 60.00% | 62.09% | 60.00% | 58.59% | ± 16.03 |
| HilbertFuseNet (CNN only) | 53.33% | 53.07% | 53.33% | 52.55% | ± 16.24 |
| Paper-style DenseNet | 50.00% | 50.25% | 50.00% | 49.36% | ± 16.67 |
| MI_LogReg | 61.67% | 61.75% | 61.67% | 59.34% | ± 12.64 |
| LogReg (all genes) | 65.00% | 61.64% | 65.00% | 60.39% | ± 13.69 |
| MI_SVM | 58.33% | 63.17% | 58.33% | 58.58% | ± 16.67 |
| MI_KNN | 48.33% | 54.41% | 48.33% | 48.19% | ± 6.97 |


### TABLE M-shipp | Pretrained model `shipp`: DLBCL vs follicular lymphoma (77 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 92.00% | 93.33% | 92.00% | 92.42% | ± 11.93 |
| HilbertFuseNet (CNN only) | 85.33% | 90.83% | 85.33% | 86.33% | ± 14.45 |
| Paper-style DenseNet | 89.33% | 92.41% | 89.33% | 89.01% | ± 15.35 |
| MI_LogReg | 92.00% | 93.56% | 92.00% | 92.29% | ± 10.95 |
| LogReg (all genes) | 94.75% | 95.56% | 94.75% | 94.73% | ± 2.94 |
| MI_SVM | 83.00% | 90.81% | 83.00% | 84.18% | ± 10.28 |
| MI_KNN | 89.50% | 93.43% | 89.50% | 90.09% | ± 9.04 |


### TABLE M-singh | Pretrained model `singh`: prostate tumor vs normal (102 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 93.24% | 94.23% | 93.24% | 93.18% | ± 6.45 |
| HilbertFuseNet (CNN only) | 92.29% | 93.36% | 92.29% | 92.23% | ± 7.26 |
| Paper-style DenseNet | 88.29% | 88.78% | 88.29% | 88.26% | ± 5.42 |
| MI_LogReg | 91.24% | 92.06% | 91.24% | 91.19% | ± 6.34 |
| LogReg (all genes) | 91.29% | 91.97% | 91.29% | 91.25% | ± 7.05 |
| MI_SVM | 91.24% | 92.35% | 91.24% | 91.16% | ± 6.34 |
| MI_KNN | 90.24% | 91.03% | 90.24% | 90.19% | ± 3.29 |


### TABLE M-su | Pretrained model `su`: 4 carcinoma tissue types (102 samples, 5-fold CV).

| Classifier | Accuracy | Precision | Recall | F1-Score | Accuracy std |
|---|---|---|---|---|---|
| HilbertFuse (CNN + linear) | 95.24% | 96.79% | 95.24% | 94.85% | ± 6.73 |
| HilbertFuseNet (CNN only) | 91.38% | 90.12% | 91.38% | 89.47% | ± 10.30 |
| Paper-style DenseNet | 95.14% | 96.12% | 95.14% | 95.01% | ± 3.37 |
| MI_LogReg | 98.10% | 98.39% | 98.10% | 98.07% | ± 2.61 |
| LogReg (all genes) | 98.10% | 98.39% | 98.10% | 98.07% | ± 2.61 |
| MI_SVM | 98.10% | 98.39% | 98.10% | 98.07% | ± 2.61 |
| MI_KNN | 98.10% | 98.39% | 98.10% | 98.07% | ± 2.61 |

