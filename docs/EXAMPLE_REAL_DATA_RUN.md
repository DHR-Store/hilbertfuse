# Example output: `hilbertfuse real-demo --dataset all --epochs 50`

One stratified 70/30 train/test split per real cancer data set (seed 0), preset `fuse` (HilbertFuse = CNN + tuned logistic regression), 1,024 genes, random initialisation, one CPU core, about 9 minutes in total. Your numbers should be very close but not necessarily identical (different CPU/library versions change floating-point results).

| dataset   |   train |   test |   genes |   accuracy |   macro_f1 |   logreg_baseline_acc |   minutes |
|:----------|--------:|-------:|--------:|-----------:|-----------:|----------------------:|----------:|
| alon      |      43 |     19 |    2000 |      0.895 |      0.878 |                 0.895 |       0.5 |
| golub     |      50 |     22 |    7129 |      0.909 |      0.902 |                 0.909 |       1.1 |
| gordon    |     126 |     55 |   12533 |      1     |      1     |                 0.982 |       2.1 |
| khan      |      44 |     19 |    2308 |      0.947 |      0.962 |                 0.947 |       0.6 |
| pomeroy   |      42 |     18 |    7128 |      0.778 |      0.75  |                 0.722 |       0.9 |
| shipp     |      53 |     24 |    7129 |      1     |      1     |                 0.958 |       1.1 |
| singh     |      71 |     31 |   12600 |      0.903 |      0.903 |                 0.903 |       1.5 |
| su        |      71 |     31 |    5565 |      1     |      1     |                 1     |       1.4 |

Mean test accuracy: HilbertFuse **0.929**, tuned logistic regression **0.914**.

**How to read this honestly**
* Test sets are tiny (18-55 samples): one wrong sample changes accuracy by 2-5 points. HilbertFuse differs from the logistic-regression baseline by at most one test sample on every data set, so these single splits do **not** show that it is better.
* The rigorous comparison (5-fold cross-validation, 7 data sets, paired tests) is in `docs/RESEARCH_HILBERTFUSE.md`: HilbertFuse is about equal to the best tuned linear model and clearly better than the paper-style CNN.
* `pomeroy` (predicting survival of brain-tumor patients) is genuinely hard: 0.78 here, near what simple models reach.
* Research use only; not a diagnostic tool.
