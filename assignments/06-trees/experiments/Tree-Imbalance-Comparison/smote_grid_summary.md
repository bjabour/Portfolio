# SMOTE Grid Tuning

- Outer evaluation: 5-fold stratified CV repeated 5 times.
- Neighbor counts: [2, 3, 4].
- Target minority-to-majority ratios: [0.2, 0.3, 0.4].
- Fixed tree: max depth 4, minimum leaf size 5.
- Single-fit selection metric: mean log loss across the 25 untouched validation folds.
- Original minority-to-majority ratio: 47/253 = 0.1858.
- Best single-fit setting: ratio `0.2`, `k_neighbors=4`.
- Best mean fold log loss: 1.373417.
- Best repeated-prediction ensemble setting: ratio `0.3`, `k_neighbors=2`.
- Its log loss after averaging five out-of-fold predictions per row: 0.393516.

Scaling and SMOTE were fitted only inside each outer training fold.
The ensemble score is smoother because each observation's five predictions
are averaged before scoring; it is not the expected score of one fitted tree.
