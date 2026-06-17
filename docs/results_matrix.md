# Results Matrix (13 ablations)

All fits use Differential Evolution, 4 iterations, popsize 3, Tversky
silhouette loss with α=0.7, β=0.3. Image files referenced live under
`../results/`.

## Main matched-bounds matrix (10 cells)

| Species | Backend | score↓ | sil↓ | b_iou↑ | IoU↑ | image |
|---|---|---|---|---|---|---|
| pine      | SCA | **2.487** | 0.46 | 0.460 | 33.7% | `fit_v5b_c_tree_pine_1.png` |
| pine      | WP  | 2.656 | 0.43 | **0.558** | **37.3%** | `fit_v5b_wp_pine_tree_pine_1.png` |
| broadleaf | SCA | **2.370** | 0.34 | **0.544** | 43.8% | `fit_v5b_broadleaf_tvk_tree_broadleaf_0.png` |
| broadleaf | WP  | 2.721 | 0.31 | 0.508 | **47.7%** | `fit_v5b_wp_broadleaf_tree_broadleaf_0.png` |
| olive     | SCA | **2.640** | 0.29 | 0.386 | 48.1% | `fit_v5b_sca_olive_tree_olive_0.png` |
| olive     | WP  | 2.668 | **0.26** | **0.434** | **52.9%** | `fit_v5b_wp_olive_tree_olive_0.png` |
| **fir**   | **SCA** | **2.547** | **0.24** | 0.381 | **56.9% ★** | `fit_v5b_sca_fir_tree_fir_0.png` |
| fir       | WP  | 3.559 | 0.25 | **0.401** | 53.7% | `fit_v5b_wp_fir_tree_fir_0.png` |
| larch     | SCA | **3.261** | 0.58 | **0.353** | 21.0% | `fit_v5b_sca_larch_tree_larch_0.png` |
| larch     | WP  | 3.511 | 0.48 | 0.184 | 29.1% | `fit_v5b_wp_larch_tree_larch_0.png` |

★ Project-wide IoU record (fir + SCA).

## Cross-bounds robustness (3 cells)

These tests deliberately mismatch the species preset to the photo —
e.g. broadleaf photo fit with `--species pine`. Tests whether the
optimiser can recover from a poor preset choice.

| Photo | Backend | Bounds | score↓ | IoU | b_iou | finding |
|---|---|---|---|---|---|---|
| broadleaf | SCA | pine | 2.484 | 43.5% | 0.526 | almost-tied with matched (2.370 / 43.8% / 0.544) |
| pine | SCA | broadleaf | 2.806 | 32.8% | 0.451 | noticeably worse than matched (2.487 / 33.7% / 0.460) |
| broadleaf | WP | pine | 3.604 | 54.5% | 0.475 | IoU record for broadleaf, but score explodes |

Takeaway: optimiser is fairly forgiving of preset mismatch within a
backend. For broadleaf photos, even pine bounds yield 43.5% IoU
(matched gets 43.8%). For pine photos, broadleaf bounds hurt
significantly (32.8% vs 33.7%).

## Ablation history (development side-quests)

Records earlier rounds where we tried (and abandoned) various ideas on
broadleaf:

| Variant | score↓ | b_iou | IoU | conclusion |
|---|---|---|---|---|
| broadleaf + Tversky (baseline) | 2.370 | 0.544 | 43.8% | champion in single-ellipsoid |
| + tier_vertical_squash | 2.389 | 0.485 | 39.8% | optimizer pushed to upper bound, no tiering |
| + multi-tier envelope | 2.472 | 0.387 | 42.5% | envelope creates tiers but sprites overlap them |
| + vertical banding loss | 2.613 | 0.474 | 39.0% | banding↓ from 0.43 to 0.30 (works), other metrics suffer |

Lesson: in v5b's rasterisation model (fluffy sprite clusters), procedural
tiering is hard to express in the silhouette — sprites blur layer gaps.
The Weber-Penn backend solves this by giving the optimiser actual
branch structure to align tiers with.
