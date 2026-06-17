# Loss terms in `compute_score`

Default weights and reference papers. All terms normalised to roughly
[0, 1] so weights are directly comparable.

| Term | Default weight | What it measures | Reference |
|---|---|---|---|
| `sil_loss` | 0.5 | Silhouette overlap (Tversky α=0.7, β=0.3) | Salehi 2017 (Tversky) |
| `col_loss` | 0.4 | Marginal per-channel RGB histogram χ² | – |
| `spat_loss` | 0.6 | Per-distance-bin RGB mean L1 (from spatial-color model) | (this work, v2) |
| `cond_loss` | 0.4 | Conditional brightness distribution KL | (this work, v3) |
| `fft_loss` | 0.4 | Log-power FFT spectrum (low frequencies) L1 | (this work, v4) |
| `ssim_loss` | 0.5 | 1 − SSIM(luma) | Wang 2004 |
| `lab_loss` | 0.3 | Lab-space percentile L1 (10/25/50/75/90) | (this work, v5a) |
| `biou_loss` | 1.0 | 1 − Boundary IoU within ±3 px outline | Cheng 2021 |
| `dist_loss` | 0.5 | Centre-of-mass distance / image diagonal | Zheng 2020 (DIoU) |
| `aspect_loss` | 0.3 | abs(log(h/w) target − log(h/w) proc) / 2 | Zheng 2020 (CIoU variant) |
| `band_loss` | 0.6 | Per-band silhouette mass + mean RGB | (this work) |
| `vgg_loss` | 0.5 | VGG16 relu_3_3 feature L2, normalised by 30 | Johnson 2016 |

## Why Tversky over IoU?

The 4-way ablation on `tree_pine.png` showed Tversky α=0.7 / β=0.3
beating raw IoU (1 − TP/(TP+FP+FN)):

```
Variant                            score   IoU    b_iou
(a) w_biou=1.5 (more boundary)     2.910   29%    0.47
(b) broadleaf preset on pine       — (sprite lib missing)
(c) Tversky asymmetric             2.487   34%    0.46  ★
(d) VGG perceptual                10.808  ★bug ↑ (vgg loss un-normalised)
```

Tversky α=0.7 penalises false positives (sprawl) 2.3× harder than false
negatives (gaps). This biases the search toward conservative,
boundary-accurate fits — exactly what IoU misses when an "expand the
blob to cover everything" strategy gives high IoU but bad boundary
match.

## Why Boundary IoU as the headline metric?

Standard IoU rewards filling the silhouette interior. A solid blob the
size of the photo's bbox gets 100 % IoU regardless of branch detail.
Boundary IoU (Cheng CVPR 2021) computes IoU over only the outline
±3 px, so it punishes
- silhouettes that are too smooth (no branch indentations)
- silhouettes that are too noisy (jagged outline)

In our experiments b_iou correlates much better with "looks like the
photo" than raw IoU.

## Why is `cond_loss` biased against Weber-Penn?

`cond_loss` measures KL divergence between target and proc on a 2D
joint distribution of (distance-from-centre bin, brightness bin). SCA
produces relatively homogeneous leaf clusters around the crown
ellipsoid surface, giving smooth conditional histograms similar to
photographed trees. Weber-Penn produces clustered foliage along branch
tips with visible branch structure, giving a less uniform conditional
histogram even when the silhouette and colour match.

This biases total score toward SCA in spite of WP producing better
visual fits. Lowering `--w-cond 0.2` removes the penalty but lets the
optimiser produce too-sparse trees (because the cond term was also
implicitly enforcing foliage density). Best practice: keep `w_cond` at
its default and read sil/biou/IoU separately when comparing backends.
