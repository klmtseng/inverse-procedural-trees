# inverse-procedural-trees

Fit a procedural tree model to a real tree photo. Given an alpha-keyed
tree photo, search over the parameter space of two procedural backends
(Space Colonization + Weber-Penn) until the synthesised silhouette,
foliage colour distribution, boundary outline and vertical structure
all match the photo.

Pure-Python, CPU-only, no neural networks required. Optional VGG16
perceptual loss for the curious.

> 中文說明在文件末尾 / [Chinese description at end](#中文說明)

## Input photos

The five test photos shipped with the repo (`photos/`):

![5 test trees](photos/contact_sheet.png)

## Best fits — left = original, right = procedural

Four of the strongest results (full 13-cell matrix is below):

![best fits](results/hero_grid.png)

---

## Result gallery (5 species × 2 backends = 10 fits)

Each cell is `original photo | INIT | best fit` after 4 DE iterations
(~300 evals, 5–15 min each on a 2020-era laptop CPU). Best metric per
species in **bold**.

| Species | SCA + Tversky | Weber-Penn |
|---|---|---|
| pine     | score 2.49 / IoU 33.7 / b\_iou 0.46 | score 2.66 / IoU 37.3 / **b\_iou 0.56** |
| broadleaf | score **2.37** / IoU 43.8 / b\_iou **0.54** | score 2.72 / IoU **47.7** / b\_iou 0.51 |
| olive    | score 2.64 / IoU 48.1 / b\_iou 0.39 | score 2.67 / IoU **52.9** / b\_iou **0.43** |
| fir      | score 2.55 / IoU **56.9** ★ / b\_iou 0.38 | score 3.56 / IoU 53.7 / b\_iou 0.40 |
| larch    | score 3.26 / IoU 21.0 / b\_iou 0.35 | score 3.51 / IoU 29.1 / b\_iou 0.18 |

★ Highest IoU achieved in this repo (fir + SCA, 56.9 %).

Result images live under [`results/`](./results/). Each file follows
`fit_v5b_<variant>_<photo>_<tree-idx>.png`.

### What we learned

- **Backend > species preset.** For broadleaf photos, Weber-Penn
  consistently wins the visual quality metrics (sil\_loss / boundary IoU
  / IoU). For conifer photos, SCA's compact ellipsoid envelope is
  already a good fit — sometimes a better fit. Choosing the wrong
  species preset within a backend costs at most ~3 % IoU; choosing the
  wrong backend can cost ~10 %.

- **`cond` term is structurally biased.** SCA produces a more
  homogeneous foliage density (all leaves clustered around the crown
  ellipsoid surface) which matches the photo's
  conditional-brightness-distribution histogram better than WP's
  branchier, less uniform output. So WP wins visual metrics but loses
  the conditional-brightness term. Setting `--w-cond 0.2` is tempting
  but produces too-sparse trees — the cond penalty is doing real work
  enforcing foliage density.

- **Larch is failure mode.** `tree_larch.png` is a multi-tree scene
  (autumn larch + dark conifer behind). Both backends struggle (IoU
  21-29 %) because the target silhouette isn't a single tree. A future
  pre-processing step to crop ROI per tree would fix this.

---

## Quick start

```bash
git clone https://github.com/klmtseng/inverse-procedural-trees
cd inverse-procedural-trees
pip install -r requirements.txt

# fit Weber-Penn to the broadleaf photo, Tversky asymmetric loss
PYTHONPATH=src python3 -m ipt.fit \
    --photo tree_broadleaf.png \
    --tree-idx 0 \
    --backend wp \
    --species broadleaf \
    --tversky \
    --max-iter 4 \
    --tag my_first_fit

# output: results/fit_my_first_fit_tree_broadleaf_0.png
```

For a fresh photo you need a sprite library first:

```bash
# put your photo in photos/, then
PYTHONPATH=src python3 -m ipt.extract_leaves \
    --photo my_tree.png \
    --n-templates 10 \
    --sprite-size 32
```

---

## Decision table — pick a backend for your photo

```
photo subject?           recommended backend / species
─────────────────────────────────────────────────────
mature broadleaf (oak,
maple, urban broadleaf)  →  --backend wp --species broadleaf
                            (highest visual quality, branchy structure)

small/young broadleaf
or shrub                 →  --backend sca --species broadleaf
                            (compact crown, score wins)

mature conifer (pine,
fir, spruce)             →  --backend sca --species pine
                            (compact ellipsoid envelope works best)

olive / fruit tree       →  --backend wp --species broadleaf
                            (broadleaf-like crown wins IoU)

multi-tree photo         →  crop ROI first; otherwise expect IoU < 30 %
```

When unsure, try `wp + broadleaf` first; it gives the visually best
output on 4 of 5 species in our test set.

---

## How it works

The fitter searches parameter space with SciPy's `differential_evolution`
to minimise a weighted multi-term objective:

```
score = w_sil  · sil_loss      ← Tversky (default) or IoU
      + w_col  · col_loss      ← marginal RGB histogram χ²
      + w_spat · spat_loss     ← per-distance-bin RGB mean L1
      + w_cond · cond_loss     ← conditional-brightness KL
      + w_fft  · fft_loss      ← log-power FFT spectrum
      + w_ssim · ssim_loss     ← 1 − SSIM (skimage)
      + w_lab  · lab_loss      ← Lab-space percentile L1
      + w_biou · biou_loss     ← Boundary IoU (Cheng 2021)
      + w_dist · dist_loss     ← Distance IoU (Zheng 2020)
      + w_aspect · aspect_loss ← bbox aspect ratio
      + w_band · band_loss     ← per-band mass + RGB
      + w_vgg  · vgg_loss      ← VGG16 perceptual (optional)
```

Each backend produces a (`segments`, `tip_positions`) tuple that the
rasteriser pastes into a 128×128 mask + a 128×128 RGB image with the
photo's own leaf sprites for foliage. Sprites are extracted up front by
`extract_leaves.py` using local maxima of the photo's distance
transform + K-means clustering — no neural network needed.

### Backends

- `src/ipt/backends/sca.py` — Space Colonization (Runions 2007) with
  multiple envelope shapes (conifer, oak, weeping, tiered broadleaf)
  + Murray's law branch thickening.
- `src/ipt/backends/wp.py` — Weber-Penn 1995 with 3 recursion levels
  (trunk → main scaffolds → twigs). Parameter vector covers the 19
  most-influential knobs.

### Loss innovations beyond v3

- **Tversky asymmetric** (Salehi 2017, default): `1 - TP / (TP + α·FP +
  β·FN)` with α=0.7 > β=0.3 — penalises foliage sprawl 2.3× harder than
  gaps. Wins our 4-way ablation.
- **Boundary IoU** (Cheng CVPR 2021): IoU of the silhouette outline
  within ±3 px instead of the filled interior. Most discriminative
  metric for tree shape similarity in our experiments.
- **Distance IoU** (Zheng AAAI 2020): center-of-mass distance
  normalised by image diagonal.
- **Vertical banding loss** (new here): per-band silhouette fraction +
  mean RGB matching, encouraging the optimiser to reproduce the photo's
  vertical structure rather than just its blob coverage.

---

## Repo layout

```
inverse-procedural-trees/
├── src/ipt/
│   ├── fit.py              ← main entry: DE search + objective + CLI
│   ├── backends/
│   │   ├── sca.py          ← Space Colonization (Runions 2007)
│   │   ├── wp.py           ← Weber-Penn adapter (flat param vector)
│   │   └── wp_raw.py       ← Weber-Penn 1995 raw generator
│   ├── target.py           ← photo → spatial-color model
│   ├── extract_leaves.py   ← sprite library builder
│   ├── render_util.py      ← polygon → PNG rasteriser
│   └── paths.py            ← centralised filesystem paths
├── photos/                 ← 5 tree photos used in our experiments
├── leaf_libs/              ← precomputed sprite libraries
├── results/                ← 13 reference fits
├── docs/
│   ├── metrics.md          ← per-term loss documentation
│   └── results_matrix.md   ← full 13-cell ablation table
├── README.md
└── LICENSE
```

---

## Known limitations

1. **CPU-only**: each fit takes 5–15 minutes single-threaded.
   Multi-photo / parallel DE eval could fix this — not implemented.
2. **2D rasterisation only**: no mesh / GLB / `.tree` export. The
   procedural parameters CAN be re-rendered in 3D (the SCA / WP modules
   return `segs` + `leaves` in world coordinates) — just not wired to
   bpy or three.js in this repo.
3. **No differentiable rendering**. ProcGen3D / CropCraft / Lopez 2023
   pipelines use PyTorch + neural net surrogates. This repo
   deliberately stays gradient-free for portability.
4. **5 photos is a small benchmark.** Generalisation beyond this set is
   unverified.

---

## Future work — GPU / deep learning extensions

The current pipeline is gradient-free by design (DE + CPU). DL
approaches in the recent literature (Lopez 2023, CropCraft 2024,
ProcGen3D 2025) produce visibly better fits but need PyTorch + a
discrete GPU. The natural progression, in increasing order of
complexity:

| Direction | What it adds |
|---|---|
| **VGG / CLIP perceptual loss** (already wired via `--vgg`) | Texture-aware similarity (catches "fluffiness" pixel χ² misses). Currently optional + slow on CPU — moving the forward pass to GPU makes it cheap enough to enable by default. |
| **Differentiable rendering** ([nvdiffrast](https://nvlabs.github.io/nvdiffrast/)) | Replace DE's 300 random evaluations with ~30 gradient descent steps. Expected 10–50× speedup per fit. |
| **CNN initialiser** | Train a small CNN to predict initial parameters from the photo's CLIP embedding, then refine with DE. Warm-start eliminates most early evals. |
| **Differentiable procedural model** ([ProcGen3D-style](https://arxiv.org/abs/2503.00045)) | Re-implement SCA / WP as differentiable PyTorch ops. End-to-end gradient through the procedural generator itself. |
| **Neural texture synthesis** | Replace photo-derived sprite library with a per-photo generative model (DCGAN / StyleGAN on cropped leaf patches). |

A discrete CUDA GPU (any modern card, e.g. RTX 3060 or better) is the
sensible jumping-off point. The single highest-ROI change would be
**switching SCA / WP rasterisation to nvdiffrast** so the existing
objective becomes gradient-based — that one swap puts per-fit cost in
the seconds rather than minutes, opening up much-larger sweeps and
ensemble fits.

---

## References

- Runions et al. 2007. *Modeling and visualization of leaf venation
  patterns.* ACM Transactions on Graphics. (SCA)
- Weber & Penn 1995. *Creation and rendering of realistic trees.*
  SIGGRAPH. (WP)
- Stava et al. 2014. *Inverse procedural modelling of trees.* Computer
  Graphics Forum.
- Salehi et al. 2017. *Tversky loss function for image segmentation
  using 3D fully convolutional deep networks.* MICCAI.
- Zheng et al. 2020. *Distance-IoU loss: Faster and better learning for
  bounding box regression.* AAAI.
- Cheng et al. 2021. *Boundary IoU: Improving object-centric image
  segmentation evaluation.* CVPR.
- Johnson, Alahi & Fei-Fei 2016. *Perceptual losses for real-time
  style transfer and super-resolution.* ECCV. (VGG perceptual loss)

---

## 中文說明

把真實樹照片轉成程序化樹模型的參數。給一張去背樹照片，用差分進化搜
索 Space Colonization 與 Weber-Penn 兩種後端的參數空間，直到合成樹的
剪影、葉色分布、邊界與垂直結構都符合照片。

純 Python、CPU-only、不需要神經網路。可選 VGG16 感知損失。

### 結論

- **後端比物種預設更重要**。闊葉樹照片用 Weber-Penn 視覺指標全贏；
  針葉樹用 SCA 緊湊橢球體已經很好，有時更好。
- **`cond` 項有結構性偏差**：SCA 的均質葉子分布天然較貼合照片條件
  亮度分布，WP 較不均質。降 `w_cond` 會讓樹變得太稀疏。
- **larch 是失敗案例**：多樹合照不適合單樹擬合。

### 怎麼用

詳見上方 Quick Start。

### 下一步：搭配獨立顯卡的深度學習擴展

本 repo 刻意是 gradient-free + CPU only，讓沒 GPU 的機器也能跑。
要再推進視覺品質，下一步的自然方向是**搭配獨立顯卡**做 DL 擴展：

| 方向 | 加什麼 |
|---|---|
| **VGG / CLIP perceptual loss**（已內建 `--vgg`）| 真正的紋理相似度（pixel loss 抓不到的「蓬鬆感」）。GPU 化之後快到可以預設開啟 |
| **可微分渲染** ([nvdiffrast](https://nvlabs.github.io/nvdiffrast/)) | 取代 DE 的 300 次隨機評估、改用 ~30 步 gradient descent。預期單張擬合速度 10–50× |
| **CNN 初始化器** | 訓小 CNN 從照片 CLIP embedding 預測初始參數、再用 DE 微調。Warm-start 砍掉前半部 eval |
| **可微分程序模型** ([ProcGen3D 風格](https://arxiv.org/abs/2503.00045)) | 把 SCA / WP 重寫成 PyTorch 可微分版，end-to-end gradient 一路通到程序生成器本身 |
| **神經紋理合成** | 用 per-photo 生成模型（DCGAN / StyleGAN on cropped leaves）取代照片切出來的 sprite 庫 |

整體最高 CP 值的第一步是 **把 SCA / WP rasterisation 換成 nvdiffrast**——
現有的 objective 立刻變成 gradient-based，單次擬合從幾分鐘掉到幾秒，
可以做更大規模 sweep + ensemble fit。

---

This repo was developed as a side-quest from the woodsmoke
diorama project (browser three.js cabin sim). The original goal was to
generate procedural billboard trees that look like real photos — turns
out fitting them is interesting in its own right.
