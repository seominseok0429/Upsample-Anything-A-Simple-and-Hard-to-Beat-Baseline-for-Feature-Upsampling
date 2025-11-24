🆕 [2025-11-24] : Initial code release (Given the high number of requests, we have decided to release the code in its current state before further cleanup.)

:warning: The current version is slightly slower than the one reported in the paper, because the speed-optimized code has not yet been fully integrated due to compatibility issues across different versions.

# Upsample Anything 

**KAIST, MIT, Microsoft**

Minseok Seo, Mark Hamilton, Changick Kim

[ :scroll: [`Paper`](https://arxiv.org/html/2511.16301v1)] [ :globe_with_meridians: [`Website`](https://seominseok0429.github.io/Upsample-Anything/)] [ :book: [`BibTeX`](#-)]

## Overview

<div align="center">
  <img width="1364" height="1024" alt="market" src="./img/main.png" />

  <i>Our method performs lightweight test-time optimization (≈0.419 s/image) without requiring any dataset-level training.It generalizes seamlessly across domains while maintaining consistent reconstruction quality for every image. (All examples are randomly selected, without cherry-picking.)</i>
</div>


### Simple usage, similar to OpenCV’s `cv2.resize`

```python

img_path = "sample.png"
img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC) # <- Any Target Size
lr_feature = dinov2_infer(img_path) # <- any low resolution modalities (e.g. depth, Probability map, feature map)
hr_feature = UPA(img, lr_feature) 
```

### Acknowledgement

We sincerely thank [Young-Tack Oh](https://scholar.google.com/citations?user=fkjUawMAAAAJ&hl=ko) for checking the typos in our paper, and we also thank [Young-Jae Park](https://scholar.google.com/citations?user=eaT9X2MAAAAJ&hl=ko) for revising our figures.
