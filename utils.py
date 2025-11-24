import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

def visualize_pca_one(feat, save_name="pca_single.png"):
    b, c, h, w = feat.shape
    flat = feat[0].permute(1, 2, 0).reshape(-1, c).cpu().numpy()

    pca = PCA(n_components=3)
    pca_feat = pca.fit_transform(flat)

    min_v, max_v = pca_feat.min(), pca_feat.max()
    img = (pca_feat - min_v) / (max_v - min_v + 1e-8)
    img = img.reshape(h, w, 3)

    img_up = F.interpolate(
        torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
        size=(224, 224), mode="nearest"
    )[0].permute(1, 2, 0).numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_up)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


    return img_up

