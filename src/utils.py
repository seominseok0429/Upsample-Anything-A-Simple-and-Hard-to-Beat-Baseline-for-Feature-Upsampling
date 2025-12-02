import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image

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

def voc_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap

palette = voc_colormap(21)


def prob_to_color(prob_3d):
    seg = np.argmax(prob_3d, axis=0)         # (H,W)
    color = palette[seg]                     # (H,W,3)
    return color



def overlay(img_np, seg_color, alpha=0.5):
    return (img_np * (1-alpha) + seg_color * alpha).astype(np.uint8)


def visualize_overlay(img, lr_prob, hr_prob):
    """
    img: PIL image (448×448)
    lr_prob: (1, 21, h, w)
    hr_prob: (1, 21, 448, 448)
    """

    img_np = np.array(img)

    lr_np = lr_prob.squeeze(0).detach().cpu().numpy()     # (21,h,w)
    lr_color = prob_to_color(lr_np)                       # (h,w,3)

    lr_color_up = Image.fromarray(lr_color).resize((448, 448), Image.NEAREST)
    lr_color_up = np.array(lr_color_up)                   # (448,448,3)

    lr_overlay = overlay(img_np, lr_color_up, alpha=0.8)

    hr_np = hr_prob.squeeze(0).detach().cpu().numpy()     # (21,448,448)
    hr_color = prob_to_color(hr_np)                       # (448,448,3)

    hr_overlay = overlay(img_np, hr_color, alpha=0.8)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original", fontsize=15)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lr_overlay)
    plt.title("LR Overlay (50%)", fontsize=15)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr_overlay)
    plt.title("HR Overlay (50%)", fontsize=15)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def fit_pca_basis(feature_4d):
    """
    feature_4d: [1, C, H, W]
    Return PCA model
    """
    feat = feature_4d[0].detach().cpu().numpy()   # [C,H,W]
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T                    # [HW, C]

    pca = PCA(n_components=3)
    pca.fit(X)
    return pca


def apply_pca(feature_4d, pca, gamma=1/1.8, brightness=1.2):
    """
    feature_4d: [1,C,H,W]
    return: brightened PCA RGB image
    """
    feat = feature_4d[0].detach().cpu().numpy()   # [C,H,W]
    C, H, W = feat.shape

    X = feat.reshape(C, -1).T                     # [HW, C]
    X_pca = pca.transform(X)                      # [HW,3]

    # Normalize 0~1
    X_pca = (X_pca - X_pca.min()) / (X_pca.max() - X_pca.min() + 1e-6)

    X_pca = np.power(X_pca, gamma)

    X_pca = X_pca * brightness
    X_pca = np.clip(X_pca, 0, 1)

    X_pca = X_pca.reshape(H, W, 3)
    return X_pca

def visualize_sim(img, point, sim_upa, sim_bilinear):
    py, px = point  
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) Image with point
    axes[0].imshow(img)
    axes[0].scatter(px, py, marker='x', s=80, linewidths=2, color='red')
    axes[0].set_title(f"Image ({py}, {px})")
    axes[0].axis("off")

    # (2) UPA similarity
    im1 = axes[1].imshow(sim_upa, cmap="viridis")
    axes[1].set_title("Similarity (UPA)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # (3) Bilinear similarity
    im2 = axes[2].imshow(sim_bilinear, cmap="viridis")
    axes[2].set_title("Similarity (Bilinear)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def visualize_all(img, guided_modality, lr_feature, hr_rgb, hr_depth):
    """
    Show: RGB / Depth / LR PCA / HR_RGB PCA / HR_Depth PCA
    """
    pca = fit_pca_basis(lr_feature)

    lr_pca  = apply_pca(lr_feature, pca)
    rgb_pca = apply_pca(hr_rgb, pca)
    dep_pca = apply_pca(hr_depth, pca)


    plt.figure(figsize=(18, 8))

    # 1. Original RGB
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("RGB")
    plt.axis("off")

    # 2. Guided Modality (Depth)
    plt.subplot(2, 3, 2)
    plt.imshow(guided_modality)
    plt.title("Guided Modality (Depth)")
    plt.axis("off")

    # 3. LR Feature PCA
    plt.subplot(2, 3, 3)
    plt.imshow(lr_pca)
    plt.title("LR Feature (PCA)")
    plt.axis("off")

    # 4. HR (RGB guided)
    plt.subplot(2, 3, 4)
    plt.imshow(rgb_pca)
    plt.title("HR Feature (RGB Guided)")
    plt.axis("off")

    # 5. HR (Depth guided)
    plt.subplot(2, 3, 5)
    plt.imshow(dep_pca)
    plt.title("HR Feature (Depth Guided)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return lr_pca, rgb_pca, dep_pca

def cal_similarity(feat, img, point=(112,112)):
    """
    feat: [1, C, H, W] tensor
    img: PIL image (224,224)
    point: (y, x)
    """
    _, C, H, W = feat.shape
    py, px = point

    # point vector
    v = feat[0, :, py, px]  # [C]

    # normalize
    v_norm = v / (v.norm() + 1e-6)
    feat_norm = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)

    # cosine similarity
    sim = (feat_norm[0] * v_norm[:, None, None]).sum(dim=0).detach().cpu()

    # normalize to 0~1
    sim_np = sim.numpy()
    sim_norm = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)

    return sim_norm
