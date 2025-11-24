from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from upsample_anything import UPA
from utils import visualize_pca_one
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def dinov2_infer(img_path):
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    img_t = transform(img).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            feats_list = dinov2_vits14.get_intermediate_layers(img_t, n=1)
            feats_all = feats_list[0].squeeze(0)

    H=W=int(feats_all.shape[0]**0.5)
    feat_map=feats_all.reshape(H,W,-1).permute(2,0,1).unsqueeze(0)

    return feat_map


if __name__ == "__main__":
    img_path = "sample.png"
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    lr_feature = dinov2_infer(img_path)
    hr_feature = UPA(img, lr_feature)
    visualize_pca_one(hr_feature)
