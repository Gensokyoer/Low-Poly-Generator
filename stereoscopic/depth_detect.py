import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

def get_depth(img):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitb'  # or 'vits', 'vitb', 'vitg'

    CHECKPOINT_PATH = "D:/pylib/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()

    depth = model.infer_image(img)

    # 归一化深度图以便可视化
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # 应用颜色映射（JET颜色映射常用于深度图）
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_colormap, depth

