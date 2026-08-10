

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION — EDIT THESE PATHS
# ============================================================
CFG = dict(
    # Path to your trained model checkpoint (256×256, no crop)
    MODEL_PATH   = r"D:\VIT_PROJECT\unet\Models\Trial2_CSDA_NoCrop_256\csda_unet_best.pth",
    
    # Input directory containing 2D sagittal slices (224×224)
    INPUT_DIR    = r"E:\PROJECT_FORNIX\Dataset\Extracted_2D\output_slices",
    
    # Output directory for masks and overlays
    OUTPUT_DIR   = r"E:\PROJECT_FORNIX\MODELS\Final_2D_Unet\Out2",
    
    # Image settings (must match training)
    IMG_SIZE     = 256,        # Model was trained at 256×256
    USE_CROP     = False,
    CROP_SIZE    = 192,
    
    # Threshold for binary mask
    THRESHOLD    = 0.5,
    
    # Overlay opacity (0-1, higher = more opaque red)
    OVERLAY_ALPHA = 0.5,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL ARCHITECTURE — EXACT MATCH TO TRAINING
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.compress = nn.Conv2d(2, in_ch // 2, 1, bias=False)
        self.branch1  = nn.Conv2d(in_ch // 2, 1, 1, padding=0)
        self.branch3  = nn.Conv2d(in_ch // 2, 1, 3, padding=1)
        self.branch5  = nn.Conv2d(in_ch // 2, 1, 5, padding=2)
        self.branch7  = nn.Conv2d(in_ch // 2, 1, 7, padding=3)
        self.sigmoid  = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        desc  = self.compress(torch.cat([avg, mx], dim=1))
        attn  = self.sigmoid(
            self.branch1(desc) + self.branch3(desc) +
            self.branch5(desc) + self.branch7(desc)
        )
        return x * attn


class CSDA_UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_features=64, dropout=0.3):
        """EXACT architecture match to training."""
        super().__init__()
        f = base_features
        
        # Encoder
        self.enc1   = ConvBlock(in_channels, f, dropout=0.0)
        self.enc2   = ConvBlock(f, f*2, dropout=dropout*0.5)
        self.enc3   = ConvBlock(f*2, f*4, dropout=dropout*0.5)
        self.enc4   = ConvBlock(f*4, f*8, dropout=dropout)
        self.pool   = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bridge = ConvBlock(f*8, f*16, dropout=dropout)
        self.sa     = SpatialAttention(f*16)

        # Decoder
        self.up1  = nn.ConvTranspose2d(f*16, f*8, 2, 2)
        self.ca1  = ChannelAttention(f*8, reduction=8)
        self.dec1 = ConvBlock(f*16, f*8, dropout=dropout)

        self.up2  = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.ca2  = ChannelAttention(f*4, reduction=8)
        self.dec2 = ConvBlock(f*8, f*4, dropout=dropout*0.5)

        self.up3  = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.ca3  = ChannelAttention(f*2, reduction=8)
        self.dec3 = ConvBlock(f*4, f*2, dropout=dropout*0.5)

        self.up4  = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.ca4  = ChannelAttention(f, reduction=8)
        self.dec4 = ConvBlock(f*2, f, dropout=0.0)

        self.final = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.sa(self.bridge(self.pool(e4)))

        d1 = self.dec1(torch.cat([self.ca1(self.up1(b)),  e4], dim=1))
        d2 = self.dec2(torch.cat([self.ca2(self.up2(d1)), e3], dim=1))
        d3 = self.dec3(torch.cat([self.ca3(self.up3(d2)), e2], dim=1))
        d4 = self.dec4(torch.cat([self.ca4(self.up4(d3)), e1], dim=1))

        return self.final(d4)


# ============================================================
# HELPERS
# ============================================================
def preprocess_image(image_path, cfg):
    """Load and preprocess a single MRI slice. Returns tensor and original size."""
    img = Image.open(image_path).convert("L")
    original_size = img.size  # (W, H)
    
    if cfg.get("USE_CROP", False):
        w, h = img.size
        cs = cfg["CROP_SIZE"]
        l = (w - cs) // 2
        t = (h - cs) // 2
        img = img.crop((l, t, l + cs, t + cs))
    
    img = img.resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"]), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_arr = (img_arr - img_arr.min()) / (np.ptp(img_arr) + 1e-8)
    
    # Convert to tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).float()
    return img_tensor, original_size


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = CSDA_UNet(in_channels=1, num_classes=1, base_features=64, dropout=0.3).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Model] Loaded from epoch {ckpt.get('epoch', '?')} | Val IoU: {ckpt.get('val_iou', 'N/A')}")
    return model


def predict_slice(model, img_tensor, device, threshold=0.5):
    """Run inference on a single slice."""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred = model(img_tensor)
        prob = torch.sigmoid(pred).squeeze().cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)
    return mask, prob


def create_overlay(image, mask, alpha=0.5):
    """
    Create a smooth red overlay of mask on original grayscale image.
    No green outline — just a clean blended overlay.
    
    Args:
        image: 2D numpy array (original size)
        mask:  2D numpy array (same size as image, binary 0/1)
        alpha: Opacity of red overlay (0=invisible, 1=solid red)
    
    Returns:
        RGB image as uint8 array
    """
    # Normalize grayscale image to [0, 1]
    img_norm = (image - image.min()) / (np.ptp(image) + 1e-8)
    
    # Create RGB base (grayscale repeated 3 times)
    overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Blend red where mask=1
    # Red channel: keep some original + add red
    overlay[:, :, 0] = img_norm * (1 - mask * alpha) + mask * alpha * 1.0
    # Green channel: reduce where mask is active
    overlay[:, :, 1] = img_norm * (1 - mask * alpha)
    # Blue channel: reduce where mask is active
    overlay[:, :, 2] = img_norm * (1 - mask * alpha)
    
    # Clip to valid range
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)


def run_inference_2d(cfg):
    """Main inference function for 2D slices."""
    
    # Create output directories
    mask_dir = os.path.join(cfg["OUTPUT_DIR"], "masks")
    overlay_dir = os.path.join(cfg["OUTPUT_DIR"], "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(cfg["MODEL_PATH"], DEVICE)
    
    # Get input images
    input_dir = cfg["INPUT_DIR"]
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"[Error] No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Original → Model input: {cfg['IMG_SIZE']}×{cfg['IMG_SIZE']} → Output: original size")
    print(f"Overlay opacity: {cfg['OVERLAY_ALPHA']}")
    print(f"Processing...\n")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Segmenting"):
        img_path = os.path.join(input_dir, img_file)
        
        # Load original image
        original_img = Image.open(img_path).convert("L")
        original_arr = np.array(original_img, dtype=np.float32)
        original_size = original_img.size
        
        # Preprocess (resize to IMG_SIZE)
        img_tensor, _ = preprocess_image(img_path, cfg)
        
        # Predict (output is IMG_SIZE × IMG_SIZE)
        mask, prob = predict_slice(model, img_tensor, DEVICE, cfg["THRESHOLD"])
        
        # Resize mask back to original size
        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.uint8) // 255
        
        # Create overlay
        overlay_arr = create_overlay(original_arr, mask_arr, alpha=cfg["OVERLAY_ALPHA"])
        
        # Save outputs
        base_name = os.path.splitext(img_file)[0]
        
        # Save mask (binary, black & white)
        mask_save_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        mask_img.save(mask_save_path)
        
        # Save overlay (red blend on MRI)
        overlay_save_path = os.path.join(overlay_dir, f"{base_name}_overlay.png")
        overlay_img = Image.fromarray(overlay_arr)
        overlay_img.save(overlay_save_path)
    
    print(f"\n[Done] All outputs saved to: {cfg['OUTPUT_DIR']}")
    print(f"  - Masks:    {mask_dir}")
    print(f"  - Overlays: {overlay_dir}")


if __name__ == "__main__":
    run_inference_2d(CFG)
