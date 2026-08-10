



import os, random, json, time, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import (binary_fill_holes, binary_closing,
                            binary_opening, label as nd_label)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================
# CONFIGURATION
# ============================================================
CFG = dict(
    IMAGE_DIR   = r"D:\VIT_PROJECT\unet\Extracted_2D _new2\output_slices",
    MASK_DIR    = r"D:\VIT_PROJECT\unet\Extracted_2D _new2\output_masks",
    OUTPUT_DIR  = r"D:\VIT_PROJECT\unet\Models\TrialFinal_CSDA_256",

    # ── IMAGE SETTINGS ──────────────────────────
    # Set IMG_SIZE to 256 or 512
    IMG_SIZE    = 256,           # 256 or 512
    
    # Set USE_CROP = True to center-crop before resize
    # Set USE_CROP = False to use full image (no crop)
    USE_CROP    = False,          # Toggle: True = crop, False = no crop
    
    # CROP_SIZE is only used when USE_CROP = True
    CROP_SIZE   = 192,           # Ignored if USE_CROP = False
    
    NUM_CLASSES = 1,

    # ── TRAINING ───────────────────────────────
    BATCH_SIZE  = 4,
    EPOCHS      = 200,
    LR          = 1e-4,
    WEIGHT_DECAY= 1e-5,
    VAL_SPLIT   = 0.15,
    TEST_SPLIT  = 0.10,
    PATIENCE    = 20,
    MIN_DELTA   = 1e-4,
    NUM_WORKERS = 0,
    SEED        = 27,
    AMP         = True,

    # ── MASK CLEANING ──────────────────────────
    CLEAN_MIN_FRAC      = 0.10,
    CLEAN_CLOSE_RADIUS  = 2,
    CLEAN_OPEN_RADIUS   = 0.5,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["SEED"])

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# ============================================================
# MASK CLEANING
# ============================================================
def make_disk(radius):
    """Create a circular structuring element of given radius."""
    r = int(radius)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y <= r*r).astype(bool)


def clean_mask(mask_arr: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Clean a binary mask (uint8, values 0/1) so that:
      1. Tiny noise blobs are removed (opening).
      2. Internal holes are filled.
      3. Remaining small disconnected fragments are dropped
         (keep only components >= CLEAN_MIN_FRAC * largest component).
      4. Edges are smoothed (closing).

    Returns cleaned uint8 mask with values 0/1.
    """
    if mask_arr.max() == 0:
        return mask_arr   # empty mask — nothing to do

    binary = mask_arr.astype(bool)

    # Step 1 — morphological opening (remove tiny isolated specks)
    r_open = cfg.get("CLEAN_OPEN_RADIUS", 1)
    if r_open > 0:
        struct = make_disk(r_open)
        binary = ndimage.binary_opening(binary, structure=struct, iterations=1)

    # Step 2 — fill internal holes
    binary = binary_fill_holes(binary)

    # Step 3 — keep only large connected components
    labeled, n_comp = nd_label(binary)
    if n_comp > 1:
        sizes = np.array([
            (labeled == i).sum() for i in range(1, n_comp + 1)
        ])
        max_size  = sizes.max()
        threshold = cfg.get("CLEAN_MIN_FRAC", 0.10) * max_size
        kept = np.zeros_like(binary)
        for i, sz in enumerate(sizes, start=1):
            if sz >= threshold:
                kept |= (labeled == i)
        binary = kept.astype(bool)

    # Step 4 — morphological closing (smooth edges)
    r_close = cfg.get("CLEAN_CLOSE_RADIUS", 3)
    if r_close > 0:
        struct = make_disk(r_close)
        binary = binary_closing(binary, structure=struct, iterations=1)

    # Step 5 — fill holes again (closing can reopen tiny ones)
    binary = binary_fill_holes(binary)

    return binary.astype(np.uint8)


# ============================================================
# HELPERS
# ============================================================
def center_crop_pil(img, crop_size):
    """Center crop a PIL image to crop_size × crop_size."""
    w, h = img.size
    l = (w - crop_size) // 2
    t = (h - crop_size) // 2
    return img.crop((l, t, l + crop_size, t + crop_size))


def process_image(img, cfg):
    """
    Process image based on USE_CROP and IMG_SIZE settings.
    If USE_CROP=True: center crop to CROP_SIZE, then resize to IMG_SIZE.
    If USE_CROP=False: resize directly to IMG_SIZE.
    """
    if cfg.get("USE_CROP", True):
        img = center_crop_pil(img, cfg["CROP_SIZE"])
    img = img.resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"]), Image.BILINEAR)
    return img


def process_mask(mask, cfg):
    """
    Process mask based on USE_CROP and IMG_SIZE settings.
    Uses NEAREST interpolation to preserve binary values.
    """
    if cfg.get("USE_CROP", True):
        mask = center_crop_pil(mask, cfg["CROP_SIZE"])
    mask = mask.resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"]), Image.NEAREST)
    return mask


def get_pairing_key(filename):
    stem = os.path.splitext(filename)[0]
    if '_slice' in stem:
        stem = stem.replace('_slice', '_#')
    elif '_mask' in stem:
        stem = stem.replace('_mask', '_#')
    else:
        stem = stem.replace('slice', '#').replace('mask', '#')
    return stem

get_subject_id = get_pairing_key

# ============================================================
# AUGMENTATION
# ============================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.9, 1.1), rotate=(-15, 15), mode=0, p=0.7),
        A.ElasticTransform(alpha=30, sigma=5, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16),
                        hole_width_range=(8, 16), fill=0, p=0.2),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()])

# ============================================================
# DATASET (mask cleaning applied in _load_mask)
# ============================================================
class FornixDataset(Dataset):
    def __init__(self, image_dir, mask_dir, subject_ids,
                 image_map, mask_map, transform, cfg):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.subject_ids = subject_ids
        self.image_map   = image_map
        self.mask_map    = mask_map
        self.transform   = transform
        self.cfg         = cfg

    def __len__(self):
        return len(self.subject_ids)

    def _load_image(self, path):
        img = Image.open(path).convert("L")
        img = process_image(img, self.cfg)
        return np.array(img, dtype=np.float32)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = process_mask(mask, self.cfg)
        mask_arr = np.array(mask, dtype=np.uint8)

        # Binarise
        mask_arr = (mask_arr > 0).astype(np.uint8)

        # ── CLEAN THE MASK ──────────────────────────────────
        mask_arr = clean_mask(mask_arr, self.cfg)
        # ────────────────────────────────────────────────────

        return mask_arr

    def __getitem__(self, idx):
        sid  = self.subject_ids[idx]
        img  = self._load_image(os.path.join(self.image_dir, self.image_map[sid]))
        mask = self._load_mask(os.path.join(self.mask_dir,  self.mask_map[sid]))

        aug     = self.transform(image=img, mask=mask)
        image_t = aug["image"].float()
        mask_t  = aug["mask"].float().unsqueeze(0)   # [1, H, W]
        return image_t, mask_t


def build_datasets(cfg):
    image_dir, mask_dir = cfg["IMAGE_DIR"], cfg["MASK_DIR"]
    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    masks  = sorted([f for f in os.listdir(mask_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    image_map = {get_subject_id(f): f for f in images}
    mask_map  = {get_subject_id(f): f for f in masks}
    matched   = sorted(set(image_map) & set(mask_map))
    print(f"[Dataset] Matched pairs: {len(matched)}")

    rng = random.Random(cfg["SEED"])
    rng.shuffle(matched)
    n_test = max(1, int(len(matched) * cfg["TEST_SPLIT"]))
    n_val  = max(1, int(len(matched) * cfg["VAL_SPLIT"]))
    test_ids  = matched[:n_test]
    val_ids   = matched[n_test:n_test + n_val]
    train_ids = matched[n_test + n_val:]
    print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    common = dict(image_dir=image_dir, mask_dir=mask_dir,
                  image_map=image_map, mask_map=mask_map, cfg=cfg)
    train_ds = FornixDataset(subject_ids=train_ids,
                             transform=get_train_transforms(cfg["IMG_SIZE"]), **common)
    val_ds   = FornixDataset(subject_ids=val_ids,
                             transform=get_val_transforms(), **common)
    test_ds  = FornixDataset(subject_ids=test_ids,
                             transform=get_val_transforms(), **common)
    return train_ds, val_ds, test_ds


# ============================================================
# MASK CLEANING PREVIEW (call once before training to verify)
# ============================================================
def preview_mask_cleaning(cfg, n_samples=6):
    """
    Saves a side-by-side comparison of raw vs cleaned masks
    so you can visually verify the cleaning parameters.
    """
    mask_dir = cfg["MASK_DIR"]
    out_dir  = cfg["OUTPUT_DIR"]
    make_dir(out_dir)

    mask_files = sorted([f for f in os.listdir(mask_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not mask_files:
        print("[Preview] No mask files found — skipping preview.")
        return

    samples = random.sample(mask_files, min(n_samples, len(mask_files)))
    fig, axes = plt.subplots(len(samples), 2,
                             figsize=(6, 3 * len(samples)))
    if len(samples) == 1:
        axes = np.array([axes])

    axes[0, 0].set_title("Original Mask", fontweight="bold", fontsize=11)
    axes[0, 1].set_title("Cleaned Mask",  fontweight="bold", fontsize=11)

    for i, fname in enumerate(samples):
        mask = Image.open(os.path.join(mask_dir, fname)).convert("L")
        mask = process_mask(mask, cfg)
        raw  = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        cln  = clean_mask(raw.copy(), cfg)

        axes[i, 0].imshow(raw, cmap="gray"); axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(fname[:20], fontsize=7)
        axes[i, 1].imshow(cln, cmap="gray"); axes[i, 1].axis("off")

        # Show pixel count change
        diff = int(raw.sum()) - int(cln.sum())
        axes[i, 1].set_xlabel(
            f"raw={raw.sum()} px  cleaned={cln.sum()} px  Δ={diff:+d}",
            fontsize=7
        )

    plt.suptitle("Mask Cleaning Preview\n"
                 f"open_r={cfg['CLEAN_OPEN_RADIUS']}  "
                 f"close_r={cfg['CLEAN_CLOSE_RADIUS']}  "
                 f"min_frac={cfg['CLEAN_MIN_FRAC']}\n"
                 f"IMG_SIZE={cfg['IMG_SIZE']}  USE_CROP={cfg['USE_CROP']}",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(out_dir, "mask_cleaning_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Preview] Mask cleaning preview saved → {save_path}")
    print("  Review it before starting training.")
    print("  Adjust CLEAN_OPEN_RADIUS / CLEAN_CLOSE_RADIUS / CLEAN_MIN_FRAC in CFG if needed.")
    print("  Training will start in 3 seconds...\n")
    time.sleep(3)  # Brief pause to read, then auto-continue


# ============================================================
# MODEL — CSDA‑UNet (unchanged architecture)
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

    def forward(self, x):
        return self.block(x)


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

    def forward(self, x):
        return x * self.fc(x)


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
        super().__init__()
        f = base_features
        self.enc1   = ConvBlock(in_channels, f)
        self.enc2   = ConvBlock(f,    f*2,  dropout=dropout*0.5)
        self.enc3   = ConvBlock(f*2,  f*4,  dropout=dropout*0.5)
        self.enc4   = ConvBlock(f*4,  f*8,  dropout=dropout)
        self.pool   = nn.MaxPool2d(2)
        self.bridge = ConvBlock(f*8, f*16,  dropout=dropout)
        self.sa     = SpatialAttention(f*16)

        self.up1  = nn.ConvTranspose2d(f*16, f*8, 2, 2)
        self.ca1  = ChannelAttention(f*8,  reduction=8)
        self.dec1 = ConvBlock(f*16, f*8,  dropout=dropout)

        self.up2  = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.ca2  = ChannelAttention(f*4,  reduction=8)
        self.dec2 = ConvBlock(f*8,  f*4,  dropout=dropout*0.5)

        self.up3  = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.ca3  = ChannelAttention(f*2,  reduction=8)
        self.dec3 = ConvBlock(f*4,  f*2,  dropout=dropout*0.5)

        self.up4  = nn.ConvTranspose2d(f*2, f,   2, 2)
        self.ca4  = ChannelAttention(f,    reduction=8)
        self.dec4 = ConvBlock(f*2,  f)

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
# LOSSES
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred  = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (denom + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce   = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.5, focal_w=0.5):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss()
        self.dw    = dice_w
        self.fw    = focal_w

    def forward(self, pred, target):
        return self.dw * self.dice(pred, target) + self.fw * self.focal(pred, target)


# ============================================================
# METRICS
# ============================================================
@torch.no_grad()
def compute_metrics(pred, target):
    p     = (torch.sigmoid(pred) > 0.5).float()
    t     = target.float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    iou   = (inter + 1e-6) / (union + 1e-6)
    dice  = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    tp    = inter
    fp    = p.sum() - inter
    fn    = t.sum() - inter
    prec  = tp / (tp + fp + 1e-6)
    rec   = tp / (tp + fn + 1e-6)
    return {
        "iou":       iou.item(),
        "dice":      dice.item(),
        "precision": prec.item(),
        "recall":    rec.item(),
    }


# ============================================================
# FULL EVALUATION
# ============================================================
def evaluate_full(model, loader, cfg, save_dir, stage="Test"):
    model.eval()
    all_probs, all_labels = [], []
    total_tp = total_fp = total_fn = total_tn = 0
    iou_sum = dice_sum = prec_sum = rec_sum = 0.0
    sample_images, sample_gts, sample_preds = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"[{stage}] Eval"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            m     = compute_metrics(preds, masks)
            n     = imgs.size(0)
            iou_sum  += m["iou"]  * n
            dice_sum += m["dice"] * n
            prec_sum += m["precision"] * n
            rec_sum  += m["recall"]    * n

            prob     = torch.sigmoid(preds)
            pred_bin = (prob > 0.5).float()
            total_tp += (pred_bin * masks).sum().item()
            total_fp += (pred_bin * (1 - masks)).sum().item()
            total_fn += ((1 - pred_bin) * masks).sum().item()
            total_tn += ((1 - pred_bin) * (1 - masks)).sum().item()

            all_probs.append(prob.cpu().numpy().ravel())
            all_labels.append(masks.cpu().numpy().ravel())

            if len(sample_images) < 8:
                for b in range(n):
                    if len(sample_images) >= 8:
                        break
                    sample_images.append(imgs[b].cpu())
                    sample_gts.append(masks[b].cpu())
                    sample_preds.append(prob[b].cpu())

    n_total  = len(loader.dataset)
    avg_iou  = iou_sum  / n_total
    avg_dice = dice_sum / n_total
    avg_prec = prec_sum / n_total
    avg_rec  = rec_sum  / n_total

    print(f"\n[{stage}] IoU={avg_iou:.4f}  Dice={avg_dice:.4f}  "
          f"Prec={avg_prec:.4f}  Recall={avg_rec:.4f}")
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    print(f"  Pixel CM — TN={total_tn:.0f}  FP={total_fp:.0f}  "
          f"FN={total_fn:.0f}  TP={total_tp:.0f}")

    # AUC-ROC
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = None
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"  Pixel-wise AUC-ROC: {auc:.4f}")
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{stage} ROC Curve", fontweight="bold")
        plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage.lower()}_roc.png"), dpi=150)
        plt.close()

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"{stage} Confusion Matrix", fontweight="bold")
    plt.colorbar(shrink=0.8)
    plt.xticks([0, 1], ["Background", "Foreground"])
    plt.yticks([0, 1], ["Background", "Foreground"])
    plt.ylabel("True"); plt.xlabel("Predicted")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]:,.0f}", ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black",
                     fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{stage.lower()}_confusion.png"), dpi=150)
    plt.close()

    # Error overlay
    n_overlay = min(len(sample_images), 4)
    if n_overlay > 0:
        from matplotlib.colors import ListedColormap
        fig, axes = plt.subplots(n_overlay, 5, figsize=(18, 3.5 * n_overlay))
        if n_overlay == 1:
            axes = np.array([axes])
        col_names = ["MRI Image", "Ground Truth", "Prediction",
                     "Error Map\n(Red=FP, Blue=FN)", "Overlay"]
        for i in range(n_overlay):
            img     = sample_images[i].squeeze().numpy()
            gt      = sample_gts[i].squeeze().numpy()
            pr_prob = sample_preds[i].squeeze().numpy()
            pr      = (pr_prob > 0.5).astype(np.uint8)
            err     = np.zeros_like(pr, dtype=np.uint8)
            err[(gt == 1) & (pr == 0)] = 2
            err[(gt == 0) & (pr == 1)] = 1
            img_disp = (img - img.min()) / (np.ptp(img) + 1e-8)
            axes[i, 0].imshow(img_disp, cmap="gray"); axes[i, 0].axis("off")
            axes[i, 1].imshow(gt,       cmap="gray"); axes[i, 1].axis("off")
            axes[i, 2].imshow(pr,       cmap="gray"); axes[i, 2].axis("off")
            axes[i, 3].imshow(err, cmap=ListedColormap(["black", "red", "blue"]),
                              vmin=0, vmax=2); axes[i, 3].axis("off")
            axes[i, 4].imshow(img_disp, cmap="gray")
            if pr.sum() > 0:
                boundary = pr - ndimage.binary_erosion(pr).astype(np.uint8)
                axes[i, 4].contour(boundary, levels=[0.5],
                                   colors="lime", linewidths=1.5)
            axes[i, 4].axis("off")
            if i == 0:
                for j, name in enumerate(col_names):
                    axes[i, j].set_title(name, fontsize=10, fontweight="bold")
        plt.suptitle(f"{stage} — Prediction Error Analysis",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage.lower()}_error_overlay.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Save JSON
    results = {
        "stage": stage, "iou": float(avg_iou), "dice": float(avg_dice),
        "precision": float(avg_prec), "recall": float(avg_rec),
        "auc": float(auc) if auc is not None else None,
        "confusion_matrix": {
            "tn": int(total_tn), "fp": int(total_fp),
            "fn": int(total_fn), "tp": int(total_tp)
        },
    }
    with open(os.path.join(save_dir, f"{stage.lower()}_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → All {stage} outputs saved to: {save_dir}")
    return avg_iou, avg_dice, auc


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ============================================================
# TRAINING LOOP
# ============================================================
def train(cfg=CFG):
    make_dir(cfg["OUTPUT_DIR"])

    # Print configuration summary
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  IMG_SIZE:  {cfg['IMG_SIZE']}×{cfg['IMG_SIZE']}")
    print(f"  USE_CROP:  {cfg['USE_CROP']}")
    if cfg['USE_CROP']:
        print(f"  CROP_SIZE: {cfg['CROP_SIZE']}×{cfg['CROP_SIZE']}")
    else:
        print(f"  CROP_SIZE: Disabled (using full image)")
    print(f"  BATCH_SIZE: {cfg['BATCH_SIZE']}")
    print(f"  EPOCHS:    {cfg['EPOCHS']}")
    print(f"  LR:        {cfg['LR']}")
    print("=" * 60 + "\n")

    # Show cleaning preview BEFORE training (auto-continues after 3 seconds)
    print("[Step 0] Generating mask cleaning preview...")
    preview_mask_cleaning(cfg, n_samples=6)
    # Training starts automatically after preview (time.sleep(3) in preview function)

    train_ds, val_ds, test_ds = build_datasets(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                              shuffle=True,  num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["BATCH_SIZE"],
                              shuffle=False, num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["BATCH_SIZE"],
                              shuffle=False, num_workers=cfg["NUM_WORKERS"])

    model = CSDA_UNet(in_channels=1, num_classes=cfg["NUM_CLASSES"],
                      base_features=64, dropout=0.3).to(DEVICE)
    print(f"[Model] Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["EPOCHS"], eta_min=1e-7)
    criterion = CombinedLoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg["AMP"])
    stopper   = EarlyStopping(patience=cfg["PATIENCE"], min_delta=cfg["MIN_DELTA"])

    history   = {"train_loss": [], "val_iou": [], "val_dice": [], "lr": []}
    best_iou  = 0.0
    best_path = os.path.join(cfg["OUTPUT_DIR"], "csda_unet_best.pth")

    print(f"\n[Training] Device: {DEVICE} | Epochs: {cfg['EPOCHS']}\n")

    for epoch in range(1, cfg["EPOCHS"] + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{cfg['EPOCHS']} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with autocast(enabled=cfg["AMP"]):
                preds = model(imgs)
                loss  = criterion(preds, masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        # ── Validate ──
        model.eval()
        iou_sum = dice_sum = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader,
                                    desc=f"Epoch {epoch}/{cfg['EPOCHS']} [Val]",
                                    leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast(enabled=cfg["AMP"]):
                    preds = model(imgs)
                m         = compute_metrics(preds, masks)
                iou_sum  += m["iou"]  * imgs.size(0)
                dice_sum += m["dice"] * imgs.size(0)

        val_iou  = iou_sum  / len(val_ds)
        val_dice = dice_sum / len(val_ds)
        cur_lr   = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
              f"Val IoU {val_iou:.4f} | Val Dice {val_dice:.4f} | LR {cur_lr:.2e}")

        if val_iou > best_iou + cfg["MIN_DELTA"]:
            best_iou = val_iou
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_iou":     val_iou,
                "val_dice":    val_dice,
                "cfg":         cfg,
            }, best_path)
            print(f"  ✔ Best model saved (IoU={best_iou:.4f})")

        if stopper.step(val_iou):
            print(f"  ⚑ Early stopping at epoch {epoch}")
            break

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], color="red")
    axes[0].set_title("Train Loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_iou"],  label="IoU",  color="green")
    axes[1].plot(history["val_dice"], label="Dice", color="blue")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUTPUT_DIR"], "training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(cfg["OUTPUT_DIR"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ──
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    evaluate_full(model, test_loader, cfg,
                  save_dir=cfg["OUTPUT_DIR"], stage="Test")

    print(f"\n[Done]  Best Val IoU: {best_iou:.4f}")
    print(f"        All outputs:  {cfg['OUTPUT_DIR']}")


if __name__ == "__main__":
    train(CFG)
