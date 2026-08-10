
import os
import re
import shutil
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from skimage import measure
import SimpleITK as sitk
from tqdm import tqdm

from radiomics import featureextractor
import radiomics

# ============================================================================
# ======================  USER CONFIGURATION — EDIT THIS  ==================
# ============================================================================

# Folder containing input slice PNGs, e.g. 001__AD_slice5.png
INPUT_SLICES_DIR = r"F:\PROJECT_FORNIX\For_Paper\CSDA_U-NET\Unet_Input_Data\output_slices"

# Folder containing U-Net predicted mask PNGs, e.g. 001__AD_slice5_mask.png
MASKS_DIR = r"F:\PROJECT_FORNIX\For_Paper\CSDA_U-NET\Model_inference\masks - Copy"

# Where all outputs (cleaned masks, fragment copies, logs, CSVs) will be written
OUTPUT_DIR = r"F:\PROJECT_FORNIX\For_Paper\2D_Features_Final"

# Suffix pattern used in your mask filenames relative to the slice filename.
# e.g. slice "001__AD_slice5.png" -> mask "001__AD_slice5_mask.png"
MASK_SUFFIX = "_mask"

# Image file extension
IMG_EXT = ".png"

# Binarization threshold for masks (masks are usually 0/255, but predicted
# masks from a sigmoid U-Net may have soft values — anything > this is foreground)
MASK_BINARY_THRESHOLD = 127

# Minimum blob area (in pixels) to even be considered a valid candidate blob.
# Tiny 1-3 pixel noise specks below this are dropped automatically before
# the "largest blob" logic runs (does NOT affect fragment flagging count
# below MIN_BLOB_AREA_FOR_FRAGMENT_FLAG).
MIN_BLOB_AREA = 3

# Groups to KEEP for feature extraction / downstream stats.
# EMCI and LMCI are parsed if present in filenames but excluded from the
# final feature CSV, as requested.
KEEP_GROUPS = ["AD", "CN", "MCI"]
EXCLUDE_GROUPS = ["EMCI", "LMCI"]

# PyRadiomics settings
RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,   # keep native 2D pixel spacing
    "interpolator": "sitkBSpline",
    "verbose": False,
    "geometryTolerance": 1e-3,
    "force2D": True,                 # CRITICAL: these are 2D mid-sagittal slices
    "force2Ddimension": 0,
}

# Which image filters to enable (matches your pipeline diagram)
ENABLED_FILTERS = [
    "Original",
    "LoG",
    "Wavelet",
    "Square",
    "SquareRoot",
    "Logarithm",
    "Exponential",
    "Gradient",
    "LBP2D",
]

# Which feature classes to enable (matches your pipeline diagram)
ENABLED_FEATURE_CLASSES = [
    "firstorder",
    "shape2D",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
]

# ============================================================================
# ==========================  END CONFIGURATION  ============================
# ============================================================================


def setup_output_dirs(output_dir):
    """Create the output folder structure."""
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "cleaned_masks": output_dir / "cleaned_masks_largest_blob",
        "fragmented_masks": output_dir / "fragmented_masks_flagged",
        "matched_slices": output_dir / "matched_input_slices",  # copy for traceability
        "logs": output_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_dir):
    log_path = Path(log_dir) / "pipeline_step1.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("step1")


def parse_group_from_filename(filename):
    """
    Expected pattern: <id>__<GROUP>_sliceN[...]
    e.g. 001__AD_slice5.png -> AD
         014__EMCI_slice5.png -> EMCI
         007__LMCI_slice5_mask.png -> LMCI
    Falls back to a broader regex search over known group tokens if the
    strict double-underscore pattern doesn't match.
    """
    ALL_KNOWN_GROUPS = ["AD", "CN", "MCI", "EMCI", "LMCI"]

    stem = Path(filename).stem

    # Strict pattern: id__GROUP_slice...
    m = re.match(r"^[^_]+__([A-Za-z]+)_slice", stem)
    if m:
        token = m.group(1).upper()
        if token in ALL_KNOWN_GROUPS:
            return token

    # Fallback: search for any known group token as a whole word, longest first
    # (so EMCI/LMCI are matched before the substring MCI)
    for grp in sorted(ALL_KNOWN_GROUPS, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z]){grp}(?![A-Za-z])", stem, flags=re.IGNORECASE):
            return grp.upper()

    return None


def get_subject_id(filename, group):
    """
    Build a composite subject identity key.

    CRITICAL cohort convention (confirmed with the data owner):
      - The leading numeric id is a per-CLASS counter, not a global one.
        "001_AD", "001_CN", and "001_MCI" are three different people who
        happen to share the number 001 -- each class (AD/CN/MCI) numbers
        its own subjects starting over.
      - A "_New_" tagged file is a separate person added to the dataset
        later, who ALSO happens to reuse a number already used within
        that same class. "001_AD" and "001_New_AD" are two different
        people. "001_New_AD" and "001_New_CN" are two different people
        in two different classes.
      - Therefore the ONLY safe unique subject key is the combination of
        (numeric id, New-flag, group). Using the numeric id alone (or
        even numeric id + New-flag, without group) silently merges
        distinct people -- this was caught and fixed after discovering
        it corrupted ~40% of subject-level aggregation in an earlier run.

    Examples:
        001__AD_slice5.png,    group="AD"  -> "001_AD"
        001_New_CN_slice5.png, group="CN"  -> "001_New_CN"
        006_New_MCI_slice5.png,group="MCI" -> "006_New_MCI"
    """
    stem = Path(filename).stem
    m = re.match(r"^([A-Za-z0-9]+)_(New)_", stem)
    if m:
        num_id, new_flag = m.group(1), m.group(2)
        return f"{num_id}_{new_flag}_{group}"
    m = re.match(r"^([A-Za-z0-9]+)_", stem)
    num_id = m.group(1) if m else stem
    return f"{num_id}_{group}"


def strip_mask_suffix(mask_filename, mask_suffix):
    """Turn '001__AD_slice5_mask.png' -> '001__AD_slice5.png' (matching slice name)."""
    stem = Path(mask_filename).stem
    ext = Path(mask_filename).suffix
    if stem.endswith(mask_suffix):
        stem = stem[: -len(mask_suffix)]
    return stem + ext


def keep_largest_blob(mask_path, binary_threshold, min_blob_area):
    """
    Load a mask, find connected components, and return:
        cleaned_mask (uint8 0/255, only the largest blob kept)
        n_blobs_found (int, count of blobs >= min_blob_area BEFORE filtering)
        largest_blob_area (int)
        all_blob_areas (list[int])
    Returns None for cleaned_mask if no valid blob found at all.
    """
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    binary = (raw > binary_threshold).astype(np.uint8)

    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)

    # keep only blobs above the minimum noise-floor area
    valid_props = [p for p in props if p.area >= min_blob_area]
    all_blob_areas = sorted([p.area for p in valid_props], reverse=True)

    if len(valid_props) == 0:
        return None, 0, 0, []

    largest = max(valid_props, key=lambda p: p.area)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    cleaned[labeled == largest.label] = 255

    return cleaned, len(valid_props), int(largest.area), all_blob_areas


def run_mask_qc(input_slices_dir, masks_dir, dirs, mask_suffix, img_ext,
                 binary_threshold, min_blob_area, logger):
    """
    Stage 1: inspect every mask, keep largest blob only, flag fragmented
    (multi-blob) masks into a separate folder, and match to input slices.

    Returns a DataFrame with one row per mask processed, and the list of
    (input_slice_path, cleaned_mask_path, group, subject_id, base_name) tuples
    that are ready for radiomics extraction.
    """
    masks_dir = Path(masks_dir)
    input_slices_dir = Path(input_slices_dir)

    mask_files = sorted([f for f in masks_dir.iterdir()
                          if f.suffix.lower() == img_ext.lower()])

    if len(mask_files) == 0:
        raise FileNotFoundError(f"No '{img_ext}' mask files found in {masks_dir}")

    logger.info(f"Found {len(mask_files)} mask files in {masks_dir}")

    qc_rows = []
    ready_for_extraction = []

    for mask_path in tqdm(mask_files, desc="Mask QC"):
        matching_slice_name = strip_mask_suffix(mask_path.name, mask_suffix)
        matching_slice_path = input_slices_dir / matching_slice_name

        row = {
            "mask_filename": mask_path.name,
            "expected_slice_filename": matching_slice_name,
            "slice_found": matching_slice_path.exists(),
            "n_blobs_found": None,
            "largest_blob_area_px": None,
            "all_blob_areas": None,
            "status": None,
            "group": None,
            "subject_id": None,
        }

        try:
            cleaned, n_blobs, largest_area, all_areas = keep_largest_blob(
                mask_path, binary_threshold, min_blob_area
            )
        except Exception as e:
            row["status"] = f"ERROR: {e}"
            logger.error(f"{mask_path.name}: {e}")
            qc_rows.append(row)
            continue

        row["n_blobs_found"] = n_blobs
        row["largest_blob_area_px"] = largest_area
        row["all_blob_areas"] = str(all_areas)

        if cleaned is None or n_blobs == 0:
            row["status"] = "EMPTY_MASK_NO_BLOB"
            logger.warning(f"{mask_path.name}: no valid blob found (empty mask)")
            # still copy original to fragmented folder for manual review
            shutil.copy2(mask_path, dirs["fragmented_masks"] / mask_path.name)
            qc_rows.append(row)
            continue

        # Save cleaned (largest-blob-only) mask regardless, so you have it
        # available even for fragmented cases if you want to inspect later.
        cleaned_path = dirs["cleaned_masks"] / mask_path.name
        cv2.imwrite(str(cleaned_path), cleaned)

        if n_blobs > 1:
            row["status"] = f"FRAGMENTED ({n_blobs} blobs) - kept largest, flagged"
            logger.warning(
                f"{mask_path.name}: {n_blobs} blobs found (areas={all_areas}); "
                f"kept largest ({largest_area}px), copied original to fragmented_masks/"
            )
            shutil.copy2(mask_path, dirs["fragmented_masks"] / mask_path.name)
        else:
            row["status"] = "OK_SINGLE_BLOB"

        if not matching_slice_path.exists():
            row["status"] += " | INPUT_SLICE_MISSING"
            logger.error(f"{mask_path.name}: no matching input slice at {matching_slice_path}")
            qc_rows.append(row)
            continue

        group = parse_group_from_filename(matching_slice_name)
        subject_id = get_subject_id(matching_slice_name, group)
        row["group"] = group
        row["subject_id"] = subject_id

        if group is None:
            row["status"] += " | GROUP_UNPARSED"
            logger.error(f"{matching_slice_name}: could not parse group label from filename")
        elif group in EXCLUDE_GROUPS:
            row["status"] += f" | EXCLUDED_GROUP({group})"
        elif group not in KEEP_GROUPS:
            row["status"] += f" | UNKNOWN_GROUP({group})"
        else:
            # ready for radiomics extraction
            base_name = Path(matching_slice_name).stem
            shutil.copy2(matching_slice_path, dirs["matched_slices"] / matching_slice_name)
            ready_for_extraction.append({
                "base_name": base_name,
                "slice_path": str(matching_slice_path),
                "mask_path": str(cleaned_path),
                "group": group,
                "subject_id": subject_id,
                "n_blobs_original": n_blobs,
            })

        qc_rows.append(row)

    qc_df = pd.DataFrame(qc_rows)
    return qc_df, ready_for_extraction


def build_extractor():
    """Configure the PyRadiomics feature extractor per the pipeline spec."""
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)

    # Disable everything first, then enable exactly what we want
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()

    for fc in ENABLED_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(fc)

    for filt in ENABLED_FILTERS:
        if filt == "Original":
            extractor.enableImageTypeByName("Original")
        elif filt == "LoG":
            # sigma values in mm; adjust if your pixel spacing differs
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
        elif filt == "Wavelet":
            extractor.enableImageTypeByName("Wavelet")
        elif filt == "Square":
            extractor.enableImageTypeByName("Square")
        elif filt == "SquareRoot":
            extractor.enableImageTypeByName("SquareRoot")
        elif filt == "Logarithm":
            extractor.enableImageTypeByName("Logarithm")
        elif filt == "Exponential":
            extractor.enableImageTypeByName("Exponential")
        elif filt == "Gradient":
            extractor.enableImageTypeByName("Gradient")
        elif filt == "LBP2D":
            extractor.enableImageTypeByName("LBP2D")

    return extractor


def png_to_sitk_image(png_path):
    """Read a grayscale PNG and return a SimpleITK 2D image (as a pseudo-3D
    single-slice volume, since PyRadiomics' force2D expects a 3D-shaped image
    with a singleton axis on the force2Ddimension)."""
    arr = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Could not read image: {png_path}")
    arr3d = arr[np.newaxis, :, :].astype(np.float32)  # shape (1, H, W)
    img = sitk.GetImageFromArray(arr3d)
    return img


def png_mask_to_sitk_image(png_path, binary_threshold):
    arr = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Could not read mask image: {png_path}")
    binary = (arr > binary_threshold).astype(np.uint8)
    binary3d = binary[np.newaxis, :, :]
    mask_img = sitk.GetImageFromArray(binary3d)
    return mask_img


def extract_features_for_all(ready_list, extractor, logger):
    """Run PyRadiomics on every (slice, cleaned mask) pair. Returns a DataFrame."""
    all_rows = []
    failed = []

    for item in tqdm(ready_list, desc="Radiomics extraction"):
        try:
            img = png_to_sitk_image(item["slice_path"])
            mask = png_mask_to_sitk_image(item["mask_path"], MASK_BINARY_THRESHOLD)

            # sanity check: mask must have at least a few foreground voxels
            mask_arr = sitk.GetArrayFromImage(mask)
            if mask_arr.sum() < MIN_BLOB_AREA:
                raise ValueError("Cleaned mask has insufficient foreground pixels")

            result = extractor.execute(img, mask)

            row = {
                "base_name": item["base_name"],
                "subject_id": item["subject_id"],
                "group": item["group"],
                "n_blobs_original": item["n_blobs_original"],
                "slice_path": item["slice_path"],
                "mask_path": item["mask_path"],
            }
            # keep only actual feature keys (skip diagnostics_* metadata? we keep
            # them too, prefixed, in case useful for QC; they won't hurt stats
            # since script 2 will separate metadata from numeric features)
            for k, v in result.items():
                row[k] = v

            all_rows.append(row)

        except Exception as e:
            logger.error(f"FAILED extraction for {item['base_name']}: {e}")
            logger.error(traceback.format_exc())
            failed.append({"base_name": item["base_name"], "error": str(e)})

    df = pd.DataFrame(all_rows)
    failed_df = pd.DataFrame(failed)
    return df, failed_df


def main():
    dirs = setup_output_dirs(OUTPUT_DIR)
    logger = setup_logging(dirs["logs"])

    logger.info("=" * 70)
    logger.info("STEP 1: Mask QC + PyRadiomics Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"PyRadiomics version: {radiomics.__version__}")
    logger.info(f"Input slices dir: {INPUT_SLICES_DIR}")
    logger.info(f"Masks dir: {MASKS_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # ---- Stage A: Mask QC (largest blob keep, fragment flagging) ----
    qc_df, ready_list = run_mask_qc(
        INPUT_SLICES_DIR, MASKS_DIR, dirs, MASK_SUFFIX, IMG_EXT,
        MASK_BINARY_THRESHOLD, MIN_BLOB_AREA, logger,
    )
    qc_csv_path = dirs["logs"] / "mask_qc_report.csv"
    qc_df.to_csv(qc_csv_path, index=False)
    logger.info(f"Saved mask QC report -> {qc_csv_path}")
    logger.info(f"Total masks processed: {len(qc_df)}")
    logger.info(f"Fragmented (multi-blob) masks flagged: "
                f"{(qc_df['n_blobs_found'].fillna(0) > 1).sum()}")
    logger.info(f"Empty masks (no blob): "
                f"{(qc_df['status'] == 'EMPTY_MASK_NO_BLOB').sum()}")
    logger.info(f"Ready for radiomics extraction (AD/CN/MCI matched): {len(ready_list)}")

    if len(ready_list) == 0:
        logger.error("No slice/mask pairs ready for extraction. Check paths, "
                      "MASK_SUFFIX, and filename group parsing. Exiting.")
        return

    group_counts = pd.Series([r["group"] for r in ready_list]).value_counts()
    logger.info(f"Group counts going into extraction:\n{group_counts}")

    # ---- Stage B: PyRadiomics extraction ----
    extractor = build_extractor()
    logger.info(f"Enabled image filters: {ENABLED_FILTERS}")
    logger.info(f"Enabled feature classes: {ENABLED_FEATURE_CLASSES}")

    features_df, failed_df = extract_features_for_all(ready_list, extractor, logger)

    features_csv_path = dirs["root"] / "raw_radiomics_features.csv"
    features_df.to_csv(features_csv_path, index=False)
    logger.info(f"Saved raw radiomics feature CSV -> {features_csv_path}")
    logger.info(f"Shape: {features_df.shape[0]} samples x {features_df.shape[1]} columns")

    if len(failed_df) > 0:
        failed_csv_path = dirs["logs"] / "failed_extractions.csv"
        failed_df.to_csv(failed_csv_path, index=False)
        logger.warning(f"{len(failed_df)} extractions FAILED. See {failed_csv_path}")

    logger.info("STEP 1 COMPLETE.")
    logger.info(f"Next: run 02_statistical_analysis.py with "
                f"INPUT_CSV = '{features_csv_path}'")


if __name__ == "__main__":
    main()
