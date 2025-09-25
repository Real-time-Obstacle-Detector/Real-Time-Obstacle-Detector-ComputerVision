from pathlib import Path
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def make_valid_tmp_thirds(dataset_root, dry_run=False):
    """
    Create <dataset_root>/valid-tmp with every 1st image of each 3-image sequence from <dataset_root>/valid/images,
    plus its corresponding label from <dataset_root>/valid/labels.

    Pattern: copy #1, skip #2-#3, copy #4, skip #5-#6, ...

    Args:
        dataset_root (str | Path): Path to dataset root containing 'valid/images' and 'valid/labels'
        dry_run (bool): If True, only prints what would be copied.

    Returns:
        dict: summary counts
    """

    #Preprocessing urls and resolving their paths
    root = Path(dataset_root).resolve()
    src_img_dir = root / "valid" / "images"
    src_lbl_dir = root / "valid" / "labels"
    dst_img_dir = root / "valid-tmp" / "images"
    dst_lbl_dir = root / "valid-tmp" / "labels"

    if not src_img_dir.is_dir() or not src_lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected '{src_img_dir}' and '{src_lbl_dir}' to exist.")

    #Collect images (recursive), sorted deterministically
    imgs = sorted([p for p in src_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images found in {src_img_dir}")

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_no_label = 0
    skipped_pattern = 0

    for idx, img_path in enumerate(imgs, start=1):
        # Copy 1st of every 10 (1, 11, 21, ...)
        if (idx - 1) % 10 != 0:
            skipped_pattern += 1
            continue

        base = img_path.stem
        lbl_path = src_lbl_dir / f"{base}.txt"
        if not lbl_path.exists():
            skipped_no_label += 1
            print(f"[warn] No label for '{img_path.relative_to(src_img_dir)}' (expected {lbl_path.name}); skipping.")
            continue

        # Preserve subfolders from valid/images under valid-tmp/images
        rel_img_subpath = img_path.relative_to(src_img_dir)
        dst_img_path = dst_img_dir / rel_img_subpath
        dst_img_path.parent.mkdir(parents=True, exist_ok=True) if not dry_run else None

        dst_lbl_path = dst_lbl_dir / lbl_path.name  # labels commonly flat; adjust if yours are nested

        if not dry_run:
            shutil.copy2(img_path, dst_img_path)
            shutil.copy2(lbl_path, dst_lbl_path)

        copied += 1

        #Log our copy results aim to control the process
        print(f"[copy] {rel_img_subpath}-> valid-tmp/images/")
        print(f"[copy] {lbl_path.name} -> valid-tmp/labels/")

    summary = {
        "total_images_scanned": len(imgs),
        "copied": copied,
        "skipped_no_label": skipped_no_label,
        "skipped_pattern": skipped_pattern,
        "dst_images_dir": str(dst_img_dir),
        "dst_labels_dir": str(dst_lbl_dir),
    }

    print("\nDone.")
    for k, v in summary.items():
        print(f" {k}: {v}")

    return summary

make_valid_tmp_thirds(dataset_root = "C:/Users/abt/Documents/Real-time-obstacle-detector/data sets/dataset/dataset")