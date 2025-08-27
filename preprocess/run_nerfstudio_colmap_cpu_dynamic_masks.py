#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import shutil
from typing import List, Set

# Ensure we can import sibling module when executed directly
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from crop_black_borders import (
    build_colmap_masks_from_seg,
    run_colmap,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def copy_or_symlink_images(src_dir: Path, dst_dir: Path, symlink: bool) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        dst = dst_dir / p.name
        if dst.exists():
            continue
        if symlink:
            try:
                os.symlink(p, dst)
            except FileExistsError:
                pass
        else:
            shutil.copy2(p, dst)
        count += 1
    return count


def parse_class_list(s: str) -> Set[int]:
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Copy/symlink frames, create dynamic masks from watershed masks with given static classes, and run COLMAP on CPU"
        )
    )
    ap.add_argument("--frames_dir", type=str, required=True, help="Directory with frames (PNG/JPG)")
    ap.add_argument(
        "--water_dir",
        type=str,
        required=True,
        help="Directory with watershed masks (color scheme mapping to classes). Filenames match frames or have suffix.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output root (creates images/ and colmap/ subfolders)",
    )
    ap.add_argument(
        "--static_classes",
        type=str,
        required=True,
        help="Comma-separated class IDs to KEEP as static (white). All others are masked (black).",
    )
    ap.add_argument(
        "--mask_suffix",
        type=str,
        default="watershed_mask",
        help="If watershed files are named <frame>_<suffix>.png (default: watershed_mask)",
    )
    ap.add_argument("--colmap_bin", type=str, default="colmap", help="Path to COLMAP binary")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    ap.add_argument("--use_xvfb", action="store_true", help="Wrap COLMAP in xvfb-run for headless")
    ap.add_argument("--gpu_sift", action="store_true", help="Use GPU SIFT in COLMAP")
    ap.add_argument("--gpu_match", action="store_true", help="Use GPU matcher in COLMAP")
    ap.add_argument("--gaussians_style", action="store_true", help="Use 4DGaussians-style layout/params")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    water_dir = Path(args.water_dir)
    out_dir = Path(args.out_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory not found: {frames_dir}")
    if not water_dir.is_dir():
        raise SystemExit(f"Watershed mask directory not found: {water_dir}")

    # Prepare outputs
    images_ns = out_dir / "images"
    images_colmap = out_dir / "colmap" / "images"
    logs_dir = out_dir / "colmap" / "Logs"
    masks_out = out_dir / "colmap" / "masks"

    n1 = copy_or_symlink_images(frames_dir, images_ns, args.symlink)
    n2 = copy_or_symlink_images(frames_dir, images_colmap, args.symlink)
    print(f"Prepared images: {n1} (images/), {n2} (colmap/images)")

    # Build binary masks: white for static classes, black for dynamic using watershed color-coded masks
    static_classes = parse_class_list(args.static_classes)
    written = build_colmap_masks_from_seg(
        images_colmap,
        water_dir,
        masks_out,
        classes_to_mask=None,
        classes_to_keep=static_classes,
        seg_suffix=args.mask_suffix,
    )
    print(f"Built {written} COLMAP masks in {masks_out}")

    # Run COLMAP with mask_dir
    run_colmap(
        args.colmap_bin,
        images_colmap,
        out_dir / "colmap",
        mask_dir=masks_out,
        log_dir=logs_dir,
        use_xvfb=args.use_xvfb,
        gpu_sift=args.gpu_sift,
        gpu_match=args.gpu_match,
        gaussians_style=args.gaussians_style,
    )
    print(f"Done. COLMAP outputs in: {out_dir / 'colmap'}")


if __name__ == "__main__":
    main()
