#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
from typing import List

import cv2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Note: Removed COLMAP-related color mappings and mask helpers


def list_images(folder: Path) -> List[Path]:
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def crop_image(img, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int):
    h, w = img.shape[:2]
    if w <= (crop_left + crop_right):
        raise ValueError(
            f"Image width {w} too small for crop L{crop_left}+R{crop_right}"
        )
    if h <= (crop_top + crop_bottom):
        raise ValueError(
            f"Image height {h} too small for crop top {crop_top} + bottom {crop_bottom}"
        )
    return img[crop_top : h - crop_bottom, crop_left : w - crop_right]


# Note: Removed COLMAP helpers and execution logic


def main():
    ap = argparse.ArgumentParser(
        description="Crop images and corresponding masks (no COLMAP)"
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing frames and masks",
    )
    ap.add_argument(
        "--crop_left", type=int, default=110, help="Pixels to crop from the left"
    )
    ap.add_argument(
        "--crop_right", type=int, default=165, help="Pixels to crop from the right"
    )
    ap.add_argument(
        "--crop_top", type=int, default=50, help="Pixels to crop from the top"
    )
    ap.add_argument(
        "--crop_bottom", type=int, default=6, help="Pixels to crop from the bottom"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite output files if they exist"
    )
    # Note: Removed COLMAP-related arguments
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    frames_out = in_dir / "frames_cropped"
    masks_out = in_dir / "watershed_masks_cropped"
    seg_out = in_dir / "segmentation_masks_cropped"
    frames_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    seg_out.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir)
    if not images:
        print(f"No images found in {in_dir}")
        return

    print(f"Found {len(images)} image files in {in_dir}")
    print(f"Writing cropped frames to {frames_out}")
    print(f"Writing cropped watershed masks to {masks_out}")
    print(f"Writing cropped segmentation masks to {seg_out}")

    filenames = [i for i in in_dir.iterdir() if i.is_file() and not i.name.startswith(".")]
    orig_seq_numbers = [int(i.name.split("_")[1]) for i in filenames]
    seq_subtract = min(orig_seq_numbers) - 1 # we want to start with 1

    n_frames = n_masks = n_seg = 0
    for img_path in images:
        name = img_path.name
        lower = name.lower()
        is_watershed = lower.endswith("_watershed_mask.png")
        is_color_seg = lower.endswith("_color_mask.png")
        is_any_mask = "mask" in lower
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[skip] failed to read {img_path}")
                continue
            cropped = crop_image(
                img, args.crop_left, args.crop_right, args.crop_top, args.crop_bottom
            )

            # remove "endo" and adjust sequence number
            name_components = name.split("_")
            new_seq = int(name_components[1]) - seq_subtract
            name_components[1] = f"{new_seq:06d}"
            out_name = "_".join(name_components)
            out_name = out_name.replace("_endo", "")

            if is_watershed:
                out_path = masks_out / out_name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_masks += 1
            elif is_color_seg:
                out_path = seg_out / out_name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_seg += 1
            elif is_any_mask:
                continue
            else:
                out_path = frames_out / out_name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_frames += 1
        except Exception as e:
            print(f"[err] {img_path}: {e}")

    print(
        f"Done cropping. Frames: {n_frames}, watershed masks: {n_masks}, seg masks: {n_seg}"
    )

    # Note: Removed optional COLMAP execution


if __name__ == "__main__":
    main()
