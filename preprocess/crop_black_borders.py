#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Dict, Set
import subprocess

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# CholecSeg8k color -> class id mapping (RGB)
COLOR_TO_CLASS: Dict[tuple, int] = {
    (50, 50, 50): 0,  # Black Background
    (11, 11, 11): 1,  # Abdominal Wall
    (21, 21, 21): 2,  # Liver
    (13, 13, 13): 3,  # Gastrointestinal Tract
    (12, 12, 12): 4,  # Fat
    (31, 31, 31): 5,  # Grasper
    (23, 23, 23): 6,  # Connective Tissue
    (24, 24, 24): 7,  # Blood
    (25, 25, 25): 8,  # Cystic Duct
    (32, 32, 32): 9,  # L-hook Electrocautery
    (22, 22, 22): 10,  # Gallbladder
    (33, 33, 33): 11,  # Hepatic Vein
    (5, 5, 5): 12,  # Liver Ligament
}

CLASS_TO_COLOR: Dict[int, tuple] = {v: k for k, v in COLOR_TO_CLASS.items()}


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


def build_colmap_masks_from_watershed(
    frames_dir: Path, masks_dir: Path, out_mask_dir: Path, invert: bool = True
):
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        p
        for p in sorted(frames_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png"
    ]
    written = 0
    for frame in frames:
        mname = frame.name.replace(".png", "_watershed_mask.png")
        mpath = masks_dir / mname
        if mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            dynamic = (m > 0).astype(np.uint8)
            allowed = (1 - dynamic) * 255 if invert else dynamic * 255
        else:
            fimg = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if fimg is None:
                continue
            allowed = np.full_like(fimg, 255, dtype=np.uint8)
        outp = out_mask_dir / frame.name
        cv2.imwrite(str(outp), allowed)
        written += 1
    print(f"Built {written} COLMAP mask images (watershed) in {out_mask_dir}")


def build_colmap_masks_from_seg(
    frames_dir: Path,
    seg_dir: Path,
    out_mask_dir: Path,
    classes_to_mask: Set[int] = None,
    classes_to_keep: Set[int] = None,
    seg_suffix: str = "mask",
):  # noqa: E501
    """Build per-image binary masks from class segmentation PNGs.
    - frames_dir: directory with frames_cropped (PNG)
    - seg_dir: directory with segmentation PNGs (already cropped to same crop)
    - classes_to_mask: mask these class IDs (exclude from COLMAP)
    - classes_to_keep: keep only these class IDs (mask all others)
    - seg_suffix: if segmentations are named <frame>.png or <frame>_<seg_suffix>.png
    Mask format: 255 allowed, 0 masked.
    """
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        p
        for p in sorted(frames_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png"
    ]
    written = 0
    for frame in frames:
        seg_path = seg_dir / frame.name
        if not seg_path.exists():
            seg_path = seg_dir / frame.name.replace(".png", f"_{seg_suffix}.png")
        if not seg_path.exists():
            fimg = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if fimg is None:
                continue
            allowed = np.full_like(fimg, 255, dtype=np.uint8)
        else:
            seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
            if seg is None:
                continue
            seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            h, w, _ = seg_rgb.shape
            masked = np.zeros((h, w), dtype=np.uint8)
            if classes_to_keep is not None:
                for cls, color in CLASS_TO_COLOR.items():
                    if cls in classes_to_keep:
                        continue
                    m = (
                        (seg_rgb[:, :, 0] == color[0])
                        & (seg_rgb[:, :, 1] == color[1])
                        & (seg_rgb[:, :, 2] == color[2])
                    )
                    masked[m] = 1
            elif classes_to_mask is not None:
                for cls in classes_to_mask:
                    color = CLASS_TO_COLOR.get(cls, None)
                    if color is None:
                        continue
                    m = (
                        (seg_rgb[:, :, 0] == color[0])
                        & (seg_rgb[:, :, 1] == color[1])
                        & (seg_rgb[:, :, 2] == color[2])
                    )
                    masked[m] = 1
            allowed = (1 - masked) * 255
        outp = out_mask_dir / frame.name
        cv2.imwrite(str(outp), allowed)
        written += 1
    print(f"Built {written} COLMAP mask images (segmentation) in {out_mask_dir}")


def _maybe_wrap_xvfb(cmd: list[str], use_xvfb: bool) -> list[str]:
    if not use_xvfb:
        return cmd
    return [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x720x24",
    ] + cmd


def run_colmap(
    colmap_bin: str,
    images_dir: Path,
    output_dir: Path,
    mask_dir: Path = None,
    log_dir: Path | None = None,
    use_xvfb: bool = False,
    gpu_sift: bool = True,
    gpu_match: bool = True,
    gaussians_style: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    # If gaussians_style, mimic 4DGaussians layout under output_dir/colmap
    if gaussians_style:
        colmap_root = output_dir  # keep using the same output root
        images_colmap = colmap_root / "images"
        images_colmap.mkdir(parents=True, exist_ok=True)
        # symlink or copy images into colmap/images
        # Use symlinks to save space
        for p in sorted(images_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".png":
                dst = images_colmap / p.name
                if not dst.exists():
                    try:
                        os.symlink(p, dst)
                    except FileExistsError:
                        pass
        images_dir = images_colmap

    db = output_dir / "database.db"
    sparse = output_dir / "sparse"
    sparse.mkdir(exist_ok=True)

    sift_gpu_flag = "1" if gpu_sift else "0"
    match_gpu_flag = "1" if gpu_match else "0"

    feat_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(images_dir),
        "--SiftExtraction.use_gpu",
        sift_gpu_flag,
    ]
    if gaussians_style:
        feat_cmd += [
            "--SiftExtraction.max_image_size",
            "4096",
            "--SiftExtraction.max_num_features",
            "16384",
            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
        ]
    if mask_dir is not None:
        feat_cmd += ["--ImageReader.mask_path", str(mask_dir)]
    feat_cmd = _maybe_wrap_xvfb(feat_cmd, use_xvfb)

    match_cmd = [
        colmap_bin,
        "exhaustive_matcher",
        "--database_path",
        str(db),
        "--SiftMatching.use_gpu",
        match_gpu_flag,
    ]
    match_cmd = _maybe_wrap_xvfb(match_cmd, use_xvfb)

    # Prefer mapper since we don't have sparse_custom like 4DGaussians conversion scripts
    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse),
    ]
    mapper_cmd = _maybe_wrap_xvfb(mapper_cmd, use_xvfb)

    def run(cmd, name: str):
        print("Running:", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        if log_dir is not None:
            with open(log_dir / f"{name}.log", "w") as f:
                subprocess.run(cmd, check=True, env=env, stdout=f, stderr=f)
        else:
            subprocess.run(cmd, check=True, env=env)

    run(feat_cmd, "feature_extractor")
    run(match_cmd, "exhaustive_matcher")
    run(mapper_cmd, "mapper")
    print(f"COLMAP finished. Outputs in {output_dir}")


def parse_class_list(s: str) -> Set[int]:
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="Crop images and (optionally) run COLMAP with per-image masks"
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
        "--crop_right", type=int, default=110, help="Pixels to crop from the right"
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
    # COLMAP controls
    ap.add_argument(
        "--run_colmap", action="store_true", help="Run COLMAP after cropping"
    )
    ap.add_argument(
        "--colmap_bin", type=str, default="colmap", help="Path to COLMAP binary"
    )
    ap.add_argument(
        "--invert_masks_for_colmap",
        action="store_true",
        help="Invert watershed masks (default recommended)",
    )
    ap.add_argument(
        "--use_xvfb",
        action="store_true",
        help="Wrap COLMAP commands in xvfb-run for headless environments",
    )
    ap.add_argument(
        "--gpu_sift", action="store_true", help="Use GPU-accelerated SIFT (COLMAP)"
    )
    ap.add_argument(
        "--gpu_match", action="store_true", help="Use GPU-accelerated matching (COLMAP)"
    )
    ap.add_argument(
        "--gaussians_style",
        action="store_true",
        help="Use 4DGaussians-style COLMAP args and folder layout",
    )
    ap.add_argument(
        "--no_colmap_masks",
        action="store_true",
        help="Do not generate or pass any masks to COLMAP",
    )
    # Segmentation-driven masks
    ap.add_argument(
        "--seg_masks_dir",
        type=str,
        default=None,
        help="Directory with segmentation PNGs (cropped)",
    )
    ap.add_argument(
        "--seg_mask_suffix",
        type=str,
        default="mask",
        help="Suffix used for seg files if not exact name match",
    )
    ap.add_argument(
        "--mask_classes",
        type=str,
        default=None,
        help="Comma-separated class IDs to mask (e.g., '5,7,9')",
    )
    ap.add_argument(
        "--keep_classes",
        type=str,
        default=None,
        help="Comma-separated class IDs to keep (mask all others)",
    )
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
            if is_watershed:
                out_path = masks_out / name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_masks += 1
            elif is_color_seg:
                out_path = seg_out / name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_seg += 1
            elif is_any_mask:
                continue
            else:
                out_path = frames_out / name
                if out_path.exists() and not args.overwrite:
                    continue
                cv2.imwrite(str(out_path), cropped)
                n_frames += 1
        except Exception as e:
            print(f"[err] {img_path}: {e}")

    print(
        f"Done cropping. Frames: {n_frames}, watershed masks: {n_masks}, seg masks: {n_seg}"
    )

    if args.run_colmap:
        suffix = ""
        if args.keep_classes:
            keep = parse_class_list(args.keep_classes)
            suffix = "keep_cls" + "_".join(str(x) for x in sorted(keep))
        elif args.mask_classes:
            mask = parse_class_list(args.mask_classes)
            suffix = "mask_cls" + "_".join(str(x) for x in sorted(mask))
        else:
            suffix = "watershed" if not args.no_colmap_masks else "nomask"
        colmap_out = in_dir / f"colmap_cropped_{suffix}"
        logs_dir = colmap_out / "Logs"
        colmap_masks = in_dir / "colmap_masks"
        mask_dir_for_colmap = None
        if not args.no_colmap_masks:
            if args.keep_classes or args.mask_classes:
                classes_to_keep = (
                    parse_class_list(args.keep_classes) if args.keep_classes else None
                )
                classes_to_mask = (
                    parse_class_list(args.mask_classes) if args.mask_classes else None
                )
                seg_source = Path(args.seg_masks_dir) if args.seg_masks_dir else seg_out
                build_colmap_masks_from_seg(
                    frames_out,
                    seg_source,
                    colmap_masks,
                    classes_to_mask=classes_to_mask,
                    classes_to_keep=classes_to_keep,
                    seg_suffix=args.seg_mask_suffix,
                )
                mask_dir_for_colmap = colmap_masks
            else:
                build_colmap_masks_from_watershed(
                    frames_out,
                    masks_out,
                    colmap_masks,
                    invert=args.invert_masks_for_colmap,
                )
                mask_dir_for_colmap = colmap_masks
        try:
            run_colmap(
                args.colmap_bin,
                frames_out,
                colmap_out,
                mask_dir=mask_dir_for_colmap,
                log_dir=logs_dir,
                use_xvfb=args.use_xvfb,
                gpu_sift=args.gpu_sift,
                gpu_match=args.gpu_match,
                gaussians_style=args.gaussians_style,
            )
        except subprocess.CalledProcessError as e:
            print(f"[colmap error] {e}")


if __name__ == "__main__":
    main()
