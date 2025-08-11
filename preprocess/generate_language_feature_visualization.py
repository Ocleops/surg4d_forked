#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
from typing import List

import numpy as np
import cv2


def list_pairs(lf_dir: Path) -> List[str]:
    names = set()
    for p in lf_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith("_f.npy"):
            stem = p.name[:-6]  # drop _f.npy
            names.add(stem)
    return sorted(names)


def load_seg(seg_path: Path, layer: int = 0) -> np.ndarray:
    seg = np.load(str(seg_path))
    if seg.ndim == 3:
        seg = seg[layer]
    elif seg.ndim != 2:
        raise ValueError(f"Unsupported seg shape {seg.shape} in {seg_path}")
    return seg.astype(np.int32)


def make_overlay(image_path: Path, seg_path: Path, out_path: Path, alpha: float = 0.35, layer_index: int = 0) -> List[int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    S = load_seg(seg_path, layer_index)
    H, W = S.shape
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H))

    valid_ids = sorted([int(i) for i in np.unique(S) if i >= 0])

    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
            [128, 255, 0],
            [255, 0, 128],
            [0, 255, 128],
        ],
        dtype=np.uint8,
    )

    overlay = img.copy()
    for sid in valid_ids:
        mask = S == sid
        if not np.any(mask):
            continue
        color = palette[sid % len(palette)].tolist()
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * np.array(color)).astype(np.uint8)

    vis = overlay.copy()
    for sid in valid_ids:
        ys, xs = np.where(S == sid)
        if ys.size == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), -1)
        cv2.putText(vis, str(sid), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(sid), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return valid_ids


def run_debug_print(debug_script: Path, f_path: Path, s_path: Path, out_txt: Path, layer_index: int = 0) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(debug_script),
        "--features",
        str(f_path),
        "--segmap",
        str(s_path),
        "--layer_index",
        str(layer_index),
    ]
    with open(out_txt, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=f)


def maybe_copy_existing_overlays(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    for p in src_dir.iterdir():
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            dst = dst_dir / p.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(p, dst)
                except Exception:
                    # fallback copy
                    import shutil
                    shutil.copy2(p, dst)


def main():
    ap = argparse.ArgumentParser(description="Collect language feature visualizations and debug prints")
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--lf_dir", type=str, required=True, help="Directory with *_f.npy and *_s.npy")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory; defaults to lf_dir/language_feature_visualisation")
    ap.add_argument("--debug_script", type=str, default=str(Path(__file__).with_name("debug_clip_feature_sanity.py")))
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--layer_index", type=int, default=0)
    ap.add_argument("--copy_existing_overlays", action="store_true")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    lf_dir = Path(args.lf_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (lf_dir / "language_feature_visualisation")
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = list_pairs(lf_dir)
    for stem in stems:
        f_path = lf_dir / f"{stem}_f.npy"
        s_path = lf_dir / f"{stem}_s.npy"
        img_path = frames_dir / f"{stem}.png"
        if not f_path.exists() or not s_path.exists():
            continue
        # overlay
        overlay_path = out_dir / f"{stem}_overlay.png"
        try:
            make_overlay(img_path, s_path, overlay_path, alpha=args.alpha, layer_index=args.layer_index)
        except Exception as e:
            print(f"[overlay skip] {stem}: {e}")
        # debug print
        txt_path = out_dir / f"{stem}_debug.txt"
        try:
            run_debug_print(Path(args.debug_script), f_path, s_path, txt_path, layer_index=args.layer_index)
        except subprocess.CalledProcessError as e:
            print(f"[debug skip] {stem}: {e}")

    if args.copy_existing_overlays:
        maybe_copy_existing_overlays(frames_dir.parent / "overlays", out_dir)

    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main() 