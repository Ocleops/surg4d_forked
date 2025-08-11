#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render colored overlay with numeric segment labels"
    )
    p.add_argument(
        "--image", type=str, required=True, help="Path to the original frame image"
    )
    p.add_argument(
        "--segmap", type=str, required=True, help="Path to *_s.npy seg-map file"
    )
    p.add_argument(
        "--layer_index", type=int, default=0, help="Layer index if segmap is (L,H,W)"
    )
    p.add_argument(
        "--alpha", type=float, default=0.35, help="Blend factor for overlay [0..1]"
    )
    p.add_argument("--out", type=str, required=True, help="Output image path")
    return p.parse_args()


def load_seg(seg_path: str, layer: int) -> np.ndarray:
    seg = np.load(seg_path)
    if seg.ndim == 3:
        seg = seg[layer]
    elif seg.ndim != 2:
        raise ValueError(f"Unsupported seg shape {seg.shape}")
    return seg.astype(np.int32)


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    S = load_seg(args.segmap, args.layer_index)

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
        color = palette[sid % len(palette)].tolist()
        overlay[mask] = (
            (1 - args.alpha) * overlay[mask] + args.alpha * np.array(color)
        ).astype(np.uint8)

    vis = overlay.copy()
    for sid in valid_ids:
        ys, xs = np.where(S == sid)
        if ys.size == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), -1)
        cv2.putText(
            vis,
            str(sid),
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            str(sid),
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"Wrote {args.out} with ids {valid_ids}")


if __name__ == "__main__":
    main()
