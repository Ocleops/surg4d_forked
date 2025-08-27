#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

# Ensure project root is on sys.path so `scene` package is importable when running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use the same COLMAP reader and PLY writer as the training code
from scene.colmap_loader import read_points3D_binary  # returns xyz, rgb, _
from scene.dataset_readers import storePly  # writes normals and float RGB


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract point cloud from COLMAP points3D.bin and save as PLY "
            "using the project's PLY writer (with normals)."
        )
    )
    parser.add_argument("--bin", required=True, help="Path to points3D.bin")
    parser.add_argument(
        "--out",
        required=False,
        help="Path to output PLY (defaults to points3D.ply next to BIN)",
    )
    args = parser.parse_args()

    bin_path = os.path.abspath(args.bin)
    if not os.path.isfile(bin_path):
        raise SystemExit(f"File not found: {bin_path}")

    out_path = args.out
    if not out_path:
        out_path = os.path.join(os.path.dirname(bin_path), "points3D.ply")
    out_path = os.path.abspath(out_path)

    xyz, rgb, _ = read_points3D_binary(bin_path)
    # storePly expects rgb as 0..255 values (floats are fine); it will write
    # x,y,z,nx,ny,nz,red,green,blue as float properties and set normals to 0.
    storePly(out_path, xyz, rgb)

    print(f"Wrote {len(xyz)} points to {out_path}")


if __name__ == "__main__":
    main()
