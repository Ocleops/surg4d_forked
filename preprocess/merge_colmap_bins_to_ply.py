#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import struct
from typing import BinaryIO, List, Tuple


def read_exact(fp: BinaryIO, size: int) -> bytes:
    data = fp.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of file")
    return data


def try_read_uint64(fp: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(fp, 8))[0]


def parse_points3D_bin(
    file_path: Path,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    xyz_list: List[Tuple[float, float, float]] = []
    rgb_list: List[Tuple[int, int, int]] = []
    if not file_path.is_file():
        return xyz_list, rgb_list

    file_size = file_path.stat().st_size
    with open(file_path, "rb") as fp:
        header_bytes = fp.read(8)
        if len(header_bytes) < 8:
            return xyz_list, rgb_list
        num_or_id = struct.unpack("<Q", header_bytes)[0]

        def min_point_record_size() -> int:
            return 8 + 24 + 3 + 8 + 8  # id + xyz + rgb + error + track_len

        has_count_header = True
        if num_or_id == 0 or num_or_id > 10_000_000:
            has_count_header = False
        else:
            if 8 + num_or_id * min_point_record_size() > file_size:
                has_count_header = False

        if not has_count_header:
            fp.seek(0)
            num_points = None
        else:
            num_points = num_or_id

        points_read = 0
        while True:
            try:
                _ = try_read_uint64(fp)  # point3d_id
            except EOFError:
                break
            x, y, z = struct.unpack("<ddd", read_exact(fp, 8 * 3))
            r, g, b = struct.unpack("<BBB", read_exact(fp, 3))
            _error = struct.unpack("<d", read_exact(fp, 8))[0]
            track_len = try_read_uint64(fp)

            # Skip track entries (unknown width: try 32-bit pairs, else 64-bit)
            save = fp.tell()
            try:
                if track_len > 0:
                    read_exact(fp, track_len * 8)  # 2 * uint32
            except EOFError:
                fp.seek(save)
                if track_len > 0:
                    read_exact(fp, track_len * 16)  # 2 * uint64

            xyz_list.append((x, y, z))
            rgb_list.append((int(r), int(g), int(b)))
            points_read += 1
            if num_points is not None and points_read >= num_points:
                break

    return xyz_list, rgb_list


def write_ply_ascii(
    ply_path: Path,
    xyz: List[Tuple[float, float, float]],
    rgb: List[Tuple[int, int, int]],
):
    assert len(xyz) == len(rgb)
    n = len(xyz)
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Merge COLMAP points3D.bin from sparse/0 and sparse/1 into a single PLY"
    )
    ap.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="Path to sparse directory containing 0/ and/or 1/",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PLY path (default: sparse/points3D_merged.ply)",
    )
    args = ap.parse_args()

    sparse_dir = Path(args.sparse_dir)
    if not sparse_dir.is_dir():
        raise SystemExit(f"Sparse directory not found: {sparse_dir}")

    bin_paths = [
        sparse_dir / "0" / "points3D.bin",
        sparse_dir / "1" / "points3D.bin",
    ]

    all_xyz: List[Tuple[float, float, float]] = []
    all_rgb: List[Tuple[int, int, int]] = []
    total = 0
    for p in bin_paths:
        xyz, rgb = parse_points3D_bin(p)
        all_xyz.extend(xyz)
        all_rgb.extend(rgb)
        total += len(xyz)

    if total == 0:
        raise SystemExit("No points found in provided bins.")

    out_path = Path(args.out) if args.out else (sparse_dir / "points3D_merged.ply")
    write_ply_ascii(out_path, all_xyz, all_rgb)
    print(f"Wrote merged PLY with {total} points to {out_path}")


if __name__ == "__main__":
    main()

