#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
import re

SCALES = {
    "images": "1x",
    "images_2": "2x",
    "images_4": "4x",
    "images_8": "8x",
    # if there is a 16x, map accordingly (not present here, but keeping for parity)
    "images_16": "16x",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        tgt_dir = dst / rel
        ensure_dir(tgt_dir)
        for f in files:
            s = Path(root) / f
            d = tgt_dir / f
            shutil.copy2(s, d)


def rename_language_features(lf_dir: Path, renumber_sequential: bool = False) -> int:
    """Rename files inside language feature folder by:
    - If filenames are 'frame_XXXXXX_endo[_f.npy|_s.npy|_rgb.png]': drop 'frame_' and '_endo'
    - Otherwise accept already-renamed 'XXXXXX[_f.npy|_s.npy|_rgb.png]'
    - If renumber_sequential is True: remap in ascending numeric order to 000001..0000NN
    Returns number of files renamed.
    """
    if not lf_dir or not lf_dir.exists():
        return 0
    pat_old = re.compile(r"^frame_(\d+)_endo(_f\.npy|_s\.npy|_rgb\.png)$")
    pat_new = re.compile(r"^(\d{1,6})(_[fs]\.npy|_rgb\.png)$")

    entries = []  # list of (orig_name, orig_num:int, suffix)
    for name in os.listdir(lf_dir):
        m = pat_old.match(name)
        if m:
            num = int(m.group(1))
            suffix = m.group(2)
            entries.append((name, num, suffix))
            continue
        m2 = pat_new.match(name)
        if m2:
            num = int(m2.group(1))
            suffix = m2.group(2)
            entries.append((name, num, suffix))

    if not entries:
        print(f"No matching language feature files in {lf_dir}")
        return 0

    # If not renumbering, just normalize to zero-padded 6-digit
    renamed = 0
    if not renumber_sequential:
        for name, num, suffix in entries:
            new_name = f"{num:06}{suffix}"
            src = lf_dir / name
            dst = lf_dir / new_name
            if src == dst:
                continue
            if dst.exists():
                print(f"SKIP (exists): {dst}")
                continue
            os.rename(src, dst)
            print(f"Renamed: {name} -> {new_name}")
            renamed += 1
        print(f"Total renamed in {lf_dir}: {renamed}")
        return renamed

    # Renumber sequentially starting at 1, preserving order by original numeric id
    entries.sort(key=lambda x: (x[1], x[2]))
    # Group by base number to enforce same index for _f and _s; png ignored for counting
    groups = {}
    for name, num, suffix in entries:
        grp = groups.setdefault(num, {"files": []})
        grp["files"].append((name, suffix))

    idx = 0
    for num in sorted(groups.keys()):
        # Only increment if there is any of f/s for this base number
        suffixes = [s for _, s in groups[num]["files"] if s in ("_f.npy", "_s.npy")]
        if not suffixes:
            # no f/s present; skip numbering (png only)
            continue
        idx += 1
        for name, suffix in groups[num]["files"]:
            new_name = f"{idx:06}{suffix}"
            src = lf_dir / name
            dst = lf_dir / new_name
            if src == dst:
                continue
            if dst.exists():
                print(f"SKIP (exists): {dst}")
                continue
            os.rename(src, dst)
            print(f"Renamed: {name} -> {new_name}")
            renamed += 1
    print(f"Total sequentially renamed in {lf_dir}: {renamed}")
    return renamed


def rebuild_from_source(lf_src_dir: Path, lf_dst_dir: Path) -> int:
    """Rebuild destination language features from source folder:
    - Group by original frame number
    - Assign sequential index starting at 1 (000001..), ignoring .png for counting
    - Copy _f.npy and _s.npy; copy _rgb.png with the same assigned index if present
    Overwrites existing destination files with same names.
    Returns number of files written.
    """
    if not lf_src_dir.exists():
        print(f"Source language feature dir not found: {lf_src_dir}")
        return 0
    ensure_dir(lf_dst_dir)
    # Clear existing target files matching patterns to avoid conflicts
    for p in lf_dst_dir.glob("*_f.npy"):
        p.unlink(missing_ok=True)
    for p in lf_dst_dir.glob("*_s.npy"):
        p.unlink(missing_ok=True)
    for p in lf_dst_dir.glob("*_rgb.png"):
        p.unlink(missing_ok=True)

    pat_old = re.compile(r"^frame_(\d+)_endo(_f\.npy|_s\.npy|_rgb\.png)$")
    pat_new = re.compile(r"^(\d{1,6})(_[fs]\.npy|_rgb\.png)$")
    groups = {}
    for name in os.listdir(lf_src_dir):
        m = pat_old.match(name)
        if m:
            num = int(m.group(1))
            suffix = m.group(2)
            groups.setdefault(num, []).append((name, suffix))
            continue
        m2 = pat_new.match(name)
        if m2:
            num = int(m2.group(1))
            suffix = m2.group(2)
            groups.setdefault(num, []).append((name, suffix))

    written = 0
    idx = 0
    for num in sorted(groups.keys()):
        suffixes = [s for _, s in groups[num] if s in ("_f.npy", "_s.npy")]
        if not suffixes:
            continue
        idx += 1
        for name, suffix in groups[num]:
            src = lf_src_dir / name
            dst = lf_dst_dir / f"{idx:06}{suffix}"
            shutil.copy2(src, dst)
            print(f"Wrote: {dst.name}")
            written += 1
    print(f"Total rebuilt files in {lf_dst_dir}: {written}")
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Prepare cholecseg8k final training structure"
    )
    ap.add_argument(
        "--src_colmap_dir",
        type=str,
        required=False,
        default="/home/tumai/team1/Ken/4DLangSplatSurgery/data/cholecseg8k/video01/video01_14939_firstry/colmap",
        help="Legacy: Source COLMAP directory containing images_*/ and sparse_pc.ply. If --src_images_root is given, this is ignored for images.",
    )
    ap.add_argument(
        "--src_images_root",
        type=str,
        required=False,
        default=None,
        help="Root folder that contains images, images_2, images_4, ... from the run you want (e.g., cropped COLMAP output).",
    )
    ap.add_argument(
        "--src_sparse_dir",
        type=str,
        required=False,
        default=None,
        help="Path to a COLMAP sparse model directory (e.g., .../colmap/sparse or .../colmap/sparse/0). Will be copied to <dst>/sparse.",
    )
    ap.add_argument(
        "--src_ply_path",
        type=str,
        required=False,
        default=None,
        help="Explicit path to point cloud .ply to copy. If not set, will look for sparse_pc.ply under src_images_root or src_colmap_dir.",
    )
    ap.add_argument(
        "--dst_root",
        type=str,
        required=False,
        default="/home/tumai/team1/Ken/4DLangSplatSurgery/data/cholecseg8k/video01_14939_final_for_training",
        help="Destination dataset root to create (hypernerf-like)",
    )
    ap.add_argument(
        "--append_cropped_to_dst",
        action="store_true",
        help="If set, append '_cropped' to the destination folder name automatically.",
    )
    ap.add_argument(
        "--dst_suffix",
        type=str,
        default=None,
        help="If set, append this suffix to the destination folder name (e.g., 'cropped' -> '<dst_root>_cropped'). Overrides --append_cropped_to_dst.",
    )
    ap.add_argument(
        "--rename_language_features_dir",
        type=str,
        required=False,
        default=None,
        help="Destination language feature folder to operate on. If not provided, defaults to <dst_root>/language_features_default.",
    )
    ap.add_argument(
        "--renumber_sequential",
        action="store_true",
        help="If set, remap numbers to contiguous 000001..0000NN by ascending original index",
    )
    ap.add_argument(
        "--lf_source_dir",
        type=str,
        default=None,
        help="If provided, copy and rebuild numbering from this source folder into the destination LF folder. Accepts frame_XXXX_endo_* or XXXXXX_*.",
    )
    args = ap.parse_args()

    # Resolve destination with optional suffix
    dst = Path(args.dst_root)
    suffix_to_apply = None
    if args.dst_suffix:
        suf = args.dst_suffix
        if not suf.startswith("_"):
            suf = "_" + suf
        suffix_to_apply = suf
    elif args.append_cropped_to_dst:
        suffix_to_apply = "_cropped"
    if suffix_to_apply and not dst.name.endswith(suffix_to_apply):
        dst = dst.parent / f"{dst.name}{suffix_to_apply}"

    # Resolve image root (where images_*, etc., live)
    images_root = Path(args.src_images_root) if args.src_images_root else Path(args.src_colmap_dir)

    # Create top-level structure similar to hypernerf/americano
    ensure_dir(dst)
    ensure_dir(dst / "rgb")
    ensure_dir(dst / "camera")  # empty placeholder
    ensure_dir(dst / "train")  # empty placeholder

    # Copy/rename rgb scales
    for folder_name, scale_name in SCALES.items():
        src_folder = images_root / folder_name
        if src_folder.exists():
            print(f"Copying {src_folder} -> {dst / 'rgb' / scale_name}")
            copy_tree(src_folder, dst / "rgb" / scale_name)
        else:
            print(f"Skip missing folder: {src_folder}")

    # Copy sparse model if requested
    if args.src_sparse_dir:
        sparse_src = Path(args.src_sparse_dir)
        sparse_dst = dst / "sparse"
        if sparse_src.exists():
            print(f"Copying sparse model {sparse_src} -> {sparse_dst}")
            copy_tree(sparse_src, sparse_dst)
        else:
            print(f"WARNING: sparse source not found: {sparse_src}")

    # Copy point cloud
    if args.src_ply_path:
        src_ply = Path(args.src_ply_path)
    else:
        # try images_root first, then fallback to legacy src_colmap_dir
        candidate1 = images_root / "sparse_pc.ply"
        candidate2 = Path(args.src_colmap_dir) / "sparse_pc.ply"
        src_ply = candidate1 if candidate1.exists() else candidate2
    dst_ply = dst / "points3D.ply"
    if src_ply.exists():
        print(f"Copying point cloud {src_ply} -> {dst_ply}")
        shutil.copy2(src_ply, dst_ply)
    else:
        print(f"WARNING: point cloud not found: {src_ply}")

    # Determine LF destination directory
    lf_dir = (
        Path(args.rename_language_features_dir)
        if args.rename_language_features_dir
        else (dst / "language_features_default")
    )
    # Ensure LF destination exists if we are going to use it
    if args.lf_source_dir or (lf_dir and lf_dir.exists()):
        ensure_dir(lf_dir)

    # Prefer rebuilding from source if provided
    if args.lf_source_dir:
        rebuild_from_source(Path(args.lf_source_dir), lf_dir)
    elif lf_dir and lf_dir.exists():
        rename_language_features(
            lf_dir, renumber_sequential=args.renumber_sequential
        )
    else:
        if lf_dir:
            print(f"Language feature dir not found, skip rename: {lf_dir}")

    print("Done. Created:")
    print(dst)
    print((dst / "rgb").glob("*"))


if __name__ == "__main__":
    main()
