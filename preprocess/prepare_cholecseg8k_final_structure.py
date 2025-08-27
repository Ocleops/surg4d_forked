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


def list_feature_pairs(d: Path):
    pairs = {}
    for f in d.glob("*_f.npy"):
        s_file = f.with_name(f.name.replace("_f.npy", "_s.npy"))
        if s_file.exists():
            pairs[f] = s_file
    return pairs


def rename_language_features(
    src_dir: Path, out_dir: Path, renumber_sequential: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_feature_pairs(src_dir)
    if not pairs:
        print(f"No matching language feature files in {src_dir}")
        return

    # Sort by original frame number
    def sort_key(item):
        m = re.search(r"(\d+)", item[0].name)
        return int(m.group(1)) if m else -1

    sorted_pairs = sorted(pairs.items(), key=sort_key)
    print(f"Renaming and copying {len(sorted_pairs)} language feature pairs...")

    for i, (f_src, s_src) in enumerate(sorted_pairs):
        if renumber_sequential:
            seq_num = i + 1
            f_dst_name = f"{seq_num:06d}_f.npy"
            s_dst_name = f"{seq_num:06d}_s.npy"
        else:
            name_no_ext = f_src.stem.replace("_f", "")
            name_adjusted = name_no_ext.replace("frame_", "").replace("_endo", "")
            # Ensure 6 digits
            m = re.search(r"(\d+)", name_adjusted)
            if m:
                num_str = m.group(1)
                name_adjusted = name_adjusted.replace(num_str, f"{int(num_str):06d}")
            f_dst_name = f"{name_adjusted}_f.npy"
            s_dst_name = f"{name_adjusted}_s.npy"

        shutil.copy(f_src, out_dir / f_dst_name)
        shutil.copy(s_src, out_dir / s_dst_name)


def rebuild_from_source(
    src_dir: Path, out_dir: Path, renumber_sequential: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )
    if not files:
        print(f"No PNGs found in {src_dir} to rebuild from")
        return

    for i, f_src in enumerate(files):
        if renumber_sequential:
            seq_num = i + 1
            dst_name = f"frame_{seq_num:06d}.png"
        else:
            # Keep original name but ensure 6-digit padding
            m = re.search(r"(\d+)", f_src.stem)
            if m:
                num_str = m.group(1)
                dst_name = f_src.name.replace(num_str, f"{int(num_str):06d}")
            else:
                dst_name = f_src.name
        shutil.copy(f_src, out_dir / dst_name)
    print(f"Copied and renumbered {len(files)} images to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    # Mode selection (NOTE: These are now all inputs to a single full build process)
    ap.add_argument(
        "--rename_language_features_dir",
        type=str,
        help="Run only language feature renaming on this source directory.",
    )
    ap.add_argument(
        "--renumber_sequential",
        action="store_true",
        help="Force renumbering of all files to be sequential from 1.",
    )
    ap.add_argument(
        "--rebuild_from_source_dir",
        type=str,
        help="Rebuild RGB folder from a source of PNGs, renumbering them.",
    )
    # Full build paths
    ap.add_argument(
        "--src_images_root",
        type=str,
        default=None,
        help="Source dir for rgb images (e.g. frames_cropped)",
    )
    ap.add_argument(
        "--src_ply_path",
        type=str,
        default=None,
        help="Source path for points3D.ply",
    )
    ap.add_argument(
        "--src_sparse_dir",
        type=str,
        default=None,
        help="Source dir for colmap sparse model (bin files)",
    )
    ap.add_argument(
        "--dst_suffix",
        type=str,
        default="",
        help="Suffix to append to the destination folder name",
    )
    args = ap.parse_args()

    if not args.src_images_root:
        raise SystemExit("Must provide --src_images_root for a full build")

    # --- Full build ---
    src_images_root = Path(args.src_images_root)
    # Derive dataset name from the parent of the images root
    dataset_name = src_images_root.parent.name
    
    # Use the parent of the parent as the base output directory
    base_out_dir = src_images_root.parent.parent
    
    dest_name = f"{dataset_name}_final_for_training{args.dst_suffix}"
    dest_root = base_out_dir / dest_name
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # Create Nerfstudio-like structure
    # 1. rgb/1x -> images
    dest_rgb_1x = dest_root / "rgb" / "1x"
    rebuild_from_source(src_images_root, dest_rgb_1x, args.renumber_sequential)
    
    # Create scaled symlinks inside rgb/
    for scale in [2, 4, 8]:
        scaled_dir = dest_root / "rgb" / f"{scale}x"
        if not scaled_dir.exists():
            print(f"Creating symlink 1x -> {scale}x")
            # Symlink needs to be relative to its own location
            scaled_dir.symlink_to("1x", target_is_directory=True)

    # Create top-level `images` symlink to rgb/1x
    images_symlink = dest_root / "images"
    if not images_symlink.exists():
        images_symlink.symlink_to("rgb/1x", target_is_directory=True)

    # 2. sparse/ -> colmap sparse model
    if args.src_sparse_dir:
        src_sparse = Path(args.src_sparse_dir)
        dest_sparse = dest_root / "sparse"
        print(f"Copying sparse model {src_sparse} -> {dest_sparse}")
        shutil.copytree(src_sparse, dest_sparse, dirs_exist_ok=True)

    # 3. points3D.ply
    if args.src_ply_path:
        src_ply = Path(args.src_ply_path)
        dest_ply = dest_root / "points3D.ply"
        print(f"Copying point cloud {src_ply} -> {dest_ply}")
        shutil.copy(src_ply, dest_ply)

    # 4. language_features/
    # The user provides the source via the --rename_language_features_dir argument
    if args.rename_language_features_dir:
        lang_feat_src_dir = Path(args.rename_language_features_dir)
        if lang_feat_src_dir.exists():
            # Name the destination folder based on the example structure
            dest_lang_feat = dest_root / "language_features_pca3"
            rename_language_features(
                lang_feat_src_dir, dest_lang_feat, args.renumber_sequential
            )
        else:
            print(f"Skipping language features, directory not found: {lang_feat_src_dir}")

    # 5. Create empty placeholder directories
    (dest_root / "camera").mkdir(exist_ok=True)
    (dest_root / "train").mkdir(exist_ok=True)

    print(f"\nDone. Created:\n{dest_root}")
    print("Final structure:")
    os.system(f"ls -l {dest_root}")
    print("\nRGB structure:")
    os.system(f"ls -l {dest_root}/rgb")


if __name__ == "__main__":
    main()
