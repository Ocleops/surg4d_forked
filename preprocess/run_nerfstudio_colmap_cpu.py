#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def copy_or_symlink_images(src_dir: Path, dst_dir: Path, symlink: bool) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
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


def run_cmd(cmd: list[str], log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # limit threads for stability in headless CPU mode
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with open(log_file, "w") as f:
        subprocess.run(cmd, check=True, env=env, stdout=f, stderr=f)


def run_colmap_cpu(colmap_bin: str, images_dir: Path, colmap_root: Path, logs_dir: Path):
    db = colmap_root / "database.db"
    sparse = colmap_root / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    feat_cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.max_image_size", "4096",
        "--SiftExtraction.max_num_features", "16384",
        # more stable on CPU/headless
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling", "0",
        "--SiftExtraction.num_threads", "7",
    ]
    match_cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "7",
    ]
    mapper_cmd = [
        colmap_bin, "mapper",
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--output_path", str(sparse),
    ]

    run_cmd(feat_cmd, logs_dir / "feature_extractor.log")
    run_cmd(match_cmd, logs_dir / "exhaustive_matcher.log")
    run_cmd(mapper_cmd, logs_dir / "mapper.log")
    # print(" ".join(feat_cmd))
    # print(" ".join(match_cmd))
    # print(" ".join(mapper_cmd))
    exit(0)


def main():
    ap = argparse.ArgumentParser(description="Prepare Nerfstudio folder and run COLMAP on CPU (headless)")
    ap.add_argument("--frames_dir", type=str, required=True, help="Directory with cropped frames")
    ap.add_argument("--out_dir", type=str, required=True, help="Nerfstudio output root (will create images/ and colmap/)" )
    ap.add_argument("--colmap_bin", type=str, default="colmap", help="Path to COLMAP binary")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory not found: {frames_dir}")

    images_ns = out_dir / "images"
    images_colmap = out_dir / "colmap" / "images"
    logs_dir = out_dir / "colmap" / "Logs"

    n1 = copy_or_symlink_images(frames_dir, images_ns, args.symlink)
    n2 = copy_or_symlink_images(frames_dir, images_colmap, args.symlink)
    print(f"Prepared Nerfstudio images: {n1} (images/), {n2} (colmap/images)")

    run_colmap_cpu(args.colmap_bin, images_colmap, out_dir / "colmap", logs_dir)
    print(f"Done. COLMAP outputs in: {out_dir / 'colmap'}")


if __name__ == "__main__":
    main() 