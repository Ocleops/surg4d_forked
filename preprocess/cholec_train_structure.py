from pathlib import Path
import argparse
import shutil

def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare cholecseg8k final training structure"
    )
    ap.add_argument(
        "--src_root",
        type=str,
    )
    ap.add_argument(
        "--dest_root",
        type=str,
    )
    return ap.parse_args()

def main():
    args = parse_args()

    src = Path(args.src_root)
    dest = Path(args.dest_root)

    assert src.exists(), f"Source root {src} does not exist"
    assert not dest.exists(), f"Destination root {dest} already exists"
    assert (src / "colmap/colmap/sparse/0").exists() and not (src / "colmap/colmap/sparse/1").exists(), "colmap produced disconnected components, delete all unwanted and name wanted 0"

    dest.mkdir(parents=True, exist_ok=True)
    
    # camera dir must exist but can be empty
    cam_dir = dest / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # train dir must exist but can be empty
    train_dir = dest / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # point cloud must be in root i guess?
    pc_file = src / "colmap/colmap/sparse" / "0" / "points3D.ply"
    shutil.copy(pc_file, dest / "points3D.ply")

    # colmap sparse goes to root
    shutil.copytree(src / "colmap/colmap/sparse", dest / "sparse")

    # language features go to root
    lang_raw_dir = src / "language_features_default"
    lang_ae_dir = src / "language_features_default-language_features_dim3"
    shutil.copytree(lang_raw_dir, dest / "language_features_default")
    shutil.copytree(lang_ae_dir, dest / "language_features_default-language_features_dim3")

    # the training script doesn't like the frame_ prefix for AE language features
    for p in lang_ae_dir.iterdir():
        if p.is_file() and p.name.startswith("frame_"):
            p.rename(lang_ae_dir / p.name.replace("frame_", "", 1))

    # frames become rgb full res (no idea if this is needed)
    (dest / "rgb").mkdir(parents=True, exist_ok=True)
    frames_dir = src / "frames_cropped"
    shutil.copytree(frames_dir, dest / "rgb/1x")

    # frames also become images
    shutil.copytree(frames_dir, dest / "images")

if __name__ == "__main__":
    main()