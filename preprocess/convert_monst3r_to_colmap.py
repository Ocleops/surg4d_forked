import argparse
from pathlib import Path
import numpy as np
import os
import sys

# Add nerfstudio colmap utils to path to use their writer
# This is a bit hacky, but avoids duplicating the code
try:
    import nerfstudio.data.utils.colmap_parsing_utils as colmap_utils
except ImportError:
    print("Could not import nerfstudio. Please ensure it is installed and in your python path.")
    # A common path might be this, let's try to add it
    ns_path = "/home/tumai/miniconda3/envs/4DSplat/lib/python3.10/site-packages"
    if ns_path not in sys.path:
        sys.path.append(ns_path)
    import nerfstudio.data.utils.colmap_parsing_utils as colmap_utils


def convert_monst3r_to_colmap(monst3r_dir: Path, images_dir: Path, output_dir: Path):
    """
    Converts MonST3R trajectory and intrinsics files to a COLMAP sparse model.

    Args:
        monst3r_dir: Directory containing pred_traj.txt and pred_intrinsics.txt
        images_dir: Directory containing the corresponding images.
        output_dir: Directory to save the COLMAP sparse model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read intrinsics
    intrinsics_path = monst3r_dir / "pred_intrinsics.txt"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
    intrinsics = np.loadtxt(intrinsics_path)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # 2. Read trajectory (camera-to-world matrices)
    traj_path = monst3r_dir / "pred_traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    # Each 4 lines is a 4x4 matrix
    poses = np.loadtxt(traj_path).reshape(-1, 4, 4)

    # 3. Get image list and dimensions
    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
        
    if len(poses) != len(image_paths):
        print(f"Warning: Mismatch between number of poses ({len(poses)}) and images ({len(image_paths)}).")
        # Truncate to the shorter length
        min_len = min(len(poses), len(image_paths))
        poses = poses[:min_len]
        image_paths = image_paths[:min_len]

    # Get image dimensions from the first image
    import cv2
    first_img = cv2.imread(str(image_paths[0]))
    height, width, _ = first_img.shape

    # 4. Create COLMAP Camera object
    # We'll use a simple Pinhole camera model
    # params = [f, cx, cy] for SIMPLE_PINHOLE
    # params = [fx, fy, cx, cy] for PINHOLE
    cam = colmap_utils.Camera(id=1, model="PINHOLE", width=width, height=height, params=np.array([fx, fy, cx, cy]))
    cameras = {1: cam}

    # 5. Create COLMAP Image objects
    images = {}
    for i, (pose_c2w, img_path) in enumerate(zip(poses, image_paths)):
        image_id = i + 1
        
        # COLMAP requires world-to-camera matrices
        pose_w2c = np.linalg.inv(pose_c2w)
        
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3, 3]

        qvec = colmap_utils.rotmat2qvec(R)
        
        # xys and point3D_ids are empty since we have no 2D-3D correspondences
        images[image_id] = colmap_utils.Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=1,
            name=img_path.name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.full(0, -1, dtype=np.int64),
        )

    # 6. Write binary files
    print(f"Writing cameras.bin to {output_dir / 'cameras.bin'}")
    colmap_utils.write_cameras_binary(cameras, output_dir / "cameras.bin")
    
    print(f"Writing images.bin to {output_dir / 'images.bin'}")
    colmap_utils.write_images_binary(images, output_dir / "images.bin")
    
    # Create an empty points3D.bin
    print(f"Writing empty points3D.bin to {output_dir / 'points3D.bin'}")
    colmap_utils.write_points3D_binary({}, output_dir / "points3D.bin")
    
    print("\nConversion successful!")
    print(f"COLMAP sparse model created at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MonST3R output to a COLMAP sparse model.")
    parser.add_argument("--monst3r_dir", type=str, required=True, help="Path to the MonST3R output directory (containing pred_traj.txt and pred_intrinsics.txt).")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory with the corresponding images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the COLMAP sparse model.")
    args = parser.parse_args()
    
    convert_monst3r_to_colmap(Path(args.monst3r_dir), Path(args.images_dir), Path(args.output_dir))
