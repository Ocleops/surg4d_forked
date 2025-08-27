import argparse
from pathlib import Path
import numpy as np
import sys

# Add nerfstudio colmap utils to path
try:
    import nerfstudio.data.utils.colmap_parsing_utils as colmap_utils
except ImportError:
    ns_path = "/home/tumai/miniconda3/envs/4DSplat/lib/python3.10/site-packages"
    if ns_path not in sys.path:
        sys.path.append(ns_path)
    import nerfstudio.data.utils.colmap_parsing_utils as colmap_utils


def tum_to_matrix(poses_tum):
    """Convert a list of TUM-style poses [tx, ty, tz, qx, qy, qz, qw] to 4x4 matrices."""
    matrices = []
    for pose in poses_tum:
        t = pose[0:3]
        q = pose[3:7]  # qx, qy, qz, qw

        # COLMAP uses qw, qx, qy, qz, but SciPy/our function uses qx, qy, qz, qw.
        # The TUM loader here assumes tx ty tz qx qy qz qw, so we need to create a w,x,y,z quat for rotmat2qvec
        # np.roll shifts the last element (qw) to the first position
        q_wxyz = np.roll(q, 1)

        R = colmap_utils.qvec2rotmat(q_wxyz)

        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        matrices.append(mat)
    return np.array(matrices)


def colmap_to_matrix(colmap_images):
    """Convert COLMAP Image objects to a list of 4x4 world-to-camera matrices."""
    matrices = []
    # Sort by image_id to ensure order
    sorted_images = sorted(colmap_images.values(), key=lambda img: img.id)
    for img in sorted_images:
        R = colmap_utils.qvec2rotmat(img.qvec)
        t = img.tvec

        # This is a world-to-camera matrix
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        matrices.append(mat)
    return np.array(matrices)


def umeyama_align(model, data):
    """
    Computes the optimal similarity transform between two sets of 3D points.
    See: https://gist.github.com/laixintao/d96c9e12b2c864f89551578f3521f52a
    """
    # Mean center the clouds
    model_mean = model.mean(axis=0)
    data_mean = data.mean(axis=0)
    model_centered = model - model_mean
    data_centered = data - data_mean

    # Covariance matrix
    C = data_centered.T @ model_centered

    # SVD
    U, D, V_t = np.linalg.svd(C)
    V = V_t.T

    # Rotation
    R = V @ U.T

    # Handle special reflection case
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    # Scale
    var_model = np.var(model_centered, axis=0).sum()
    s = sum(D) / var_model

    # Translation
    t = data_mean - s * R @ model_mean

    return R, s, t


def compare_poses(monst3r_dir: Path, colmap_sparse_dir: Path):
    # 1. Load MonST3r poses from pred_traj.txt (TUM format)
    traj_path = monst3r_dir / "pred_traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"MonST3r trajectory file not found: {traj_path}")

    # Loads as [ts, tx, ty, tz, qx, qy, qz, qw]
    monst3r_data = np.loadtxt(traj_path)
    monst3r_poses_tum = monst3r_data[:, 1:]

    # Convert to camera-to-world matrices
    monst3r_c2w = tum_to_matrix(monst3r_poses_tum)
    monst3r_positions = monst3r_c2w[:, :3, 3]

    # 2. Load COLMAP poses
    images_bin_path = colmap_sparse_dir / "images.bin"
    if not images_bin_path.exists():
        raise FileNotFoundError(f"COLMAP images.bin not found: {images_bin_path}")
    colmap_images = colmap_utils.read_images_binary(images_bin_path)

    # Convert to world-to-camera matrices and then invert to get camera-to-world
    colmap_w2c = colmap_to_matrix(colmap_images)
    colmap_c2w = np.linalg.inv(colmap_w2c)
    colmap_positions = colmap_c2w[:, :3, 3]

    # Ensure same number of poses
    num_poses = min(len(monst3r_positions), len(colmap_positions))
    monst3r_positions = monst3r_positions[:num_poses]
    colmap_positions = colmap_positions[:num_poses]
    print(f"Comparing {num_poses} poses...")

    # 3. Align trajectories using Umeyama on the camera positions
    R, s, t = umeyama_align(monst3r_positions, colmap_positions)

    monst3r_positions_aligned = s * R @ monst3r_positions.T + t[:, np.newaxis]
    monst3r_positions_aligned = monst3r_positions_aligned.T

    # 4. Compare and Report
    translation_errors = np.linalg.norm(
        monst3r_positions_aligned - colmap_positions, axis=1
    )

    print("\n--- Pose Comparison ---")
    print(f"Alignment scale (s): {s:.4f}")
    print(f"Alignment rotation (R):\n{R}")
    print(f"Alignment translation (t): {t}")
    print("\n--- Translation Error (after alignment) ---")
    print(f"Mean Error:   {np.mean(translation_errors):.4f}")
    print(f"Median Error: {np.median(translation_errors):.4f}")
    print(f"Max Error:    {np.max(translation_errors):.4f}")

    print("\nError for first 5 frames:")
    for i in range(min(5, num_poses)):
        print(f"  Frame {i}: {translation_errors[i]:.4f}")

    print("\nConclusion:")
    if np.median(translation_errors) < 0.1:  # Heuristic threshold
        print("Poses appear to be comparable and consistent after alignment.")
    else:
        print(
            "Warning: Poses have significant differences even after alignment. Check coordinate system conventions."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MonST3r and COLMAP camera poses."
    )
    parser.add_argument(
        "--monst3r_dir",
        type=str,
        required=True,
        help="Path to the MonST3r output directory.",
    )
    parser.add_argument(
        "--colmap_sparse_dir",
        type=str,
        required=True,
        help="Path to the COLMAP sparse model directory.",
    )
    args = parser.parse_args()

    compare_poses(Path(args.monst3r_dir), Path(args.colmap_sparse_dir))
