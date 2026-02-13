from pathlib import Path
import random

import hydra
import numpy as np
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch
from loguru import logger
from umap import UMAP
import rerun as rr

from autoencoder.model_qwen import QwenAutoencoder


# Output folder for all scene visualizations (edit this as needed).
OUTPUT_DIR = Path("output/ae_unitsphere_viz")

INCLUDE_BACKGROUND = False
ENCODE_BATCH_SIZE = 8192
UMAP_NEIGHBORS = 60
UMAP_MIN_DIST = 0.02
MAX_UMAP_POINTS = 8000

CLASS_ID_TO_NAME = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

CLASS_COLORS = {
    0: "#1f1f1f",
    1: "#e6194b",
    2: "#8B4513",
    3: "#ffe119",
    4: "#fabebe",
    5: "#4363d8",
    6: "#f58231",
    7: "#911eb4",
    8: "#46f0f0",
    9: "#3cb44b",
    10: "#f032e6",
    11: "#bcf60c",
    12: "#9a6324",
}


def majority_class_per_patch(
    patch_map: np.ndarray,
    semantic_mask: np.ndarray,
    n_patches: int,
) -> np.ndarray:
    patch_ids = patch_map[0].reshape(-1).astype(np.int64)
    semantic_ids = semantic_mask.reshape(-1).astype(np.int64)

    assert patch_ids.max() < n_patches, "Patch map references patch id outside feature array."
    assert semantic_ids.min() >= 0, "Semantic mask contains negative class ids."

    n_classes = int(semantic_ids.max()) + 1
    class_counts = np.zeros(
        (n_patches, n_classes),
        dtype=np.int32,
    )
    np.add.at(class_counts, (patch_ids, semantic_ids), 1)
    return class_counts.argmax(axis=1)


def load_scene_patch_features_and_labels(
    clip: DictConfig,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    patch_feat_dir = clip_dir / cfg.autoencoder.patch_feat_subdir
    semantic_mask_dir = clip_dir / cfg.preprocessing.semantic_mask_subdir

    feature_files = sorted(patch_feat_dir.glob("*_f.npy"))
    assert len(feature_files) > 0, f"No patch features found in {patch_feat_dir}"

    all_features = []
    all_labels = []
    for feature_file in feature_files:
        frame_stem = feature_file.stem.replace("_f", "")
        patch_features = np.load(feature_file)
        patch_map = np.load(patch_feat_dir / f"{frame_stem}_s.npy")
        semantic_mask = np.load(semantic_mask_dir / f"frame_{frame_stem}.npy")

        patch_labels = majority_class_per_patch(
            patch_map=patch_map,
            semantic_mask=semantic_mask,
            n_patches=patch_features.shape[0],
        )

        all_features.append(patch_features)
        all_labels.append(patch_labels)

    scene_features = np.concatenate(all_features, axis=0).astype(np.float32)
    scene_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    unknown_class_ids = sorted(set(np.unique(scene_labels).tolist()) - set(CLASS_ID_TO_NAME.keys()))
    if len(unknown_class_ids) > 0:
        logger.warning(f"{clip.name}: found unknown semantic class ids {unknown_class_ids}")
    return scene_features, scene_labels


def encode_to_highdim_unitsphere(
    features: np.ndarray,
    autoencoder: QwenAutoencoder,
    device: torch.device,
) -> np.ndarray:
    latent_batches = []
    with torch.no_grad():
        for start_idx in range(0, features.shape[0], ENCODE_BATCH_SIZE):
            end_idx = start_idx + ENCODE_BATCH_SIZE
            feat_batch = torch.from_numpy(features[start_idx:end_idx]).to(
                device=device,
                dtype=torch.float32,
            )
            latent_batch = autoencoder.encode(feat_batch).detach().cpu().numpy()
            latent_batches.append(latent_batch)

    latents = np.concatenate(latent_batches, axis=0)
    return latents


def get_sampling_indices(
    n_samples: int,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Get indices for sampling data for visualization.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    labels : np.ndarray, shape (n_samples,)
        Semantic class IDs
        
    Returns
    -------
    sampled_indices : np.ndarray
        Indices to use for sampling
    """
    if n_samples <= MAX_UMAP_POINTS:
        return np.arange(n_samples)

    class_ids = np.unique(labels)
    per_class_indices = []
    for class_id in class_ids:
        class_idx = np.where(labels == class_id)[0]
        target = max(1, int(np.round(class_idx.shape[0] / n_samples * MAX_UMAP_POINTS)))
        if target >= class_idx.shape[0]:
            per_class_indices.append(class_idx)
        else:
            selected = np.random.choice(class_idx, size=target, replace=False)
            per_class_indices.append(selected)

    sampled_indices = np.concatenate(per_class_indices, axis=0)
    if sampled_indices.shape[0] > MAX_UMAP_POINTS:
        sampled_indices = np.random.choice(sampled_indices, size=MAX_UMAP_POINTS, replace=False)
    sampled_indices.sort()
    return sampled_indices


def sample_for_umap(
    latents: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_indices = get_sampling_indices(latents.shape[0], labels)
    return latents[sampled_indices], labels[sampled_indices]


def spherical_haversine_distance_matrix(latents: np.ndarray) -> np.ndarray:
    # Inputs are already L2-normalized by encoder. Great-circle distance:
    # d(x, y) = 2 * arcsin(||x - y|| / 2), equivalent to angular distance.
    dot = np.clip(latents @ latents.T, -1.0, 1.0)
    chord_sq = np.clip(2.0 - 2.0 * dot, 0.0, 4.0)
    half_chord = 0.5 * np.sqrt(chord_sq)
    return 2.0 * np.arcsin(np.clip(half_chord, 0.0, 1.0))


def umap_to_3d_unitsphere(latents: np.ndarray) -> np.ndarray:
    dist_matrix = spherical_haversine_distance_matrix(latents)
    reducer = UMAP(
        n_components=3,
        metric="precomputed",
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=42,
    )
    embedded = reducer.fit_transform(dist_matrix)
    embedded /= np.linalg.norm(embedded, axis=1, keepdims=True)
    return embedded


def plot_scene_unitsphere(
    clip_name: str,
    points_3d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    class_ids = np.unique(labels)
    if not INCLUDE_BACKGROUND:
        class_ids = class_ids[class_ids != 0]

    for class_id in class_ids:
        class_mask = labels == class_id
        class_points = points_3d[class_mask]
        class_name = CLASS_ID_TO_NAME.get(int(class_id), f"Unknown-{int(class_id)}")
        class_color = CLASS_COLORS.get(int(class_id), plt.cm.tab20(int(class_id) % 20))
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            class_points[:, 2],
            color=class_color,
            s=3,
            alpha=0.65,
            label=f"{class_id}: {class_name} ({class_points.shape[0]})",
        )

    # Reference unit sphere wireframe to visually verify spherical support.
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.4)

    # Enforce equal axis scaling; otherwise matplotlib distorts 3D geometry.
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_title(f"{clip_name} - AE latent UMAP (3D unit sphere)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_tsne_2d(
    latents: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """
    Compute 2D t-SNE embedding of latents.
    
    Parameters
    ----------
    latents : np.ndarray, shape (n_samples, n_features)
        High-dimensional latent vectors
    random_state : int
        Random seed for reproducibility
    perplexity : float
        Perplexity parameter for t-SNE
        
    Returns
    -------
    embedding : np.ndarray, shape (n_samples, 2)
        2D t-SNE embedding
    """
    # Adjust perplexity if we have fewer samples than default
    n_samples = latents.shape[0]
    max_perplexity = min(perplexity, (n_samples - 1) / 3)
    
    reducer = TSNE(
        n_components=2,
        perplexity=max_perplexity,
        random_state=random_state,
        max_iter=1000,
        init="pca",
    )
    embedded = reducer.fit_transform(latents)
    return embedded


def plot_tsne_2d(
    clip_name: str,
    tsne_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    data_type: str = "AE Latents",
) -> None:
    """
    Plot 2D t-SNE embedding colored by semantic class with legend.
    
    Parameters
    ----------
    clip_name : str
        Name of the clip
    tsne_2d : np.ndarray, shape (n_points, 2)
        2D t-SNE coordinates
    labels : np.ndarray, shape (n_points,)
        Semantic class IDs
    output_path : Path
        Output file path for the plot
    data_type : str
        Type of data being visualized (e.g., "AE Latents" or "Original Features")
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_ids = np.unique(labels)
    if not INCLUDE_BACKGROUND:
        class_ids = class_ids[class_ids != 0]
    
    for class_id in class_ids:
        class_mask = labels == class_id
        class_points = tsne_2d[class_mask]
        class_name = CLASS_ID_TO_NAME.get(int(class_id), f"Unknown-{int(class_id)}")
        class_color = CLASS_COLORS.get(int(class_id), plt.cm.tab20(int(class_id) % 20))
        
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            color=class_color,
            s=8,
            alpha=0.6,
            label=f"{class_id}: {class_name} ({class_points.shape[0]})",
        )
    
    ax.set_title(f"{clip_name} - t-SNE 2D Embedding of {data_type}", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_autoencoder_for_clip(
    clip: DictConfig,
    cfg: DictConfig,
    device: torch.device,
    global_autoencoder: QwenAutoencoder | None,
) -> QwenAutoencoder:
    if cfg.autoencoder.global_mode:
        assert global_autoencoder is not None, "Global autoencoder expected but not loaded."
        return global_autoencoder

    clip_checkpoint = (
        Path(cfg.preprocessed_root)
        / clip.name
        / cfg.autoencoder.checkpoint_subdir
        / "best_ckpt.pth"
    )
    autoencoder = QwenAutoencoder(
        input_dim=cfg.autoencoder.full_dim,
        latent_dim=cfg.autoencoder.latent_dim,
    ).to(device)
    autoencoder.load_state_dict(torch.load(clip_checkpoint, map_location=device))
    autoencoder.eval()
    return autoencoder

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple (0-255)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def log_unitsphere_to_rerun(
    points_3d: np.ndarray,
    labels: np.ndarray,
    clip_name: str,
    output_rrd_path: Path,
) -> None:
    """
    Log 3D unit sphere points to Rerun, with each semantic class as a separate point cloud.
    
    Parameters
    ----------
    points_3d : np.ndarray, shape (n_points, 3)
        3D coordinates on unit sphere
    labels : np.ndarray, shape (n_points,)
        Semantic class IDs
    clip_name : str
        Name of the clip for logging
    output_rrd_path : Path
        Path to save .rrd file
    """
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points_3d, axis=1, keepdims=True)
    points_3d_normalized = points_3d / (norms + 1e-10)
    
    # Initialize rerun
    rr.init(f"ae_unitsphere_{clip_name}")
    rr.save(str(output_rrd_path))
    
    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Compute scene extent for point radius
    scene_extent = 2.0  # Unit sphere has diameter 2
    point_radius = max(scene_extent * 0.005, 1e-5)
    
    class_ids = np.unique(labels)
    if not INCLUDE_BACKGROUND:
        class_ids = class_ids[class_ids != 0]
    
    for class_id in class_ids:
        class_mask = labels == class_id
        class_points = points_3d_normalized[class_mask]
        
        if class_points.shape[0] == 0:
            continue
        
        class_name = CLASS_ID_TO_NAME.get(int(class_id), f"Unknown-{int(class_id)}")
        # Sanitize entity path (replace spaces and special chars)
        safe_name = class_name.replace(" ", "_").replace("-", "_")
        entity_path = f"world/class_{int(class_id):02d}_{safe_name}"
        
        # Get color for this class
        hex_color = CLASS_COLORS.get(int(class_id), "#808080")
        rgb_color = np.array(hex_to_rgb(hex_color), dtype=np.uint8)
        
        # Create color array (same color for all points in this class)
        class_colors = np.tile(rgb_color, (class_points.shape[0], 1))
        
        # Log point cloud
        rr.log(
            entity_path,
            rr.Points3D(
                positions=class_points,
                colors=class_colors,
                radii=point_radius,
            ),
        )
        
        logger.info(f"Logged {class_points.shape[0]} points for class {int(class_id)} ({class_name}) to {entity_path}")


def landmark_cosine_mds(X, n_components=2, n_landmarks=1000, random_state=None):
    """
    Fast Landmark MDS that preserves absolute cosine distances.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=2
        Number of dimensions for the embedding
    n_landmarks : int, default=1000
        Number of landmark points to use
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Returns
    -------
    embedding : array, shape (n_samples, n_components)
        The embedded coordinates
    """
    rng = check_random_state(random_state)
    n_samples = X.shape[0]
    
    # Select landmark indices
    n_landmarks = min(n_landmarks, n_samples)
    landmark_indices = rng.choice(n_samples, size=n_landmarks, replace=False)
    landmarks = X[landmark_indices]
    
    # Normalize for cosine distance
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    landmarks_norm = landmarks / (np.linalg.norm(landmarks, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarities efficiently
    cos_sim_landmarks = landmarks_norm @ landmarks_norm.T
    cos_sim_all = X_norm @ landmarks_norm.T
    
    # Convert to absolute cosine distances
    D_landmarks = np.abs(1 - cos_sim_landmarks)
    D_all = np.abs(1 - cos_sim_all)
    
    # Classical MDS on landmarks
    # Double centering
    n = n_landmarks
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_landmarks ** 2) @ H
    
    # Eigendecomposition (only top k)
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort in descending order and take top n_components
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Handle negative eigenvalues
    eigvals = np.maximum(eigvals, 0)
    
    # Landmark embedding
    Y_landmarks = eigvecs * np.sqrt(eigvals)
    
    # Triangulate all points using Nyström-like extension
    D_landmarks_mean = D_landmarks.mean(axis=1)
    Y_all = -0.5 * (D_all ** 2 - D_landmarks_mean) @ np.linalg.pinv(Y_landmarks.T)
    
    return Y_all

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, OUTPUT_DIR / "resolved_config.yaml")

    global_autoencoder = None
    if cfg.autoencoder.global_mode:
        checkpoint_path = (
            Path(cfg.preprocessed_root) / cfg.autoencoder.global_checkpoint_dir / "best_ckpt.pth"
        )
        global_autoencoder = QwenAutoencoder(
            input_dim=cfg.autoencoder.full_dim,
            latent_dim=cfg.autoencoder.latent_dim,
        ).to(device)
        global_autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        global_autoencoder.eval()

    for clip in cfg.clips:
        logger.info(f"Processing scene {clip.name}")
        scene_dir = OUTPUT_DIR / clip.name
        scene_dir.mkdir(parents=True, exist_ok=True)

        features, labels = load_scene_patch_features_and_labels(clip, cfg)
        autoencoder = load_autoencoder_for_clip(
            clip=clip,
            cfg=cfg,
            device=device,
            global_autoencoder=global_autoencoder,
        )
        latents = encode_to_highdim_unitsphere(features, autoencoder, device=device)
        
        # Get sampling indices to use for both features and latents
        sampled_indices = get_sampling_indices(features.shape[0], labels)
        features_umap = features[sampled_indices]
        latents_umap = latents[sampled_indices]
        labels_umap = labels[sampled_indices]
        
        points_3d = landmark_cosine_mds(latents_umap, n_components=3)

        np.save(scene_dir / "encoded_latents.npy", latents)
        np.save(scene_dir / "labels.npy", labels)
        np.save(scene_dir / "features_umap.npy", features_umap)
        np.save(scene_dir / "encoded_latents_umap.npy", latents_umap)
        np.save(scene_dir / "labels_umap.npy", labels_umap)
        np.save(scene_dir / "umap_unitsphere_points.npy", points_3d)
        plot_scene_unitsphere(
            clip_name=clip.name,
            points_3d=points_3d,
            labels=labels_umap,
            output_path=scene_dir / "umap_unitsphere.png",
        )
        
        # Log to rerun
        rrd_path = scene_dir / "unitsphere.rrd"
        log_unitsphere_to_rerun(
            points_3d=points_3d,
            labels=labels_umap,
            clip_name=clip.name,
            output_rrd_path=rrd_path,
        )
        logger.info(f"Saved Rerun visualization to: {rrd_path}")
        
        # Compute and plot 2D t-SNE for latents
        logger.info("Computing 2D t-SNE embedding of latents...")
        tsne_2d_latents = compute_tsne_2d(latents_umap, random_state=42)
        np.save(scene_dir / "tsne_2d_latents.npy", tsne_2d_latents)
        plot_tsne_2d(
            clip_name=clip.name,
            tsne_2d=tsne_2d_latents,
            labels=labels_umap,
            output_path=scene_dir / "tsne_2d_latents.png",
            data_type="AE Latents",
        )
        logger.info(f"Saved 2D t-SNE visualization of latents to: {scene_dir / 'tsne_2d_latents.png'}")
        
        # Compute and plot 2D t-SNE for original features
        logger.info("Computing 2D t-SNE embedding of original features...")
        tsne_2d_features = compute_tsne_2d(features_umap, random_state=42)
        np.save(scene_dir / "tsne_2d_features.npy", tsne_2d_features)
        plot_tsne_2d(
            clip_name=clip.name,
            tsne_2d=tsne_2d_features,
            labels=labels_umap,
            output_path=scene_dir / "tsne_2d_features.png",
            data_type="Original Features",
        )
        logger.info(f"Saved 2D t-SNE visualization of features to: {scene_dir / 'tsne_2d_features.png'}")

        counts = {
            str(class_id): int((labels == class_id).sum())
            for class_id in np.unique(labels)
            if INCLUDE_BACKGROUND or class_id != 0
        }
        with open(scene_dir / "class_counts.txt", "w") as f:
            for class_id, count in counts.items():
                class_name = CLASS_ID_TO_NAME.get(int(class_id), f"Unknown-{int(class_id)}")
                f.write(f"{class_id}: {class_name} -> {count}\n")

        logger.info(
            f"{clip.name}: encoded {latents.shape[0]} patches, projected {points_3d.shape[0]} sampled patches to 3D sphere, output -> {scene_dir}"
        )


if __name__ == "__main__":
    main()
