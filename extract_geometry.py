from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
import torch
from loguru import logger
import cv2
import re
from depth_anything_3.api import DepthAnything3

from da3_utils import da3_to_multi_view_colmap, filter_prediction_edge_artifacts


def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def delete_unused_files(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    # colmap txt version
    if (clip_dir / "cameras.txt").exists():
        (clip_dir / "cameras.txt").unlink()
    if (clip_dir / "images.txt").exists():
        (clip_dir / "images.txt").unlink()
    if (clip_dir / "points3D.txt").exists():
        (clip_dir / "points3D.txt").unlink()


def da3(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    images_dir = clip_dir / "images"

    # load da3 model
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
    model = model.to("cuda:0")

    # construct image paths and determine processing resolution close to orig
    image_filenames = sorted(
        list(images_dir.glob("*.png")), key=extract_frame_number
    )
    image_filenames = [str(img_file) for img_file in image_filenames]
    orig_w, orig_h = Image.open(image_filenames[0]).size
    processing_res = max(orig_w, orig_h)

    # da3 inference
    prediction = model.inference(
        image=image_filenames,
        ref_view_strategy="middle",  # good for video according to docs
        process_res=processing_res,
        process_res_method="upper_bound_resize",
    )

    # Apply edge filtering to prediction at processed resolution (if configured)
    edge_gradient_threshold = cfg.extract_geometry.da3_edge_gradient_threshold
    if edge_gradient_threshold is not None:
        logger.info(f"Applying depth edge filtering with threshold: {edge_gradient_threshold}")
        prediction = filter_prediction_edge_artifacts(
            prediction,
            gradient_threshold=edge_gradient_threshold,
        )

    # dump to colmap with pc consisting of multi-frame depth projection (first, middle, last)
    colmap_dir = clip_dir / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Use first, middle, and last frames for initialization
    num_frames_total = len(prediction.depth)
    init_frame_indices = [0, num_frames_total // 2, num_frames_total - 1]
    logger.info(f"Initializing point cloud from frames: {init_frame_indices}")
    
    view_point_counts = da3_to_multi_view_colmap(
        prediction,
        colmap_dir,
        image_filenames,
        view_indices=init_frame_indices,
        conf_thresh_percentile=cfg.extract_geometry.da3_conf_thresh_percentile,
        pixel_stride=cfg.extract_geometry.da3_pc_pixel_stride,
        densify_ratio=cfg.extract_geometry.da3_densify_ratio,
    )
    logger.info(f"Point counts per view: {dict(zip(init_frame_indices, view_point_counts))}")

    # Store depth maps at both processed and original resolution
    # Processed resolution: used by point cloud and cotracker (guarantees consistency)
    # Original resolution: available for other downstream uses like depth loss supervision for rendered depths in original resolution
    depth_dir = clip_dir / cfg.extract_geometry.depth_subdir
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_processed_dir = clip_dir / cfg.extract_geometry.depth_processed_subdir
    depth_processed_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir = clip_dir / cfg.extract_geometry.confidence_subdir
    confidence_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(prediction.depth)
    for frame_idx in range(num_frames):
        depth_proc = prediction.depth[frame_idx]
        
        # Save at processed resolution (for point cloud / cotracker consistency)
        depth_processed_path = depth_processed_dir / f"{frame_idx:06d}.npy"
        np.save(depth_processed_path, depth_proc)
        
        # Save at original resolution (for other uses)
        depth_orig = cv2.resize(
            depth_proc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        depth_path = depth_dir / f"{frame_idx:06d}.npy"
        np.save(depth_path, depth_orig)
        
        # Save confidence at original resolution
        confidence_proc = prediction.conf[frame_idx]
        confidence_orig = cv2.resize(
            confidence_proc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        confidence_path = confidence_dir / f"{frame_idx:06d}.npy"
        np.save(confidence_path, confidence_orig)


    # Clean up GPU memory from da3 model
    del model
    del prediction
    torch.cuda.empty_cache()


def extract_geometry(clip: DictConfig, cfg: DictConfig):
    if not cfg.extract_geometry.only_update_annotations:
        da3(clip, cfg)

        if not cfg.extract_geometry.verbose_output:
            delete_unused_files(clip, cfg)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        extract_geometry(clip, cfg)


if __name__ == "__main__":
    main()
