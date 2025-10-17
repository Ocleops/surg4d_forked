from scene.gaussian_model import GaussianModel
from pathlib import Path
from extract_graph import init_params, load_all_models, filter_gaussians
from gaussian_renderer import render as gs_render
import argparse
from arguments import ModelParams, PipelineParams, ModelHiddenParams
import sys
import torch

N_TIMESTEPS = 80
out_path = Path('splat_viz')

# ------------- input arguments ----------------------
clip_name = "video27_00480_qwen_cat_depthl2_nods"
frame = 50
exp_pc_name = "pointcloud.ply"

# ----------------------------------------------------
timestep = frame / (N_TIMESTEPS - 1)
language_feature_name = "qwen_cat_features_dim6"
video_name = clip_name[:7]
clip_prefix = clip_name[:13]

import os
# Simulate: export language_feature_hiddendim=${clip_feat_dim}
clip_feat_dim = '6'
os.environ['language_feature_hiddendim'] = clip_feat_dim
# manual argv override (copied from the setattr values)
sys.argv = [
    "splat_viz.py",
    "-s", f"data/cholecseg8k/preprocessed_ssg/{video_name}/{clip_prefix}",
    "--language_features_name", language_feature_name,
    "--model_path", f"output/cholecseg8k/{clip_name}",
    "--feature_level", "0",
    "--skip_train",
    "--skip_test",
    "--configs", "arguments/cholecseg8k/no_tv.py",
    "--mode", "lang",
    "--no_dlang", "1",
    "--load_stage", "fine-lang",
    "--num_views", "5",
    "--qwen_autoencoder_ckpt_path", f"data/cholecseg8k/preprocessed_ssg/{video_name}/{clip_prefix}/autoencoder/best_ckpt.pth",
]
args, model_params, pipeline, hyperparam = init_params()
gaussians, scene, dataset = load_all_models(
    args, model_params, pipeline, hyperparam
)
time = torch.full(
    (gaussians.get_xyz.shape[0], 1),
    float(timestep),
    device=gaussians.get_xyz.device,
    dtype=gaussians.get_xyz.dtype,
)
xyz, scaling, rotation, _, _, _, _ = gaussians._deformation(gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, gaussians.get_language_feature, time)

gaussians._xyz, gaussians._scaling, gaussians._rotation = xyz, scaling, rotation

# ------------- custom overrides ---------------------

# gaussians._opacity = torch.full_like(gaussians._opacity, 100)

# ----------------------------------------------------

gaussians.save_ply(out_path / exp_pc_name)
