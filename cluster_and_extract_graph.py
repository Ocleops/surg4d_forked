from sklearn.cluster import HDBSCAN
import torch
import argparse
import os
import numpy as np
import torchvision
from pathlib import Path
import logging
import mmcv

from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from utils.sh_utils import RGB2SH
import random
# from utils.render_utils import get_state_at_time

# from plyfile import PlyData
# import numpy as np
# import torch
from autoencoder.model import Autoencoder
import itertools
import networkx as nx
# import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Select which frame index to cluster and render
N_TIMESTEPS = 5  # change this to choose the frame index
TIMES = np.linspace(0 + 1e-6, 1 - 1e-6, N_TIMESTEPS)

def init_params():
    """Setup parameters similar to the train_eval.sh script"""
    parser = argparse.ArgumentParser()

    # these register parameters to the parser
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    # parser.add_argument("--skip_new_view", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", choices=["rgb", "lang"], default="rgb")
    parser.add_argument("--novideo", type=int, default=0)
    parser.add_argument("--noimage", type=int, default=0)
    parser.add_argument("--nonpy", type=int, default=0)
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    parser.add_argument("--num_views", type=int, default=5)

    # extra args for graph extraction
    parser.add_argument("--autoencoder_ckpt_path", type=str, help="The path to trained autoencoder checkpoint")
    parser.add_argument("--output_path", type=str, help="Where to save the output GraphML file")
    parser.add_argument("--cluster_std_thresh", type=float, default=0.1, help="The threshold for cluster-wise language feature variance used to filter clusters")
    parser.add_argument("--distance_thresh", type=float, default=0.05, help="The threshold for the distance threshold to add an edge between two objects, as measured by the Bhattacharyya distance")

    # load config file if specified
    args = parser.parse_args()
    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    return args, model_params, pipeline, hyperparam

def filter_gaussians(gaussians: GaussianModel, mask: torch.Tensor):
    """Filter set of gaussians based on a mask.

    Args:
        gaussians (GaussianModel): The gaussian model to filter.
        mask (torch.Tensor): The mask to filter the gaussians. Shape (n_gaussians,)
    """
    for prop in dir(gaussians):
        attribute = getattr(gaussians, prop)
        a_type = type(attribute)
        if a_type == torch.Tensor or a_type == torch.nn.Parameter:
            if attribute.shape[0] == len(mask):
                setattr(gaussians, prop, attribute[mask])
                logger.info(f"Filtered {prop} with shape {attribute.shape}")

def normalize_indep_dim(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def normalize_dep_dim(x):
    return (x - x.mean()) / x.std()

def positions_at_timestep(gaussians: GaussianModel, timestep: float, scene: Scene):
    with torch.no_grad():
        means3D = gaussians.get_xyz
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        lang = gaussians.get_language_feature
        # Ensure time has the same dtype/device as model tensors
        time = torch.full(
            (means3D.shape[0], 1), float(timestep), device=means3D.device, dtype=means3D.dtype
        )
        means3D_final, _, _, _, _, _, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )
    return means3D_final.detach().cpu().numpy()


def cluster_gaussians(gaussians: GaussianModel, timestep: float, scene: Scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, timestep, scene))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    # graph = build_graph(pos, lf, k=10)
    # clusters = ng_jordan_weiss_spectral_clustering(graph, min_cluster_size=100, d_spectral=10)
    clusters = HDBSCAN(min_cluster_size=100, metric="euclidean").fit_predict(
        np.concatenate([pos, lf], axis=1)
    )

    return clusters

def filter_clusters(clusters, gaussians, scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, 0.0, scene))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    i = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        opacity = gaussians.get_opacity[cluster_mask].mean()
        std_pos = pos[cluster_mask].std()
        std_lang = lf[cluster_mask].std()

        logger.info(
            f"Cluster {cluster_id}\tn_points {cluster_mask.sum()}\topacity {opacity:.4f}\tstd_pos {std_pos:.4f}\tstd_lang {std_lang:.4f}"
        )

        if cluster_id >= 0:
            # filter clusters
            if std_lang > 1.2:
                clusters[cluster_mask] = -1
                logger.info(f"\tFiltered because std_lang > 1.0")
                continue
            if opacity < 0.4:
                clusters[cluster_mask] = -1
                logger.info(f"\tFiltered because opacity < 0.5")
                continue

            # restore contiguousness of cluster ids
            clusters[cluster_mask] = i
            i += 1

def set_cluster_colors(gaussians: GaussianModel, clusters: np.ndarray):
    colors = torch.zeros_like(gaussians._features_dc)  # outliers black
    all_clusters_mask = clusters >= 0
    cluster_colors, palette = clusters_to_rgb(clusters[all_clusters_mask])
    sh_dc = RGB2SH(cluster_colors)  # (N,3)
    colors[all_clusters_mask, 0, :] = torch.tensor(sh_dc, device=colors.device, dtype=colors.dtype)
    gaussians._features_dc.data = colors  # constant part becomes cluster color
    gaussians._features_rest.data = torch.zeros_like(
        gaussians._features_rest
    )  # higher order coefficients (handle view dependence) become 0

    return palette

# def render(cam: Camera, timestep: float, gaussians: GaussianModel, pipe: PipelineParams, scene: Scene, args: argparse.Namespace, dataset: ModelParams):
#     cam.time = timestep
#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#     pkg = gs_render(
#         cam,
#         gaussians,
#         pipe,
#         background,
#         None,
#         stage=args.load_stage,
#         cam_type=scene.dataset_type,
#         args=args,
#     )
#     img = torch.clamp(pkg["render"], 0.0, 1.0)
#     return img

# def render_and_save_all(gaussians: GaussianModel, pipe: PipelineParams, scene: Scene, args: argparse.Namespace, dataset: ModelParams, out: Path):
#     save_dir = out / "cluster_renders"
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # pick random views
#     test_cams = scene.getTestCameras()
#     random_idx = random.sample(range(len(test_cams)), args.num_views)
#     cams = [test_cams[i] for i in random_idx]

#     # evenly spaced timesteps
#     timesteps = np.linspace(0, 1, args.num_views, dtype=np.float32)

#     # render and save
#     for i, cam in enumerate(cams):
#         cam_dir = save_dir / f"cam_{i:02d}"
#         cam_dir.mkdir(parents=True, exist_ok=True)
#         for j, timestep in enumerate(timesteps):
#             img = render(cam, timestep, gaussians, pipe, scene, args, dataset)
#             torchvision.utils.save_image(img, cam_dir / f"timestep_{j:02d}.png")

def bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2):
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    Sigma1, Sigma2 = np.asarray(Sigma1), np.asarray(Sigma2)

    # Average covariance
    Sigma = 0.5 * (Sigma1 + Sigma2)

    # Cholesky factorization for stability
    L = np.linalg.cholesky(Sigma)
    # Solve for (mu2 - mu1) without explicit inverse
    diff = mu2 - mu1
    sol = np.linalg.solve(L, diff)
    sol = np.linalg.solve(L.T, sol)
    term1 = 0.125 * np.dot(diff, sol)  # (1/8) Δμᵀ Σ⁻¹ Δμ

    # log-determinants via Cholesky
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
    logdet_Sigma1 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma1))))
    logdet_Sigma2 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma2))))
    term2 = 0.5 * (logdet_Sigma - 0.5 * (logdet_Sigma1 + logdet_Sigma2))

    DB = term1 + term2
    return np.exp(-DB)  # Bhattacharyya coefficient

def extract_graph(gaussians: GaussianModel,
                  clusters: np.ndarray,
                  autoencoder_ckpt_path,
                  encoder_hidden_dims,
                  decoder_hidden_dims,
                  feature_dim,
                  std_thresh: float = 0.1,
                  distance_thresh: float = 0.05):
    cluster: torch.Tensor = torch.tensor(clusters)
    pos: torch.Tensor = gaussians.get_xyz
    lf: torch.Tensor = gaussians.get_language_feature
    n_clusters = torch.unique(cluster).size
    std_metric = torch.tensor([lf[cluster == cluster_id].std() for cluster_id in range(n_clusters)])
    keep_cluster = (std_metric >= std_thresh) & (torch.arange(n_clusters) != 0)
    # cluster_filter = keep_cluster[cluster]
    n_final_clusters = int(keep_cluster.sum().item())
    means = torch.stack([pos[cluster==i].mean(dim=0) for i in range(n_clusters) if keep_cluster[i]])
    covs = torch.stack([torch.cov(pos[cluster==i].T) for i in range(n_clusters) if keep_cluster[i]])
    mean_lfs_latent = torch.stack([lf[cluster==i].mean(dim=0) for i in range(n_clusters) if keep_cluster[i]])

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=feature_dim)
    model.load_state_dict(torch.load(autoencoder_ckpt_path, map_location='cuda'))
    model.eval()

    with torch.no_grad():
        decoded_features = model.decode(torch.tensor(mean_lfs_latent, dtype=torch.float32))
        mean_lfs = decoded_features

    distances = np.zeros((n_final_clusters, n_final_clusters))
    for i, j in itertools.combinations(range(n_final_clusters), 2):
        distances[i, j] = bhattacharyya_coefficient(means[i], covs[i], means[j], covs[j])

    edges = np.indices((n_final_clusters, n_final_clusters)).transpose((1, 2, 0))[distances >= distance_thresh]

    G = nx.Graph()
    for i, (p, l) in enumerate(zip(means, mean_lfs)):
        # G.add_node(i, pos_x=p[0], pos_y=p[1], pos_z=p[2],
        #               lang_feat_0=l[0], lang_feat_1=l[1], lang_feat_2=l[2])
        G.add_node(i, pos_x=p[0], pos_y=p[1], pos_z=p[2],
                **{f'lang_feat_{j}' : f for j, f in enumerate(l)})

    for u, v in edges:
        G.add_edge(u, v)
    
    return G

def main():
    # determistic seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # mock render.py config
    args, model_params, pipeline, hyperparam = init_params()

    # construct all objects
    dataset = model_params.extract(args)
    pipe = pipeline.extract(args)
    hyper = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)  # type:ignore
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        load_stage=args.load_stage,
    )

    # filter gaussians, cluster, filter clusters, set cluster colors
    mask = (gaussians.get_opacity > 0.1).squeeze()
    filter_gaussians(gaussians, mask)
    clusters = cluster_gaussians(gaussians, timestep=0.0, scene=scene)
    filter_clusters(clusters, gaussians, scene)
    palette = set_cluster_colors(gaussians, clusters)


    graph = extract_graph(gaussians,
                          clusters,
                          args.autoencoder_ckpt_path,
                          [256, 128, 64, 32, 3],
                          [16, 32, 64, 128, 256, 512],
                          512,
                          std_thresh = args.std_thresh,
                          distance_thresh = args.distance_thresh)
    nx.write_graphml(graph, args.output_path)
    print('\033[1m' + f"Wrote scene graph to {args.output_path}" + '\033[0m')

    # render and save everything
    # out = Path(args.model_path) / "graph"
    # out.mkdir(parents=True, exist_ok=True)
    # render_and_save_all(gaussians, pipe, scene, args, dataset, out)
    # store_palette(palette, out / "cluster_palette.png")
    # gaussians.save_ply(out / "clustered_gaussians.ply")

if __name__ == "__main__":
    main()