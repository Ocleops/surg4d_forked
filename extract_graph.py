from plyfile import PlyData
import numpy as np
import torch
from autoencoder.model import Autoencoder
import itertools
import networkx as nx
import argparse


parser = argparse.ArgumentParser(description="ArgParser for the Graph Extraction script")
parser.add_argument("pointcloud_path", type=str, help="The path to the language-clustered point cloud file returned by the clustering script")
parser.add_argument("autoencoder_ckpt_path", type=str, help="The path to trained autoencoder checkpoint")
parser.add_argument("output_path", type=str, help="Where to save the output GraphML file")

parser.add_argument("--extend_ply", default=False, action='store_true')
parser.add_argument("--cluster_std_thresh", type=float, default=0.1, help="The threshold for cluster-wise language feature variance used to filter clusters")
parser.add_argument("--distance_thresh", type=float, default=0.05, help="The threshold for the distance threshold to add an edge between two objects, as measured by the Bhattacharyya distance")

args = parser.parse_args()

in_filepath: str = args.pointcloud_path
ckpt_path: str = args.autoencoder_ckpt_path
out_filepath: str = args.output_path
std_thresh: float = args.cluster_std_thresh
distance_thresh: float = args.distance_thresh


# ---------------------------------------------------------------------
#              DATA EXTRACTION & CLUSTER AGGREGATION

# Load the PLY file
plydata = PlyData.read(in_filepath)

# Get the vertex data (structured array)
vertex_data = plydata['vertex'].data

# Extract specific attributes into NumPy arrays
x = np.array(vertex_data['x'])
y = np.array(vertex_data['y'])
z = np.array(vertex_data['z'])

f_lang_0 = np.array(vertex_data['f_lang_0'])
f_lang_1 = np.array(vertex_data['f_lang_1'])
f_lang_2 = np.array(vertex_data['f_lang_2'])

f_dc_0 = np.array(vertex_data['f_dc_0'])
f_dc_1 = np.array(vertex_data['f_dc_1'])
f_dc_2 = np.array(vertex_data['f_dc_2'])

opacity = np.array(vertex_data['opacity'])

# Combine into single arrays
pos = np.column_stack((x, y, z))
lf = np.column_stack((f_lang_0, f_lang_1, f_lang_2))
cluster_col = np.column_stack((f_dc_0, f_dc_1, f_dc_2))

_, idx = np.unique(cluster_col, axis=0, return_index=True)
unique_vals = cluster_col[np.sort(idx)]
cluster = np.sum(
    np.all(
        (cluster_col[:, np.newaxis, :] == unique_vals[np.newaxis, :, :])
    , axis=2) * np.arange(len(unique_vals))[np.newaxis, :]
, axis=1)

if args.extend_ply:
    from plyfile import PlyElement
    # Step 1: Convert structured array to a dict of arrays
    vertex_dict = {name: vertex_data[name] for name in vertex_data.dtype.names}
    # Step 2: Add new attribute to dict
    vertex_dict['cluster'] = cluster
    # Step 3: Create new structured array with updated dtype
    new_dtype = vertex_data.dtype.descr + [('cluster', 'u1')]  # f4 = 32-bit float
    new_vertex_array = np.empty(cluster.size, dtype=new_dtype)
    for name in vertex_dict:
        new_vertex_array[name] = vertex_dict[name]
    # Step 4: Create new PlyElement and save
    new_vertex_element = PlyElement.describe(new_vertex_array, 'vertex')
    new_ply = PlyData([new_vertex_element], text=plydata.text)  # keep ASCII/binary format
    new_ply.write(in_filepath)

n_clusters = np.unique(cluster).size
std_metric = np.array([lf[cluster == cluster_id].std() for cluster_id in range(n_clusters)])
is_not_none = np.array([not np.all(cluster_col[cluster == cluster_id]==0) for cluster_id in range(n_clusters)])
keep_cluster = (std_metric >= std_thresh) & (is_not_none)
# cluster_filter = keep_cluster[cluster]
n_final_clusters = keep_cluster.sum()
# give the user some info for debugging
print(f'Started with {n_clusters} clusters (numbered 0-{n_clusters-1})')
print(f'Kept clusters {np.arange(n_clusters)[keep_cluster]}')

means = np.stack([pos[cluster==i].mean(axis=0) for i in range(n_clusters) if keep_cluster[i]])
covs = np.stack([np.cov(pos[cluster==i], rowvar=False) for i in range(n_clusters) if keep_cluster[i]])
mean_lfs_latent = np.stack([lf[cluster==i].mean(axis=0) for i in range(n_clusters) if keep_cluster[i]])

# Decode language features using the scene autoencoder
encoder_hidden_dims = [256, 128, 64, 32, 3]  # NEEDS TO MATCH TRAINING ARGS
decoder_hidden_dims = [16, 32, 64, 128, 256, 512]  # NEEDS TO MATCH TRAINING ARGS
feature_dim = 512  # NEEDS TO MATCH TRAINING ARGS

model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=feature_dim)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

with torch.no_grad():
    decoded_features = model.decode(torch.tensor(mean_lfs_latent, dtype=torch.float32)).numpy()
mean_lfs = decoded_features

mean_cols = unique_vals[keep_cluster]


# ---------------------------------------------------------------------
#             DISTANCE ESTIMATION & EDGE CREATION


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


distances = np.zeros((n_final_clusters, n_final_clusters))
for i, j in itertools.combinations(range(n_final_clusters), 2):
    distances[i, j] = bhattacharyya_coefficient(means[i], covs[i], means[j], covs[j])

edges = np.indices((n_final_clusters, n_final_clusters)).transpose((1, 2, 0))[distances >= distance_thresh]


# ---------------------------------------------------------------------
#                         OUTPUT GRAPHML FILE (& MEANS)


from plyfile import PlyElement
assert means.shape[1] == 3, "Array must have shape (N, 3)"

# Create structured array for PLY
vertices = np.array(
    [tuple(p) for p in means],
    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
)

# Combine XYZ and RGB into a structured array
vertex_data = np.empty(means.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
vertex_data['x'] = means[:, 0]
vertex_data['y'] = means[:, 1]
vertex_data['z'] = means[:, 2]
vertex_data['red'] = mean_cols[:, 0]
vertex_data['green'] = mean_cols[:, 1]
vertex_data['blue'] = mean_cols[:, 2]

# Create PLY element
ply = PlyData([PlyElement.describe(vertex_data, 'vertex')], text=True)
ply.write(out_filepath[:-8] + '_means.ply')





G = nx.Graph()
for i, (p, l) in enumerate(zip(means, mean_lfs)):
    # G.add_node(i, pos_x=p[0], pos_y=p[1], pos_z=p[2],
    #               lang_feat_0=l[0], lang_feat_1=l[1], lang_feat_2=l[2])
    G.add_node(i, pos_x=p[0], pos_y=p[1], pos_z=p[2],
               **{f'lang_feat_{j}' : f for j, f in enumerate(l)})

for u, v in edges:
    G.add_edge(u, v)

nx.write_graphml(G, out_filepath)
print('\033[1m' + f"Wrote scene graph to {out_filepath}" + '\033[0m')