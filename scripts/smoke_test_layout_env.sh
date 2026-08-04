#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HOUSELAYOUT3D_LAYOUT_PYTHON:-/tmp/tmp_data/miniconda3/envs/houselayout3d-layout/bin/python}"
export HOUSELAYOUT3D_PROJECT_ROOT="${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import numpy as np
import pgeof
import torch
from pytorch3d.ops import knn_points
from torch_cluster import knn_graph
from torch_scatter import scatter_mean

import clip
import frnn
import hovsg
from hovsg.graph.room import Room
from src.data import Data
from src.transforms.partition import CutPursuitPartition
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

root = Path(os.environ["HOUSELAYOUT3D_PROJECT_ROOT"])
assert torch.cuda.is_available()

points = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
    device="cuda",
)
lengths = torch.tensor([3], dtype=torch.long, device="cuda")
assert knn_points(points, points, K=2).idx.shape == (1, 3, 2)
assert scatter_mean(
    points[0], torch.tensor([0, 0, 1], device="cuda"), dim=0
).shape == (2, 3)
assert knn_graph(points[0], k=1).shape == (2, 3)
_, frnn_idx, _, _ = frnn.frnn_grid_points(
    points, points, lengths, lengths, K=2, r=3.0
)
assert frnn_idx.shape == (1, 3, 2)

xyz = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]],
    dtype=np.float32,
)
neighbors, _ = pgeof.knn_search(xyz, xyz, 3)
pointers = np.arange(0, neighbors.size + 1, 3, dtype=np.uint32)
features = pgeof.compute_features(
    xyz, neighbors.reshape(-1), pointers, k_min=1
)
edges = torch.tensor(
    [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
     [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]]
)
nag = CutPursuitPartition(
    regularization=0.05,
    spatial_weight=1.0,
    cutoff=1,
    parallel=False,
    iterations=3,
    k_adjacency=2,
)(Data(
    pos=torch.from_numpy(xyz),
    x=torch.from_numpy(features[:, :3]),
    edge_index=edges,
    edge_attr=torch.ones(edges.shape[1]),
))
assert nag.num_levels == 2

assert Room is not None and hovsg.__file__
assert clip.available_models()
assert OneFormerProcessor is not None
assert OneFormerForUniversalSegmentation is not None
assert (root / "weights/OneFormer/oneformer_coco_swin_large/pytorch_model.bin").is_file()
assert (root / "weights/Metric3D/metric_depth_vit_large_800k.pth").is_file()
assert (root / "weights/OpenAI-CLIP/ViT-L-14-336px.pt").is_file()

print("layout environment: ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name())
print("SPT levels", nag.num_levels, "nodes", nag.num_points)
PY
