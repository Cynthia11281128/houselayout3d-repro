# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Union
import torch
try:
    from pytorch3d.ops.knn import knn_points, knn_gather
except ImportError:
    print("Pytorch3d not found. Falling back to custom implementation.")



def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError(
            'point_reduction must be one of ["mean", "sum", "max"] or None'
        )
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")

def _chamfer_retrieve_feature_of_nearest(features, neighbor_idx, y_lengths):
    return knn_gather(features, neighbor_idx, y_lengths)[..., 0, :]

def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    norm: int,
):

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    distances = x_nn.dists  # (N, P1)
    idx = x_nn.idx
    return (distances, idx)


def chamfer_distance_indices(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.

    Returns:
        2-element tuple containing

        - **closest_y_to_x of size N, P1 containing the index of the closest point in y to each point in x
        - **closest_x_to_y of size N, P2
    """

    x_lengths = x_lengths if x_lengths is not None else torch.ones(x.shape[0], device=x.device, dtype=torch.long) * x.shape[1]
    y_lengths = y_lengths if y_lengths is not None else torch.ones(y.shape[0], device=y.device, dtype=torch.long) * y.shape[1]

    cham_x, ixes_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        norm=2,
    )
    cham_y, ixes_y = _chamfer_distance_single_direction(
        y,
        x,
        y_lengths,
        x_lengths,
        norm=2,
    )
    return cham_x, cham_y, ixes_x, ixes_y


def mutual_nearest_neighbors(x, y, x_lengths, y_lengths, k=1, use_pytorch3d=True):
    """
    Computes the mutual nearest neighbors between two sets of points.

    Parameters:
    x (array-like): The first set of points.
    y (array-like): The second set of points.
    x_lengths (array-like): Lengths associated with the first set of points.
    y_lenghts (array-like): Lengths associated with the second set of points.

    Returns:
    tuple: Indices of mutual nearest neighbors in the first and second sets.
    """
    if use_pytorch3d:
        with torch.no_grad():
            x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=2, K=k)
            y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=2, K=k)
        return x_nn.idx, y_nn.idx
    else:
        x_nn = get_closest_points_to_point(x.squeeze(0), y.squeeze(0), k=k)
        y_nn = get_closest_points_to_point(y.squeeze(0), x.squeeze(0), k=k)
        return x_nn, y_nn
    

def get_closest_points_to_point(points1: torch.Tensor, points2: torch.Tensor, k: int = 3):
    """Computes the k-nearest neighbors for each point in points1 from points2.
        Args:
            points1: torch.Tensor of shape (P1, 3) where P1 is the number of points.
            points2: torch.Tensor of shape (P2, 3) where P2 is the number of points.
            k: int, number of nearest neighbors to find.
        Returns:
            indices: torch.Tensor of shape (P1, k) where indices[p, i] is the index of the i-th nearest neighbor in points2 to the p-th point in points1.
    """
    P1, P2 = points1.size(0), points2.size(0)

    with torch.no_grad():
        batch_size = 100_000_000 // P2  # Define a batch size
        indices = torch.empty((P1, k), dtype=torch.int64, device=points1.device)  # Initialize the indices tensor

        for i in range(0, P1, batch_size):
            batch_points1 = points1[i:i + batch_size]
            dists = torch.cdist(batch_points1, points2)  # Compute pairwise distances
            _, batch_indices = torch.topk(dists, k, dim=1, largest=False)  # Get k-nearest neighbors
            indices[i:i + batch_size] = batch_indices

    return indices