# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
try:
    from pytorch3d import _C
    from pytorch3d.structures import Meshes, Pointclouds
    import pytorch3d
    from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance
except ImportError:
    print("Pytorch3d not installed. Please install it using the following command: pip install pytorch3d or avoid using it")

import torch

"""
This file defines distances between meshes and pointclouds.
The functions make use of the definition of a distance between a point and
an edge segment or the distance of a point and a triangle (face).

The exact mathematical formulations and implementations of these
distances can be found in `csrc/utils/geometry_utils.cuh`.
"""

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3



def _get_closest_face_per_point(pcls, meshes, max_dist=float("inf"), min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA):
        """
        Args:
            pcls: Pointclouds object representing a batch of point clouds.
            meshes: Meshes object representing a batch of meshes.
            min_triangle_area: Minimum area of a triangle to consider.
        Returns:
            idxs: LongTensor of shape (P,) where P is the total number of
                points in the batch of point clouds. The tensor contains the
                index of the closest face in the batch of meshes for each point.

        """
        
        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_tris = meshes.num_faces_per_mesh().max().item()

        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        idxs[dists > max_dist] = -1
        return idxs


def get_closest_face_per_point(faces : torch.Tensor, vertices : torch.Tensor, query_points: torch.Tensor, max_dist=float("inf"), min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA):
    """ Gets the index of the closest triangle in the mesh to each point in the point cloud.
    Args:
        faces: torch.Tensor of shape (F, 3) where F is the number of faces in the mesh.
        vertices: torch.Tensor of shape (V, 3) where V is the number of vertices in the mesh.
        query_points: torch.Tensor of shape (P, 3) where P is the number of points in the point cloud.
    Returns:
        idxs: LongTensor of shape (P,) where P is the number of points in the point cloud.
            The tensor contains the index of the closest face in the mesh for each point.
    """
    triangles = faces.unsqueeze(0)
    vertices = vertices.unsqueeze(0)
    pytorch3d_meshes = pytorch3d.structures.Meshes(verts=vertices, faces=triangles)
    pytorch3d_pcds = pytorch3d.structures.Pointclouds(points=query_points.unsqueeze(0))
    idxs = _get_closest_face_per_point(pytorch3d_pcds, pytorch3d_meshes, max_dist, min_triangle_area)
    return idxs



def point_to_face_distance(
    meshes,
    pcls,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist


def point_to_mesh_distance(
    meshes,
    pcls,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    return point_to_face


def point_mesh_face_distance(
    meshes,
    pcls,
    max_dist: float = float("inf"),
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = (point_to_face * (point_to_face > max_dist)) * weights_p
    point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = (face_to_point * (face_to_point < max_dist)) * weights_t
    face_dist = face_to_point.sum() / N

    return point_dist + face_dist


def _point_edge_sq_distance_all_v_all(points, edges):

    
    l2 = ((edges[:, 1] - edges[:, 0]) ** 2).sum(dim=1)
    
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of point p onto the line. 
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    PV = points[:, None] - edges[None, :, 0]
    WV = (edges[:, 1] - edges[:, 0])[None]
    dot = torch.sum(PV * WV, dim=2)
    t = torch.clamp(dot / l2, 0, 1)
    
    projection = edges[:, 0][None] + t[:, :, None] * (edges[:, 1] - edges[:, 0])[None]  # Projection falls on the segment
    
    dists = ((points[:, None] - projection) ** 2).sum(dim=2)
    
    too_close_mask = l2 < 1e-8
    dists[:, too_close_mask] = ((points.unsqueeze(1) - edges[None, too_close_mask, 0]) ** 2).sum(dim=2)
    
    return dists

    # # Extract edge endpoints
    # A = edges[:, 0, :]  # Shape: (E, 2) - Start points of edges
    # B = edges[:, 1, :]  # Shape: (E, 2) - End points of edges

    # # Compute edge vectors and squared lengths
    # AB = B - A  # Shape: (E, 2)
    # AB_squared = torch.sum(AB**2, dim=1)  # Shape: (E,)

    # # Expand vertices and edges to compute distances
    # V_exp = points[:, None, :]  # Shape: (V, 1, 2)
    # A_exp = A[None, :, :]         # Shape: (1, E, 2)
    # B_exp = B[None, :, :]         # Shape: (1, E, 2)

    # # Compute vectors from vertices to edge endpoints
    # AP = V_exp - A_exp  # Shape: (V, E, 2)
    # BP = V_exp - B_exp  # Shape: (V, E, 2)


    # # Project AP onto AB to compute the projection scalar t
    # t = torch.sum(AP * AB[None, :, :], dim=2) / (AB_squared + 1e-12)  # Shape: (V, E)
    # t = torch.clamp(t, 0, 1)  # Clamp t to [0, 1] to stay within the segment

    # # Compute the projection points on the edges
    # projection = A_exp + t[:, :, None] * AB[None, :, :]  # Shape: (V, E, 2)

    # # Compute distances from vertices to the projection points
    # dist_to_projection = torch.norm(V_exp - projection, dim=2)  # Shape: (V, E)

    # # Compute distances from vertices to edge endpoints
    # dist_to_A = torch.norm(AP, dim=2)  # Shape: (V, E)
    # dist_to_B = torch.norm(BP, dim=2)  # Shape: (V, E)

    # # Combine distances: minimum of the three distances
    # distances = torch.minimum(dist_to_projection, torch.minimum(dist_to_A, dist_to_B))  # Shape: (V, E)

    # return distances


def _point_edge_sq_distance_along_last_dimension(points, edges):
    """Same as above but we expect (N, 3) and (N, 2, 3) tensors to compute (N,) distances"""
    l2 = ((edges[:, 1] - edges[:, 0]) ** 2).sum(dim=1)

    PV = points - edges[:, 0]
    WV = edges[:, 1] - edges[:, 0]

    dot = torch.sum(PV * WV, dim=1)
    t = torch.clamp(dot / (l2 + 1e-6), 0, 1)

    projection = edges[:, 0] + t[:, None] * (edges[:, 1] - edges[:, 0])

    dists = ((points - projection) ** 2).sum(dim=1)

    too_close_mask = l2 < 1e-8
    dists[too_close_mask] = ((points[too_close_mask] - edges[too_close_mask, 0]) ** 2).sum(dim=1)
    return dists


    # Extract edge endpoints
    # A = edges[:, 0, :]  # Shape: (N, 3) - Start points of edges
    # B = edges[:, 1, :]  # Shape: (N, 3) - End points of edges

    # # Compute edge vectors and squared lengths
    # AB = B - A  # Shape: (N, 3)
    # AB_squared = torch.sum(AB**2, dim=1)  # Shape: (N,)

    # # Compute vectors from points to edge endpoints
    # AP = points - A  # Shape: (N, 3)
    # BP = points - B  # Shape: (N, 3)

    # # Project AP onto AB to compute the projection scalar t
    # t = torch.sum(AP * AB, dim=1) / (AB_squared + 1e-12)  # Shape: (N,)
    # t = torch.clamp(t, 0, 1)  # Clamp t to [0, 1] to stay within the segment

    # # Compute the projection points on the edges
    # projection = A + t[:, None] * AB  # Shape: (N, 3)

    # # Compute distances from points to the projection points
    # dist_to_projection = torch.norm(points - projection, dim=1)  # Shape: (N,)

    # # Compute distances from points to edge endpoints
    # dist_to_A = torch.norm(AP, dim=1)  # Shape: (N,)
    # dist_to_B = torch.norm(BP, dim=1)  # Shape: (N,)

    # # Combine distances: minimum of the three distances
    # distances = torch.minimum(dist_to_projection, torch.minimum(dist_to_A, dist_to_B))  # Shape: (N,)
    # return distances

# def _point_face_distance_along_last_dimension(points : torch.Tensor, faces : torch.Tensor):
#     """Computes the distance between points and faces in 3D. Uses _point_edge_distance as intermediate step
#         Args:
#             points: torch.Tensor of shape (P, 3) where P is the number of points.
#             faces: torch.Tensor of shape (F, 3, 3) where F is the number of faces.
#         Returns:
#             dists: torch.Tensor of shape (P, F) where dists[p, f] is the distance between the point p and the face f.
#     """
#     face_normals = torch.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
#     face_normals = face_normals / torch.norm(face_normals, dim=1)[:, None]
    
#     p





     

def get_closest_edges_to_point_sq_dist(points : torch.Tensor, edges: torch.Tensor, k: int = 3, return_self_incident : bool = False):
    """Computes the pairwise distances between points and edges in 3D.
        Args:
            points: torch.Tensor of shape (P, 3) where P is the number of points.
            edges: torch.Tensor of shape (E, 2, 3) where E is the number of edges.
        Returns:
            dists: torch.Tensor of shape (P, E) where dists[p, e] is the distance
            b   etween the point p and the edge e.
    """
    # Expand vertices and edges for vectorized computation
    V, E = points.size(0), edges.size(0)

    with torch.no_grad():
        batch_size = 50_000_000 // len(edges) # Define a batch size
        indices = torch.empty((V, k), dtype=torch.int64, device=points.device)  # Initialize the indices tensor
        dists = torch.empty((V, k), device=points.device)  # Initialize the distances tensor

        for i in range(0, V, batch_size):
            batch_points = points[i:i + batch_size]
            batch_dists = _point_edge_sq_distance_all_v_all(batch_points, edges)
            if not return_self_incident:
                batch_dists[batch_dists == 0] = float("inf")
            batch_dists, batch_indices = torch.topk(batch_dists, k, dim=1, largest=False)
            indices[i:i + batch_size] = batch_indices
            dists[i:i + batch_size] = batch_dists
    
    dists_new = _point_edge_sq_distance_along_last_dimension(points.unsqueeze(1).repeat(1, k, 1).reshape(-1, 3), edges[indices].reshape(-1, 2, 3)).reshape(V, k)
    # assert torch.allclose(dists, dists_new)
    return indices, dists_new


@torch.jit.script
def point_to_triangle_edges_sq_dist(points : torch.Tensor, triangles : torch.Tensor):
    """Takes points and triangle vertices as input and returns the distance of each point to the edges of the triangle.
    Args:
        points: torch.Tensor of shape (P, 3) where P is the number of points.
        triangles: torch.Tensor of shape (P, 3, 3) where T is the number of triangles.
    Returns:
        dists: torch.Tensor of shape (P, 3) where dists[p, t] is the distance between the point p and the t-th edge (connecting points [t-1, t]) of triangle p.
    """
    assert len(points) == len(triangles)
    P, _ = points.size()
    edge_1 = torch.stack([triangles[:, 0], triangles[:, 1]], dim=1) # shape (P, 2, 3)
    edge_2 = torch.stack([triangles[:, 1], triangles[:, 2]], dim=1)
    edge_3 = torch.stack([triangles[:, 2], triangles[:, 0]], dim=1)

    all_edges_flat = torch.cat([edge_1, edge_2, edge_3], dim=0) # shape (3P, 2, 3)
    all_points_flat = torch.cat([points, points, points], dim=0) # shape (3P, 3)
    # triangle_edges = triangle_edges.
    dists = _point_edge_sq_distance_along_last_dimension(all_points_flat, all_edges_flat)
    dists = torch.stack([dists[:P], dists[P:2*P], dists[2*P:]], dim=1)
    return dists

    



# def _point_triangle_distance_all_v_all(points : torch.Tensor, triangles: torch.Tensor):
#     # Extract triangle vertices
#     A = triangles[:, 0, :]  # Shape: (T, 3) - First vertex of triangles
#     B = triangles[:, 1, :]  # Shape: (T, 3) - Second vertex of triangles
#     C = triangles[:, 2, :]  # Shape: (T, 3) - Third vertex of triangles

#     # Compute triangle vectors and squared lengths
#     AB = B - A  # Shape: (T, 3)
#     AC = C - A  # Shape: (T, 3)
#     BC = C - B  # Shape: (T, 3)
#     AB_squared = torch.sum(AB**2, dim=1)  # Shape: (T,)
#     AC_squared = torch.sum(AC**2, dim=1)  # Shape: (T,)
#     BC_squared = torch.sum(BC**2, dim=1)  # Shape: (T,)

#     # Compute triangle normals
#     N = torch.cross(AB, AC)  # Shape: (T, 3)
#     N_norm = torch.norm(N, dim=1)  # Shape: (T,)
#     N = N / N_norm[:, None]  # Shape: (T, 3)

#     # Compute vectors from points to triangle vertices
#     AP = points[:, None, :] - A[None, :, :]  # Shape: (V, T, 3)
#     BP = points[:, None, :] - B[None, :, :]  # Shape: (V, T, 3)
#     CP = points[:, None, :] - C[None, :, :]  # Shape: (V, T, 3)

#     # Compute triangle areas
#     areas = 0.5 * torch.norm(torch.cross(AB, AC), dim=1)  # Shape: (T,)

#     # Compute the projection of AP onto the triangle plane
#     AP_N = torch.sum(AP * N[None, :, :], dim=2)  # Shape: (V, T)
#     AP_proj = points[:, None, :] - AP_N[:, :, None] * N[None, :, :]  # Shape: (V, T, 3)

#     # Compute the barycentric coordinates of the projection points
#     AB_dot_AP = torch.sum(AB[None

