"""@TODO source"""
import sys
from typing import Union, Tuple, List
import torch

@torch.jit.script
def _rand_barycentric_coords(
    n_samples : int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, n_samples, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2


@torch.jit.script
def triangle_areas(triangle_points: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of a batch of triangles defined by the coordinates of their
    vertices.

    Args:
        triangle_points: FloatTensor of shape (N, 3, 3) giving the coordinates of
            the vertices of the triangles.

    Returns:
        areas: FloatTensor of shape (N,) giving the area of each triangle.
    """
    v1 = triangle_points[:, 1] - triangle_points[:, 0]
    v2 = triangle_points[:, 2] - triangle_points[:, 0]
    cross = torch.cross(v1, v2, dim=1)
    cross[cross.isnan().any(dim=1)] = 0
    return 0.5 * cross.norm(p=2, dim=1)


@torch.jit.script
def sample_points_from_mesh(
    triangles : torch.Tensor,
    verts : torch.Tensor,
    triangle_features: Union[None, List[torch.Tensor]] = None,
    num_samples: int = 10000,
    return_normals: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """

    triangle_points = verts[triangles]
    with torch.no_grad():
        areas = triangle_areas(triangle_points)
        sample_face_idxs = torch.multinomial(
            areas, num_samples, replacement=True
        )  # (N, num_samples)

    # faces = triangles[sample_face_idxs]
    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[triangles]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples = w0[:, None] * a + w1[:, None] * b + w2[:, None] * c

    # Normals for the sampled points are face normals computed from
    # the vertices of the face in which the sampled point lies.
    normals = torch.zeros((num_samples, 3), device=verts.device)
    vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
    vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
        min=1e-14
    )
    normals = vert_normals[sample_face_idxs]

    if triangle_features is not None:
        features = [feat[sample_face_idxs] for feat in triangle_features]
    else:
        features = []

    return samples, normals, features
    