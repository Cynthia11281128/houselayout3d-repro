from sklearn.neighbors import NearestNeighbors
import colorsys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
# import matplotlib.path as mplPath
# from math import sqrt

from tqdm import tqdm
from geometry_utils import compute_ray_mesh_intersections_ray_tracing, project_points_2d_to_3d_plane_based, project_points_3d_to_2d_plane_based, compute_closest_triangle_to_points_o3d
from cgal_triangulations import triangulate_polygon, triangulate_polygon_from_edges, triangulate_polygon_from_edges2, triangulate_polygon_from_triangles
import numpy as np
import cv2
import rdp
import torch.multiprocessing as mp
from shapely.geometry import Polygon

from merge_split_util import get_closest_edges_to_point_sq_dist
import point_triangle_distance_vectorized
from polygon_fitting_config import PolygonFittingConfig
from point_triangle_distance_vectorized import find_k_closest_triangles


def stable_plane_color(plane_id: int) -> np.ndarray:
    hue = (int(plane_id) * 0.6180339887498949) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 0.95), dtype=np.float64)


def triangulate_3d_polygon(args):
    vertices_3d, plane, contours = args
    # remove last = first point if present for each contour
    contour_vertices_2d = [project_points_3d_to_2d_plane_based(vertices_3d[contour], plane) for contour in contours]
    contour_areas = [Polygon(contour + [contour[-1]]).area for contour in contour_vertices_2d]
    contours_sorted =np.argsort(contour_areas)
    vertices_2d, inner_vertices_2d = contour_vertices_2d[contours_sorted[-1]], [contour_vertices_2d[i] for i in contours_sorted[:-1]]
    new_points_2d, triangles_2d, triangle_ixes = triangulate_polygon(vertices_2d, holes=inner_vertices_2d)
    new_points_3d = project_points_2d_to_3d_plane_based(new_points_2d, plane)
    triangle_ixes = np.array(triangle_ixes).astype(np.int32)
    return new_points_3d, triangle_ixes


def retriangulate_3d_triangles(vertices_3d, polygon_triangles,  plane_eq, polygon, delete_small=True, plot=False):
    if True:
        triangle_vertices_2d = project_points_3d_to_2d_plane_based(vertices_3d[polygon_triangles.cpu().numpy()].reshape(-1, 3), plane_eq)
        new_points_2d, _, triangle_ixes = triangulate_polygon_from_triangles(triangle_vertices_2d.reshape(-1, 3, 2), simplify=True, delete_small=delete_small, plot=plot)
        new_points_3d = project_points_2d_to_3d_plane_based(new_points_2d, plane_eq)
        triangle_ixes = np.array(triangle_ixes).astype(np.int32)
    else:
        polygon_edges = torch.cat([polygon_triangles[:, [0, 1]], polygon_triangles[:, [1, 2]], polygon_triangles[:, [2, 0]]], dim=0)
        polygon_edges = polygon_edges[polygon_edges[:, 0] != polygon_edges[:, 1]]
        # all_edges = torch.sort(all_edges, dim=1)[0]
        # polygon_edges = all_edges[(vertex_polygons[all_edges] == polygon).any(dim=2).all(dim=1)]
        polygon_edges = polygon_edges.sort(dim=1).values
        polygon_edges, edge_counts = torch.unique(polygon_edges, return_counts=True, dim=0)
        polygon_edges = polygon_edges[edge_counts % 2 == 1]

        if len(polygon_edges) < 3:
            print(f"Polygon {polygon} has less than 3 edges.")
            return None, None

        edges = polygon_edges.cpu().numpy()



        assert len(edges) == len(np.unique(np.sort(np.array(edges), axis=1), axis=0))

        contours = triangulate_polygon_from_edges2(edges)
        
        contours = [contour[:-1] if np.all(contour[0] == contour[-1]) else contour for contour in contours]
        contours = [c for c in contours if len(c) >= 3]
        if len(contours) == 0 or np.isnan(np.concatenate(contours)).any():
            print(f"Polygon {polygon} has no valid contours.")
            return None, None
    
        new_points_3d, triangle_ixes = triangulate_3d_polygon((vertices_3d, plane_eq, contours))

    return new_points_3d, triangle_ixes


def project_points_3d_to_2d_plane_based_torch(points: torch.Tensor, plane_equation: torch.Tensor) -> torch.Tensor:
    """
    Takes points in 3D on a 3D plane. Returns 2D coordinates in the plane coordinate system corresponding to the points.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N 3D points.
        plane_equation (torch.Tensor): Tensor of shape (4,) representing the plane equation coefficients (a, b, c, d).
    
    Returns:
        torch.Tensor: Tensor of shape (N, 2) representing the 2D coordinates of the points in the plane coordinate system.
    """
    # Normalize the plane normal
    n = plane_equation[:3] / torch.norm(plane_equation[:3])
    
    # Define an arbitrary origin on the plane
    r_O = torch.tensor([0, 0, -plane_equation[3] / n[2]], dtype=points.dtype, device=points.device)
    
    # Define two orthogonal directions in the plane
    e_1 = torch.tensor([1, 0, -n[0] / n[2]], dtype=points.dtype, device=points.device)
    e_1 /= torch.norm(e_1)
    e_2 = torch.cross(n, e_1)
    e_2 /= torch.norm(e_2)
    
    # Project points onto the plane
    r_P_minus_r_O = points - r_O
    t_1 = torch.matmul(e_1, r_P_minus_r_O.T)
    t_2 = torch.matmul(e_2, r_P_minus_r_O.T)
    projected_points = torch.stack([t_1, t_2], dim=-1)
    
    return projected_points

# def project_points_3d_to_2d_plane_batched()

@torch.jit.script
def project_points_3d_to_3d_plane(points: torch.Tensor, planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project points to planes in a batched manner.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N 3D points.
        planes (torch.Tensor): Tensor of shape (N, 4) representing N plane equations (a, b, c, d).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Tensor of shape (N, 3) representing the projected points.
            - Tensor of shape (N,) representing the distances from the points to the planes.
    """
    a, b, c, d = planes[:, 0], planes[:, 1], planes[:, 2], planes[:, 3]
    plane_normals = torch.stack([a, b, c], dim=1)
    plane_origins = torch.stack([torch.zeros_like(d), torch.zeros_like(d), -d / (c + 1e-14)], dim=1)
    
    vec = points - plane_origins
    dist = torch.sum(vec * plane_normals, dim=1)
    projected_points = points - dist[:, None] * plane_normals
    
    return projected_points, torch.abs(dist)


def approximate_polygon_rdp(points, epsilon=0.0005, thresh=0.001):
    projected_contour_int =(points/ thresh).astype(int)
    peri = cv2.arcLength(projected_contour_int, True)
    mask = rdp.rdp(projected_contour_int, epsilon=epsilon * peri, return_mask=True)
    return mask


@torch.jit.script
def project_points_to_lines_torch(points: torch.Tensor, line_eqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projects k points to k lines.
    Args:
        points (torch.Tensor): The points to project with shape (k, 3)
        line_eqs (torch.Tensor): The line equations with shape (k, 6)
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - The projected points with shape (k, 3)
            - The distances from the points to the lines with shape (k,)
    """
    p1, p2 = line_eqs[:, :3], line_eqs[:, 3:]
    vec = p2 - p1
    vec = vec / torch.norm(vec, dim=1, keepdim=True)
    # p1, vec = p1[:, None, :], vec[:, None, :]
    projected_points = p1 + ((points - p1) * vec).sum(dim=1, keepdim=True) * vec
    dists = torch.norm(projected_points - points, dim=1)
    return projected_points, dists



def intersection_line_between_planes(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
    a, b   Tensors of shape (4,)
            Ax + By + Cz + D = 0
            A,B,C,D in order  

    output: 2 points on line of intersection, Tensors of shape (3,)
    """
    a_vec, b_vec = a[:3], b[:3]
    aXb_vec = torch.cross(a_vec, b_vec)

    A = torch.stack([a_vec, b_vec, aXb_vec])
    d = torch.tensor([-a[3], -b[3], 0.], dtype=a.dtype, device=a.device).reshape(3, 1)

    if torch.abs(torch.det(A)) < 1e-8:
        raise ValueError("The planes are parallel.")

    p_inter = torch.linalg.solve(A, d).T

    result1 = p_inter[0], (p_inter + aXb_vec)[0]
    return p_inter[0], (p_inter + aXb_vec)[0]

@torch.jit.script
def fit_plane_torch(vertices: torch.Tensor, vertex_normals: Optional[torch.Tensor] = None, max_vertices: int = 5000) -> torch.Tensor:
    
    if not torch.isfinite(vertices).all():
        print(f"Plane has NaN vertices.")
        vertices = vertices[torch.isfinite(vertices).all(dim=1)]

    if len(vertices) < 3:
        # print(f"Plane has less than 3 vertices.")
        return float("inf") * torch.ones(4, dtype=vertices.dtype, device=vertices.device)
    
    if len(vertices) > max_vertices:
        vertices = vertices[torch.randperm(len(vertices))[:max_vertices]]
        
    centroid = vertices.mean(dim=0)
    centered_vertices = vertices - centroid
    _, _, vh = torch.svd(centered_vertices)
    normal = vh[:, 2]

    # Plane equation: normal[0] * x + normal[1] * y + normal[2] * z = d
    d = torch.dot(normal, centroid)
    plane_equation = torch.cat([normal, -d.unsqueeze(0)])

    # rotate the plane normal so it aligns with the majority
    if vertex_normals is not None:
        if (torch.sum(plane_equation[:3] * vertex_normals, dim=1) < 0).sum() > len(vertices) / 2:
            plane_equation = -plane_equation

    return plane_equation

# @torch.jit.script
def intersection_line_between_planes_batched(plane_eqs_A: torch.Tensor, plane_eqs_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes the intersections between two sets of planes.
    Args:
        plane_eqs_A (torch.Tensor): The plane equations with shape (n_planes, 4)
        plane_eqs_B (torch.Tensor): The plane equations with shape (n_planes, 4)
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The starting points of the lines with shape (n_planes, 3)
            - The ending points of the lines with shape (n_planes, 3)
    """
    if len(plane_eqs_B) == 0:
        return float("inf") * torch.zeros(0, 3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device), float("inf") * torch.zeros(0, 3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device)
    
    a_vec, b_vec = plane_eqs_A[:, :3], plane_eqs_B[:, :3]
    aXb_vec = torch.cross(a_vec, b_vec, dim=-1)

    A = torch.stack([a_vec, b_vec, aXb_vec], dim=1)
    d = torch.stack([-plane_eqs_A[:, 3], -plane_eqs_B[:, 3], torch.zeros_like(plane_eqs_A[:, 3])]).T.unsqueeze(-1)

    p_inter = torch.empty(len(plane_eqs_A), 3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device)

    parallel_mask = torch.abs(torch.det(A)) < 1e-8
    p_inter[~parallel_mask] = torch.linalg.solve(A[~parallel_mask], d[~parallel_mask]).squeeze(-1)

    # mask out parallel planes
    p_inter[parallel_mask] = float("inf")


    # for i in range(len(plane_eqs_A)):
    #     a = plane_eqs_A[i]
    #     b = plane_eqs_B[i]
    #     a_vec_i, b_vec_i = a[:3], b[:3]
    #     assert (a_vec_i == a_vec[i]).all()
    #     assert (b_vec_i == b_vec[i]).all()
    #     aXb_vec_i = torch.cross(a_vec_i, b_vec_i)

    #     assert torch.isclose(aXb_vec_i, aXb_vec[i]).all()

    #     A_i = torch.stack([a_vec_i, b_vec_i, aXb_vec_i])

    #     assert (A_i == A[i]).all()
    #     d_i = torch.tensor([-a[3], -b[3], 0.], dtype=a.dtype, device=a.device).reshape(3, 1)

    #     assert torch.isclose(d_i, d[i]).all()
    #     # try:
    #     p_inter_i = torch.linalg.solve(A_i, d_i).squeeze(-1)

    #     assert torch.isclose(p_inter_i, p_inter[i]).all()


        
        # except:
        #     p_inter[i] = float("inf") * torch.ones(3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device)




    # validate against non-batched
    # p_inter_non_batched = []
    # p_plus_aXb_non_batched = []
    # for i in range(len(plane_eqs_A)):
    #     try:
    #         p1, p2 = intersection_line_between_planes(plane_eqs_A[i], plane_eqs_B[i])
    #     except ValueError:
    #         p1 = float("inf") * torch.ones(3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device)
    #         p2 = float("inf") * torch.ones(3, dtype=plane_eqs_A.dtype, device=plane_eqs_A.device)
    #     p_inter_non_batched.append(p1)
    #     p_plus_aXb_non_batched.append(p2)

    # p_inter_non_batched = torch.stack(p_inter_non_batched)
    # p_plus_aXb_non_batched = torch.stack(p_plus_aXb_non_batched)
    # assert (torch.isclose(p_inter, p_inter_non_batched, atol=1e-5) | (~torch.isfinite(p_inter) & ~torch.isfinite(p_inter_non_batched))).all(), f"Batched and non-batched results do not match: {p_inter} vs {p_inter_non_batched}"
    
    
    return p_inter, p_inter + aXb_vec

def line_plane_intersection(line_eq : torch.Tensor, plane_eq : torch.Tensor) -> torch.Tensor:
    """ Returns the intersection point between a line and a plane
    Args:
        line_eq (torch.Tensor): The line equation with shape (6,) given by two points on the line (x1, y1, z1, x2, y2, z2)
        plane_eq (torch.Tensor): The plane equation with shape (4,) given by (a, b, c, d) in ax + by + cz + d = 0
    Returns:
        torch.Tensor: The intersection point between the line and the plane with shape (3,)
    """
    p1, p2 = line_eq[:3], line_eq[3:]
    a, b, c, d = plane_eq
    n = torch.tensor([a, b, c], dtype=p1.dtype, device=p1.device)
    p1p2 = p2 - p1
    t = - (torch.dot(p1, n) + d) / (torch.dot(p1p2, n) + 1e-8)
    return p1 + t * p1p2


def line_plane_intersection_batched(line_eqs : torch.Tensor, plane_eqs : torch.Tensor) -> torch.Tensor:
    """ Returns the intersection points between N lines and N planes
    Args:
        line_eqs (torch.Tensor): The line equations with shape (N, 6) given by two points on the line (x1, y1, z1, x2, y2, z2)
        plane_eqs (torch.Tensor): The plane equations with shape (N, 4) given by (a, b, c, d) in ax + by + cz + d = 0
    Returns:
        torch.Tensor: The intersection points between the lines and the planes with shape (N, 3)
    """
    p1, p2 = line_eqs[:, :3], line_eqs[:, 3:]
    a, b, c, d = plane_eqs[:, 0], plane_eqs[:, 1], plane_eqs[:, 2], plane_eqs[:, 3]
    n = torch.stack([a, b, c], dim=1)
    p1p2 = p2 - p1
    t = - (torch.sum(p1 * n, dim=1) + d) / (torch.sum(p1p2 * n, dim=1) + 1e-8)

    # validata against non-batched
    # if len(line_eqs) > 0:
    #     p_inter = p1 + t[:, None] * p1p2
    #     p_inter_non_batched = []
    #     for i in range(len(line_eqs)):
    #         p_inter_non_batched.append(line_plane_intersection(line_eqs[i], plane_eqs[i]))
    #     p_inter_non_batched = torch.stack(p_inter_non_batched)
    #     assert (torch.isclose(p_inter, p_inter_non_batched, atol=1e-5) | (~torch.isfinite(p_inter) & ~torch.isfinite(p_inter_non_batched))).all(), f"Batched and non-batched results do not match: {p_inter} vs {p_inter_non_batched}"
    return p1 + t[:, None] * p1p2



@torch.jit.script
def distances_to_line_segment_torch_batched(As : torch.Tensor, Bs : torch.Tensor, points : torch.Tensor, signed : bool = False) -> torch.Tensor:
    """Compute the distances of the points to the line segment AB using PyTorch.
    Args:
        As (torch.Tensor): The starting points of the line segments with shape (n_segments, 2)
        Bs (torch.Tensor): The ending points of the line segments with shape (n_segments, 2)
        points (torch.Tensor): The points to compute the distances to with shape (n_points, 2)
        signed (bool): Whether to compute signed distances or not
    Returns:
        torch.Tensor: The distances of the points to the line segments with shape (n_segments, n_points)
    """
    assert As.shape == Bs.shape and As.shape[1] == 2, "As and Bs should have shape (n_segments, 2)"
    assert points.shape[1] == 2, "points should have shape (n_points, 2)"
    
    AP = points[None, :] - As[:, None]
    AB = Bs - As
    dot = torch.sum(AP * AB[:, None], dim=-1)
    norm = torch.sum(AB * AB, dim=1)
    t = torch.clamp(dot / (norm[:, None] + 1e-8), 0, 1).unsqueeze(-1)
    projection = As[:, None] + t * AB[:, None]
    distances = torch.norm(points - projection, dim=-1)
    if signed:
        sign = torch.sign(AB[:, None, 0] * (points[:, 1][None] - As[:, None, 1]) - AB[:, None, 1] * (points[None, :,  0] - As[:, None, 0]))
        return sign * distances
    else:
        return distances
        

def distances_to_line_segment_3d(As: torch.Tensor, Bs: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Compute the squared distances of the points to the line segment AB using PyTorch.
    Args:
    As (torch.Tensor): The starting points of the line segments with shape (n_segments, 3)
    Bs (torch.Tensor): The ending points of the line segments with shape (n_segments, 3)
    points (torch.Tensor): The points to compute the distances to with shape (n_points, 3)
    signed (bool): Whether to compute signed distances or not
    Returns:
    torch.Tensor: The squared distances of the points to the line segments with shape (n_segments, n_points)
    """
    assert As.shape == Bs.shape and As.shape[1] == 3, "As and Bs should have shape (n_segments, 3)"
    assert points.shape[1] == 3, "points should have shape (n_points, 3)"
    
    AP = points[None, :] - As[:, None]
    AB = Bs - As
    dot = torch.sum(AP * AB[:, None], dim=-1)
    norm = torch.sum(AB * AB, dim=1)
    t = torch.clamp(dot / (norm[:, None] + 1e-8), 0, 1).unsqueeze(-1)
    projection = As[:, None] + t * AB[:, None]
    distances = torch.norm(points - projection, dim=-1)
    return distances ** 2


@torch.jit.script
def compute_sq_dist_vertices_to_polygon(points : torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Compute the squared distances of the points to the polygon.
    Args:
        points (torch.Tensor): The points to compute the distances to with shape (n_points, 2)
        polygon (torch.Tensor): The vertices of the polygon with shape (n_vertices, 2)
    Returns:
        torch.Tensor: The squared distances of the points to the polygon with shape (n_points,)
    """
    sq_dists = distances_to_line_segment_torch_batched(polygon, torch.roll(polygon, -1, dims=0), points, signed=False) ** 2
    mindist = torch.min(sq_dists, dim=0).values
    return mindist



# # add the line constraints to the loss
# def distances_to_lines(vertices, lines):
#     dist = 0
#     for (i1, i2), line_eq in lines.items():
#         a, b, c = line_eq
#         # penalize the distance of the vertices i1 and i2 to the line
#         p1, p2 = vertices[i1], vertices[i2]
#         # import matplotlib.pyplot as plt
#         # plt.plot([p1[0].detach().numpy(), p2[0].detach().numpy()], [p1[1].detach().numpy(), p2[1].detach().numpy()], 'r')
#         # # plot the line
#         # x = np.linspace(min(vertices[:, 0].detach().numpy()), max(vertices[:, 0].detach().numpy()), 100)
#         # y = (-a * x - c.numpy()) / (b.numpy() + 1e-8)
#         # plt.plot(x, y, 'g')
#         # plt.show()

#         dist1 = torch.abs(a * p1[0] + b * p1[1] + c) / sqrt(a ** 2 + b ** 2 + 1e-8)
#         dist2 = torch.abs(a * p2[0] + b * p2[1] + c) / sqrt(a ** 2 + b ** 2 + 1e-8)
#         dist += dist1 + dist2

#     return dist



# class TriangleMesh3D(nn.Module):
#     def __init__(self, vertices: torch.nn.Parameter, triangles: torch.Tensor):
#         """
#         Args:
#             vertices: Tensor of shape (V, 3) where V is the number of vertices.
#             triangles: Tensor of shape (T, 3) where T is the number of triangles, each triangle is a triplet of vertex indices.
#         """
#         super(TriangleMesh3D, self).__init__()
#         self.vertices = vertices  # (V, 3)
#         self.triangles = triangles              # (T, 3)

#     def compute_distances(self, point_set: torch.Tensor, point_triangles: torch.Tensor) -> torch.Tensor:
#         """
#         Compute distances between points and triangles in the mesh.
        
#         Args:
#             point_set: Tensor of shape (N, 3) where N is the number of points.
#             point_triangles: Tensor of shape (M, 3) where M is the number of point-triangle pairs to compute.
        
#         Returns:
#             distances: Tensor of shape (N, M) with distance from each point to each triangle.
#         """
#         raise NotImplementedError("Implement distance computation in a subclass.")


class VertexSet3D(nn.Module):
    def __init__(self, vertices: torch.Tensor):
        """
        Args:
            vertices: Tensor of shape (V, 3) where V is the number of vertices.
        """
        super(VertexSet3D, self).__init__()
        if len(vertices) == 0:
            print("[WARNING] VertexSet3D has no vertices. Initializing with uninitialized parameters.")
            self.vertices = nn.UninitializedParameter(requires_grad=True)
        else:
            self.vertices = nn.Parameter(vertices, requires_grad=True)  # (V, 3)

    def get(self):
        return self.vertices


class PlaneConstrainedVertexSet3D(VertexSet3D):
    def __init__(self, vertices: torch.Tensor, vertex_plane_assignments: torch.Tensor, vertex_normals : torch.Tensor = None, max_distance: float = 0.01, learn_plane_equations : bool = True, init_planes : Optional[torch.Tensor] = None, device='cpu'):
        """
        Args:
            vertices: Tensor of shape (V, 3) where V is the number of vertices.
            vertex_plane_ixes: Tensor of shape (V, max_planes_per_vertex) where each entry is the index of the plane the vertex belongs to.
            planes: Tensor of shape (P, 4) where P is the number of planes, each plane is defined by (a, b, c, d) in ax + by + cz + d = 0.
        """
        vertices = vertices.to(device)
        super(PlaneConstrainedVertexSet3D, self).__init__(vertices)
        self.n_planes = vertex_plane_assignments.max().item() + 1 if vertex_plane_assignments.numel() > 0 else 0
        assert len(torch.unique(vertex_plane_assignments[vertex_plane_assignments != -1])) == self.n_planes, "The planes should be indexed from 0 to n_planes - 1, with -1 for no plane."
        # computed_planes = self.get_planes(vertices, vertex_plane_assignments)
        # assert (torch.isclose(computed_planes, planes, atol=1e-1) | torch.isclose(-computed_planes, planes, atol=1e-1)).all(), "The computed planes do not match the input planes."

        assert vertex_plane_assignments.shape[1] <= 3, "Only up to 3 planes per vertex are supported."
        vertex_plane_assignments = vertex_plane_assignments.clone().to(device).long()
        # sort the vertex plane assignments such that the plane with the largest number of vertices is the last
        n_vertices_per_plane = torch.zeros(self.n_planes + 1, dtype=torch.int, device=device)
        for i in range(self.n_planes):
            n_vertices_per_plane[i] = (vertex_plane_assignments == i).sum().item()
        n_vertices_per_plane[-1] = len(vertices)
        sorted_indices = torch.argsort(n_vertices_per_plane[vertex_plane_assignments], dim=1, descending=False)
        vertex_plane_assignments = torch.gather(vertex_plane_assignments, 1, sorted_indices)
        self.vertex_plane_assignments = vertex_plane_assignments
        # self.planes = planes.to(device)

        if vertex_normals is not None:
            self.vertex_normals = vertex_normals.to(device)
        else:
            self.vertex_normals = None

        
        
        self.learn_plane_equations = learn_plane_equations
        if len(vertices) == 0:
            print("[WARNING] PlaneConstrainedVertexSet3D has no vertices. Initializing with uninitialized parameters.")
            self.planes = nn.UninitializedParameter(requires_grad=True)
        elif self.learn_plane_equations:
            if init_planes is not None:
                init_planes = init_planes.float().to(device)
                assert init_planes.shape[0] == self.n_planes, "The number of initial planes should match the number of planes."
                assert init_planes.shape[1] == 4, "The plane equations should have shape (P, 4)."
                init_planes = init_planes / torch.norm(init_planes[:, :3], dim=1, keepdim=True)
                self.planes = nn.Parameter(init_planes, requires_grad=True)
            else:
                self.planes = nn.Parameter(self.compute_planes(vertices, vertex_plane_assignments, self.vertex_normals), requires_grad=True)
        else:
            self.planes = None
            


        # Types of vertices
        # 0 constraints: vertices on no planes
        # 1 constraint: vertices on 1 plane
        # 2 constraints: vertices on 2 planes -> line-constrained points
        # 3 constraints: vertices on 3 planes -> fixed points
        self.max_constraint_distance = max_distance
        if len(vertices) > 0:
            self._remove_unsatisfiable_constraints(max_distance=self.max_constraint_distance)


    def _remove_unsatisfiable_constraints(self, max_distance: float):
        # correct constraints so projections are not too far
        for retry in range(10):
            
            fixed_vertex_mask = (self.vertex_plane_assignments[:, 2] != -1)
            line_constrained_vertex_mask = (self.vertex_plane_assignments[:, 1] != -1) & (~fixed_vertex_mask)
            plane_constrained_vertex_mask = (self.vertex_plane_assignments[:, 0] != -1) & (~fixed_vertex_mask) & (~line_constrained_vertex_mask)

            projected_points, distances = self.get()
            too_far = ~(distances < max_distance)
            # correct the constraints
            # for each fixed point that is further than max_distance, remove the last constraint
            self.vertex_plane_assignments[fixed_vertex_mask & too_far, 2] = -1
            # for each line-constrained point that is further than max_distance, remove the last constraint
            self.vertex_plane_assignments[line_constrained_vertex_mask & too_far, 1] = -1
            # for each plane-constrained point that is further than max_distance, remove the last constraint
            # self.vertex_plane_assignments[plane_constrained_vertex_mask & too_far, 0] = -1
            
            print(f"Number of corrected fixed points: {(fixed_vertex_mask & too_far).sum().item()}/{fixed_vertex_mask.sum().item()}")
            print(f"Number of corrected line-constrained points: {(line_constrained_vertex_mask & too_far).sum().item()}/{line_constrained_vertex_mask.sum().item()}")
            # print(f"Number of corrected plane-constrained points: {(plane_constrained_vertex_mask & too_far).sum().item()}/{plane_constrained_vertex_mask.sum().item()}")
            n_changed = (fixed_vertex_mask & too_far).sum() + (line_constrained_vertex_mask & too_far).sum() #+ (plane_constrained_vertex_mask & too_far).sum()
            if n_changed == 0:
                break
            print(f"Corrected {n_changed} constraints.")

        projected_points, distances = self.get()
        # assert not (distances > max_distance).any(), "The constraints are not satisfied."
        assert not projected_points.isnan().any(), "The projected points contain NaNs."
            

            

    def compute_planes(self, vertices, vertex_plane_assignments, normals=None):
        """Compute a plane equation from all the vertices that are assigned to the plane for each plane."""
        # unique_plane_ixes = torch.unique(vertex_plane_assignments[:, 0])
        planes = float('inf') * torch.zeros((self.n_planes, 4), dtype=torch.float, device=vertices.device)
        for plane_ix in range(self.n_planes):
            mask = vertex_plane_assignments[:, 0] == plane_ix
            plane_vertices = vertices[mask]
            planes[plane_ix] = fit_plane_torch(plane_vertices, normals[mask] if normals is not None else None)
        return planes
    
    def get_or_compute_planes(self, vertices = None, vertex_plane_assignments = None, normals=None):
        if self.learn_plane_equations:
            # normalize the plane equations
            planes = self.planes
            planes = planes / torch.norm(planes[:, :3], dim=1, keepdim=True)
            return planes
        else:
            return self.compute_planes(vertices, vertex_plane_assignments, normals=normals)
    

    def get_vertex_lines(self, vertex_plane_assignments, planes):
        """Compute the intersecting line for each pair of planes that shares at least one vertex.
        Args:
            vertices (torch.Tensor): The vertices with shape (V, 3)
            vertex_plane_assignments (torch.Tensor): The vertex plane assignments with shape (V, 2)
            planes (torch.Tensor): The plane equations with shape (P, 4)
        Returns:
            intersection_lines (torch.Tensor): The intersection lines with shape (V, 6)
        """
        vertex_plane_a = planes[vertex_plane_assignments[:, 0]]
        vertex_plane_b = planes[vertex_plane_assignments[:, 1]]
        p1, p2 = intersection_line_between_planes_batched(vertex_plane_a, vertex_plane_b)
        return  torch.cat([p1, p2], dim=1)  
        

    def get(self):
        """Get the vertices with the constraints applied. Todo: better way to replace values in tensors with grad?"""
        vertices = super(PlaneConstrainedVertexSet3D, self).get()
        planes = self.get_or_compute_planes(vertices, self.vertex_plane_assignments, self.vertex_normals) # shape (P, 4)
        vertices_projected = torch.empty_like(vertices)

        
        fixed_vertex_mask = (self.vertex_plane_assignments[:, 2] != -1)
        line_constrained_vertex_mask = (self.vertex_plane_assignments[:, 1] != -1) & (~fixed_vertex_mask)
        plane_constrained_vertex_mask = (self.vertex_plane_assignments[:, 0] != -1) & (~fixed_vertex_mask) & (~line_constrained_vertex_mask)

        unassigned_mask = ~fixed_vertex_mask & ~line_constrained_vertex_mask & ~plane_constrained_vertex_mask
        if unassigned_mask.sum() > 0:
            vertices_projected[unassigned_mask] = vertices[unassigned_mask]
        distances = torch.zeros(len(vertices), dtype=torch.float, device=vertices.device)
        # project plane-constrained to planes
        projected_plane_points, plane_distances = project_points_3d_to_3d_plane(vertices[plane_constrained_vertex_mask], planes[self.vertex_plane_assignments[plane_constrained_vertex_mask, 0]])
        # projected_plane_points_full_dim = torch.zeros_like(vertices)
        # projected_plane_points_full_dim[plane_constrained_vertex_mask] = projected_plane_points.float()
        # vertices = vertices * (~plane_constrained_vertex_mask[:,None]) + projected_plane_points_full_dim
        vertices_projected[plane_constrained_vertex_mask] = projected_plane_points
        distances[plane_constrained_vertex_mask] = plane_distances
        # project line-constrained to lines
        vertex_lines = self.get_vertex_lines(self.vertex_plane_assignments[line_constrained_vertex_mask][:, :2], planes)
        projected_line_points, line_distances = project_points_to_lines_torch(vertices[line_constrained_vertex_mask], vertex_lines)
        #projected_line_points, _ = project_points_to_lines_torch(vertices[line_constrained_vertex_mask], self.line_eqs_3d[self.point_line_assignment])
        # projected_line_points_full_dim = torch.zeros_like(vertices)
        # projected_line_points_full_dim[line_constrained_vertex_mask] = projected_line_points.float()
        # vertices = vertices * (~line_constrained_vertex_mask[:,None]) + projected_line_points_full_dim
        vertices_projected[line_constrained_vertex_mask] = projected_line_points
        distances[line_constrained_vertex_mask] = line_distances

        # compute fixed points
        vertex_lines = self.get_vertex_lines(self.vertex_plane_assignments[fixed_vertex_mask][:, :2], planes)
        fixed_points = line_plane_intersection_batched(vertex_lines, planes[self.vertex_plane_assignments[fixed_vertex_mask, 2]])
        fixed_point_dist = torch.linalg.norm(fixed_points - vertices[fixed_vertex_mask], dim=1)
        # fixed_points_full_dim = torch.zeros_like(vertices)
        # fixed_points_full_dim[fixed_vertex_mask] = fixed_points.float()
        # vertices = vertices * (~fixed_vertex_mask[:, None]) + fixed_points_full_dim
        vertices_projected[fixed_vertex_mask] = fixed_points
        distances[fixed_vertex_mask] = fixed_point_dist
        return vertices_projected, distances
    
    # def get_constraint_loss(self):
    #     # return 0
    #     vertices = self.get()
    #     _, distances_to_planes = project_points_3d_to_3d_plane(vertices[plane_constrained_vertex_mask], self.planes[self.point_plane_assignment])
    #     _, distances_to_lines = project_points_to_lines_torch(vertices[line_constrained_vertex_mask], self.line_eqs_3d[self.point_line_assignment])
    #     distances_to_fixed = torch.linalg.norm(vertices[fixed_vertex_mask] - self.fixed_points, dim=1)
    #     # return distances_to_fixed.sum()
    #     return distances_to_planes.sum() + distances_to_lines.sum()

    
    def get_angle_to_orthogonal(self, planes : torch.Tensor):
        if len(planes) == 3:
            planes_1, planes_2, planes_3 = planes
            # angle_1_2 = torch.acos(torch.clamp(torch.sum(planes_1[:, :3] * planes_2[:, :3], dim=-1), -1, 1))
            # angle_1_3 = torch.acos(torch.clamp(torch.sum(planes_1[:, :3] * planes_3[:, :3], dim=-1), -1, 1))
            # angle_2_3 = torch.acos(torch.clamp(torch.sum(planes_2[:, :3] * planes_3[:, :3], dim=-1), -1, 1))
            # angle_1_2 = torch.abs(90 - angle_1_2 * 180 / np.pi)
            # angle_1_3 = torch.abs(90 - angle_1_3 * 180 / np.pi)
            # angle_2_3 = torch.abs(90 - angle_2_3 * 180 / np.pi)
            # angle_to_90 = torch.maximum(angle_1_2, torch.minimum(angle_1_3, angle_2_3))
            return torch.maximum(self.get_angle_to_orthogonal((planes_1, planes_2)), torch.maximum(self.get_angle_to_orthogonal((planes_1, planes_3)), self.get_angle_to_orthogonal((planes_2, planes_3))))
        elif len(planes) == 2:
            planes_1, planes_2 = planes
            angle = torch.acos(torch.clamp(torch.sum(planes_1[:, :3] * planes_2[:, :3], dim=-1), -1, 1))
            # angle_deg = angle * 180 / np.pi
            angle_to_90 = torch.abs(90 - angle * 180 / np.pi)
            return angle_to_90
        elif len(planes) == 1:
            return torch.zeros(1, device=planes[0].device)
        else:
            raise ValueError("Invalid number of planes.")

    @torch.no_grad()
    def get_num_assigned_planes_per_vertex(self):
        
        fixed_vertex_mask = (self.vertex_plane_assignments[:, 2] != -1)
        line_constrained_vertex_mask = (self.vertex_plane_assignments[:, 1] != -1) & (~fixed_vertex_mask)
        plane_constrained_vertex_mask = (self.vertex_plane_assignments[:, 0] != -1) & (~fixed_vertex_mask) & (~line_constrained_vertex_mask)


        vertex_info = torch.zeros((len(self.vertices)), dtype=torch.int, device=self.vertices.device)
        vertex_info[fixed_vertex_mask] = 3
        vertex_info[line_constrained_vertex_mask] = 2
        vertex_info[plane_constrained_vertex_mask] = 1
        return vertex_info
    
    @torch.no_grad()
    def get_vertex_constraints(self, vertex_id):
        
        fixed_vertex_mask = (self.vertex_plane_assignments[:, 2] != -1)
        line_constrained_vertex_mask = (self.vertex_plane_assignments[:, 1] != -1) & (~fixed_vertex_mask)
        plane_constrained_vertex_mask = (self.vertex_plane_assignments[:, 0] != -1) & (~fixed_vertex_mask) & (~line_constrained_vertex_mask)


        if fixed_vertex_mask[vertex_id]:
            return self.vertex_plane_assignments[vertex_id]
        elif line_constrained_vertex_mask[vertex_id]:
            return self.vertex_plane_assignments[vertex_id, :2]
        elif plane_constrained_vertex_mask[vertex_id]:
            return self.vertex_plane_assignments[vertex_id, :1]
        else:
            return torch.tensor([], dtype=torch.int, device=self.vertices.device)
        

    @torch.no_grad()
    def update_vertex_constraints(self, vertex_id : int, plane_ids : torch.Tensor):
        plane_ids = plane_ids[plane_ids != -1]
        n_new_planes = len(plane_ids)
        assert n_new_planes <= 3, "Only up to 3 planes are supported."
        self.vertex_plane_assignments[vertex_id, :n_new_planes] = plane_ids
        self.vertex_plane_assignments[vertex_id, n_new_planes:] = -1

    
    @torch.no_grad()
    def project_vertex_to_constraints(self, point_3d: torch.Tensor, plane_ids : torch.Tensor):
        """Projects a vertex to the constraints given by the plane_ids
        Args:
            vertex_3d (torch.Tensor): The vertex to project with shape (3,)
            plane_ids (torch.Tensor): The plane ids to project to with shape (3,)
        Returns:
            torch.Tensor: The vertex projected to the planes, with shape (3,)
        """
        plane_ids = plane_ids.unique().long()
        planes = self.get_or_compute_planes(self.vertices, self.vertex_plane_assignments, self.vertex_normals) # shape (P, 4)
        if len(plane_ids) > 3:
            raise ValueError("Only up to 3 planes are supported.")
        if len(plane_ids) == 0:
            return point_3d
        if len(plane_ids) == 3:
            # return intersection beteween 3 planes
            p1, p2 = intersection_line_between_planes(planes[plane_ids[0]], planes[plane_ids[1]])
            return line_plane_intersection(torch.cat([p1, p2]), planes[plane_ids[2]])
        elif len(plane_ids) == 2:
            # project point on intersection line
            p1, p2 = intersection_line_between_planes(planes[plane_ids[0]], planes[plane_ids[1]])
            projected_point = project_points_to_lines_torch(point_3d[None], torch.cat([p1, p2])[None])[0]
            return projected_point
        elif len(plane_ids) == 1:
            # project point on plane
            return project_points_3d_to_3d_plane(point_3d[None], planes[plane_ids[0]])[0]


    def filter_mask_based(self, mask):
        """Filters the vertices based on a mask, only keeping the ones where the mask is True."""
        self.vertices = torch.nn.Parameter(self.vertices[mask], requires_grad=True)
        self.vertex_plane_assignments = self.vertex_plane_assignments[mask]

        old2new = torch.zeros(len(mask), dtype=torch.long, device=self.vertices.device)
        old2new[mask] = torch.arange(len(self.vertices), dtype=torch.long, device=self.vertices.device)
        return old2new


    def update_vertex_plane_assignments(self, vertex_plane_assignments):
        self.vertex_plane_assignments = vertex_plane_assignments
        self._remove_unsatisfiable_constraints(max_distance=self.max_constraint_distance)

    
    def get_params(self):
        params = [self.vertices]
        if self.learn_plane_equations:
            params.append(self.planes)
        return params
    
    def add_vertices(self, new_vertices: torch.Tensor, new_vertex_constraints: torch.Tensor, new_vertex_normals: torch.Tensor = None):
        """Add new vertices to the vertex set."""
        assert len(new_vertices) == len(new_vertex_constraints), "The number of vertices and constraints should match."
        assert new_vertices.shape == (len(new_vertices), 3), "The vertices should have shape (N, 3)."
        assert new_vertex_constraints.shape == (len(new_vertices), 3), "The constraints should have shape (N, 3)."

        vertices_values = torch.cat([self.vertices.data, new_vertices], dim=0)
        self.vertices = torch.nn.Parameter(vertices_values, requires_grad=True)
        self.vertex_plane_assignments = torch.cat([self.vertex_plane_assignments, new_vertex_constraints], dim=0)

        if self.vertex_normals is not None:
            if new_vertex_normals is not None:
                assert new_vertex_normals.shape == (len(new_vertices), 3), "The normals should have shape (N, 3)."
            else:
                new_vertex_normals = self.get_planes()[new_vertex_constraints[:, 0]][:, :3]
            self.vertex_normals = torch.cat([self.vertex_normals, new_vertex_normals], dim=0)



class PolygonSet3D(nn.Module):

    polygon_classes : torch.Tensor = None

    def __init__(self, vertices: torch.tensor, polygons: List[List[int]], planes: torch.Tensor, polygon2inner: Dict = None, colors: torch.Tensor = None, device : str ='cpu', polygon_meta_info : List[Dict] = None, merge_thresh: float = 0.02, config : PolygonFittingConfig = None):
        """
        Args:
            vertices: Tensor of shape (V, 3) where V is the number of vertices.
            triangles: Tensor of shape (T, 3) where T is the number of triangles.
            polygons: List of lists, where each sublist contains vertex indices defining a polygon.
            planes: Tensor of shape (P, 4) where P is the number of polygons, each plane is defined by (a, b, c, d) in ax + by + cz + d = 0.
            polygon2inner: Dictionary mapping polygon indices to inner polygon indices. Any inner polygon is a "hole" in the outer poylgon".
        """
        super(PolygonSet3D, self).__init__()

        self.config = config
        
        self.pool = None #mp.Pool(mp.cpu_count() // 8)

        self.device = device
        self.merge_thresh = merge_thresh

        if len(polygons) > 0:
            # close the polygons if they are not closed
            for i, polygon in enumerate(polygons):
                if polygon[0] != polygon[-1]:
                    polygons[i] = list(polygons[i]) + [polygon[0]]


            # vertex_polygon_assignments = -1 * torch.ones((len(vertices), 3), dtype=torch.int, device=device)
            # for i, polygon in enumerate(polygons):
            #     for v in polygon:
            #         for j in range(3):
            #             if vertex_polygon_assignments[v, j] == i:
            #                 break
            #             elif vertex_polygon_assignments[v, j] == -1:
            #                 vertex_polygon_assignments[v, j] = i
            #                 break
                
            polygons_closed = polygons  # List of vertex indices for each polygon
            is_outer_polygon = torch.ones(len(polygons), dtype=torch.bool, device=device)
            if polygon2inner is not None:
                for i, inner_polygons in polygon2inner.items():
                    is_outer_polygon[inner_polygons] = False

            polygon2inner = {k : polygon2inner.get(k, []) if polygon2inner is not None else [] for k in range(len(polygons))}
            # self.planes = planes.float().to(device)      # (P, 4)
            # self.vertices = VertexSet3D(torch.Tensor(vertices).float())
            polygon_colors = colors.to(device) if colors is not None else torch.zeros(len(polygons), dtype=torch.float, device=device)  # (P, 3)
            # self.retriangulate_polygons(vertices_3d=vertices, planes=planes)
            self.update_triangles(*self.triangulate_polygons(vertices, planes,  polygons_closed, is_outer_polygon, polygon2inner, polygon_colors))
            # self.retriangulate_polygons(vertices_3d=vertices, planes=planes, vertex_polygon_assignments=vertex_polygon_assignments)
            self.clean()
            # self.simplify_vertices(max_dist=1e-4)
            # self.simplify_edges()
            # self.retriangulate_polygons(delete_small=False)
            # if False:
                # self.simplify_vertices(self.merge_thresh)
            #     self.simplify_edges()
            # self.retriangulate_polygons(delete_small=False)
            #     self.attach_vertices_to_close_lines()
            #     self.compute_adjacency()
            self.simplify()
        
        # self.polygon_meta_info = polygon_meta_info
        else:
            self.vertices = PlaneConstrainedVertexSet3D(vertices, torch.zeros((len(vertices), 3), dtype=torch.int, device=device), device=device)


    def compute_adjacency(self):
        
        # compute adjacency list
        self.adjacency_list = [set() for _ in range(len(self.vertices.vertices))]
        for i, (v1, v2, v3) in enumerate(self.triangles):
            self.adjacency_list[v1].add(v2)
            self.adjacency_list[v1].add(v3)
            self.adjacency_list[v2].add(v1)
            self.adjacency_list[v2].add(v3)
            self.adjacency_list[v3].add(v1)
            self.adjacency_list[v3].add(v2)
        
        self.adjacency_list = [torch.tensor(list(adj)) for adj in self.adjacency_list]

        contour_edges = self.get_contour_edges()
        
        self.contour_adjacency_list = [set() for _ in range(len(self.vertices.vertices))]
        for i, (v1, v2) in enumerate(contour_edges):
            self.contour_adjacency_list[v1].add(v2)
            self.contour_adjacency_list[v2].add(v1)
        self.contour_adjacency_list = [torch.tensor(list(adj)) for adj in self.contour_adjacency_list]

        

    def update_triangles(self, triangles, new_vertices, triangle_polygons, vertex_polygons, polygon_colors, polygon_plane_eqs, polygon_classes=None):
        # remap triangle polygons to a continuous range
        unique_polygons, remap = torch.unique(torch.cat([triangle_polygons, vertex_polygons]), return_inverse=True)
        polygon_remap, vertex_remap = remap[:len(triangle_polygons)], remap[len(triangle_polygons):]
        if len(polygon_colors) != len(unique_polygons):
            polygon_colors = polygon_colors[unique_polygons]
        if len(polygon_plane_eqs) != len(unique_polygons):
            polygon_plane_eqs = polygon_plane_eqs[unique_polygons]
        if polygon_classes is not None and len(polygon_classes) != len(unique_polygons):
            polygon_classes = polygon_classes[unique_polygons]
        device = self.device
        new_vertex_polygon_assignments = -1 * torch.ones((len(new_vertices), 3), dtype=torch.int, device=device)
        new_vertex_polygon_assignments[:, 0] = vertex_remap
        # vertex_normals = planes[vertex_polygons][:, :3]  # (V, 3)
        assert new_vertex_polygon_assignments.max().item() + 1 == len(torch.unique(new_vertex_polygon_assignments[new_vertex_polygon_assignments != -1])), "The remapped polygons do not match the original polygons."
        self.vertices = PlaneConstrainedVertexSet3D(new_vertices, new_vertex_polygon_assignments, init_planes=polygon_plane_eqs, device=device).to(device)
        assert triangles.max().item() < len(new_vertices), "The triangles contain invalid vertex indices."
        # assert len(torch.unique(triangles.sort(dim=1).values, dim=0)) == len(triangles), "The triangles contain duplicates."
        self.triangles = triangles.to(device).long()
        assert len(triangle_polygons) == len(triangles), "The number of triangles should match the number of triangle polygon assignments."
        self.triangle_polygons = polygon_remap.to(device).long()
        assert len(polygon_colors) == self.vertices.n_planes, "The number of colors should match the number of polygons."
        self.polygon_colors = polygon_colors.to(device)
        if polygon_classes is not None:
            self.polygon_classes = polygon_classes.to(device)
        else:
            self.polygon_classes = None
        # assert len(polygon_plane_eqs) == self.vertices.n_planes, "The number of normals should match the number of polygons."
        # self.polygon_plane_eqs = polygon_plane_eqs.to(device)
        self.clear_cache()
        # self.flip_misaligned_triangles()

        # self.compute_adjacency()



    # def update_polygon_paths(self):
    #     polygon_vertices_2d = self.get_polygon_vertices_2d()
    #     self.polygon_paths_mpl = [mplPath.Path(polygon_vertices_2d[i].detach().cpu().numpy()) for i in range(len(self.polygons_closed))]

    

    @staticmethod
    def from_polygon_info(polygon_info, vertices, simplified=True, device='cpu', merge_thresh=0.02, config : PolygonFittingConfig = None):
        
        n_occurrences = np.zeros(len(vertices))
        for i, polygon in enumerate(polygon_info):
            for v in polygon['contours'][0]:
                n_occurrences[v] += 1

        polygon_contours = []
        inner_polygon_contours = {}
        polygon_planes = []
        polygon_colors = []
        for v in polygon_info:
            polygon_planes.append(v['plane_eq'])
            polygon_colors.append(v['color'])
        
            polygon_v_contours = []

            for k, contour_ixes in enumerate(v['contours']):
                # contour_vertices_3D = vertices[contour_ixes]
                # contour_vertices_2D = project_points_3d_to_2d_plane_based(contour_vertices_3D.cpu().numpy(), v['plane_eq'])
                # contour_len = np.linalg.norm(np.roll(contour_vertices_2D, -1, axis=0) - contour_vertices_2D, axis=1).sum()
                # if k != 0 and contour_len < 1:
                #     continue
                if simplified and "contour_masks_rdp" in v:
                    # do not throw away points that are on multiple contours to avoid losing shared vertices

                    # contour_mask = approximate_polygon_rdp(contour_vertices_2D, epsilon=1e-6, thresh=1e-6)
                    contour_mask = np.array(v['contour_masks_rdp'][k])
                    is_on_multiple = n_occurrences[contour_ixes] > 1
                    polygon_v_contours.append(np.array(contour_ixes)[contour_mask | is_on_multiple])
                else:
                    polygon_v_contours.append(np.array(contour_ixes))
                
            polygon_contours.append(polygon_v_contours[0])
            inner_polygon_contours[len(polygon_contours) - 1] = polygon_v_contours[1:]

        polygon2inner = {}
        for i, inner_contours in inner_polygon_contours.items():
            polygon2inner[i] = []
            for inner_contour in inner_contours:
                polygon2inner[i].append(len(polygon_contours))
                polygon_contours.append(inner_contour)
                polygon_planes.append(polygon_planes[i])

        # import matplotlib.pyplot as plt     
        # plt.hist(contour_lens, bins=100)
            
        
        return PolygonSet3D(vertices, polygon_contours, torch.tensor(polygon_planes), polygon2inner=polygon2inner, colors=torch.tensor(polygon_colors), device=device, polygon_meta_info=polygon_info, merge_thresh=merge_thresh, config=config)


    def to_polygon_info(self):
        vertices = self.get_vertices().detach().cpu().numpy()
        planes = self.get_planes()
        polygon_info = []
        for i, (polygon_fitted, polygon_original) in enumerate(zip(self.polygons_closed, self.polygon_meta_info)):
            if self.is_outer_polygon[i] and len(polygon_fitted) > 2:
                contours = [polygon_fitted] + [self.polygons_closed[j] for j in self.polygon2inner.get(i, []) if len(self.polygons_closed[j]) > 2]
                new_polygon_info = polygon_original.copy()
                new_polygon_info.update({
                    'contours': contours,
                    'plane_eq': planes[i].tolist(),
                    'color': self.polygon_colors[i].tolist()
                })
                polygon_info.append(new_polygon_info)
        return polygon_info, vertices


    def get_vertices(self, return_distances=False, cache=True):
        if return_distances:
            return self.vertices.get()
        else:
            return self.vertices.get()[0]
        # if cache:
        #     if hasattr(self, 'vertices_cache'):
        #         if return_distances:
        #             return self.vertices_cache
        #         else:
        #             return self.vertices_cache[0]
        #     else:
        #         self.vertices_cache = self.vertices.get()
        #         if return_distances:
        #             return self.vertices_cache
        #         else:
        #             return self.vertices_cache[0]
        # else:
        #     return self.vertices.get()
        
    def get_planes(self):
        vertices = self.get_vertices()
        return self.vertices.get_or_compute_planes(vertices, self.vertices.vertex_plane_assignments, self.vertices.vertex_normals)
    

    # def points_in_polygons_check(self, points: torch.Tensor, point_polygons: List[int], return_dist : bool = False, precision: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Check if points are inside polygons.
        
    #     Args:
    #         points: Tensor of shape (N, 3), representing N 3D points.
    #         point_polygons: List of polygon indices for each point of length N.
    #         precision: Float, threshold for distance to classify points as inside or outside.
        
    #     Returns:
    #         is_inside: Tensor of shape (N,), boolean indicating if each point is inside its polygon.
    #         signed_distance: Tensor of shape (N,), signed distance for each point to its assigned polygon plane.
    #     """
    #     points_2d = self.project_points_to_planes_3d_to_2d(points, point_polygons)
    #     is_inside = torch.zeros(len(points), dtype=torch.bool)
    #     sq_distances = torch.zeros(len(points), dtype=torch.float32) if return_dist else None
    #     for i in torch.unique(point_polygons):
    #         polygon_path = self.polygon_paths_mpl[i]
    #         mask = point_polygons == i
    #         points_i = points_2d[mask]
    #         # if len(points_i) == 0:
    #             # continue
    #         is_inside_polygon_i = polygon_path.contains_points(points_i[:, :2].detach().numpy(), radius=precision)
    #         is_inside[mask] = torch.tensor(is_inside_polygon_i, dtype=torch.bool)
    #         if return_dist:
    #             polygon_2d = self.project_points_to_planes_3d_to_2d(self.get_vertices()[self.polygons_closed[i]], torch.Tensor([i]).repeat(len(self.polygons_closed[i])))
    #             sq_distances[mask] = compute_sq_dist_vertices_to_polygon(points_i[:, :2], polygon_2d)
        
    #     return is_inside, sq_distances


    # def get_polygon_vertices_2d_nocache(self):
    #     polygon_vertex_indices = torch.cat([torch.Tensor(p).long() for p in self.polygons_closed])
    #     polygon_ids = torch.cat([torch.Tensor([i]).repeat(len(p)) for i, p in enumerate(self.polygons_closed)])
    #     projected_vertices = self.project_points_to_planes_3d_to_2d(self.get_vertices()[polygon_vertex_indices], polygon_ids)
    #     cutpoints = (polygon_ids[1:] != polygon_ids[:-1]).nonzero().squeeze(-1) + 1
    
    #     polygon_vertices = [projected_vertices[start:end] for start, end in zip([0] + list(cutpoints), list(cutpoints) + [len(polygon_vertex_indices)])]
        
    #     for i, p in enumerate(self.polygons_closed):
    #         if len(p) == 0:
    #             # insert an empty polygon
    #             polygon_vertices.insert(i, torch.empty((0, 2)))
        
    #     assert len(polygon_vertices) == len(self.polygons_closed)
    #     for i in range(len(polygon_vertices)):
    #         assert len(polygon_vertices[i]) == len(self.polygons_closed[i])
    #     return polygon_vertices
    #     # return [self.project_points_to_planes_3d_to_2d(self.get_vertices()[p], torch.Tensor([i]).repeat(len(p))) for i, p in enumerate(self.polygons_closed)]
    
    def get_polygon_vertices_2d(self, cache=True):
        if cache:
            if hasattr(self, 'polygon_vertices_cache'):
                return self.polygon_vertices_cache
            else:
                self.polygon_vertices_cache = self.get_polygon_vertices_2d_nocache()
                return self.polygon_vertices_cache
        else:
            return self.get_polygon_vertices_2d_nocache()
        

    # def points_in_polygons_check_fast(self, points: torch.Tensor, point_polygons: torch.Tensor, return_dist : bool = False, precision: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """ same as above but uses a faster implementation"""
    #     points_2d = self.project_points_to_planes_3d_to_2d(points, point_polygons)
    #     polygon_vertices_2d = self.get_polygon_vertices_2d()
        
    #     sorted_vals, order = point_polygons.sort()
    #     sorted_points = points_2d[order]
    #     delta = sorted_vals[1:] != sorted_vals[:-1]
    #     cutpoints = delta.nonzero().squeeze(-1) + 1
        
    #     is_inside = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    #     sq_distances = torch.zeros(len(points), dtype=torch.float32, device=points.device) if return_dist else None

    #     for start, end in zip([0] + list(cutpoints), list(cutpoints) + [len(point_polygons)]):
    #         assert point_polygons[order[start:end]].unique().shape[0] == 1
        
    #         i = point_polygons[order[start]]
            
    #         polygon_2d = polygon_vertices_2d[i]
    #         if len(polygon_2d) <= 2:
    #             continue

    #         points_i = sorted_points[start:end]
    #         is_inside_polygon_i = self.polygon_paths_mpl[i].contains_points(points_i[:, :2].cpu().detach().numpy(), radius=precision)
    #         is_inside[order[start:end]] = torch.tensor(is_inside_polygon_i, dtype=torch.bool, device=points.device)
    #         if return_dist:
    #             sq_distances[order[start:end]] = compute_sq_dist_vertices_to_polygon(points_i[:, :2], polygon_2d)
            
    #         hole_polygons = self.polygon2inner[i.item()]
    #         for hole_polygon in hole_polygons:
    #             hole_polygon_2d = polygon_vertices_2d[hole_polygon]
    #             if len(hole_polygon_2d) <= 2:
    #                 continue
    #             is_inside_hole = self.polygon_paths_mpl[hole_polygon].contains_points(points_i[:, :2].cpu().detach().numpy(), radius=precision)
    #             is_inside[order[start:end]] = is_inside[order[start:end]] & (~torch.tensor(is_inside_hole, dtype=torch.bool, device=points.device))
    #             if return_dist:
    #                 sq_distances[order[start:end]] = torch.min(sq_distances[order[start:end]], compute_sq_dist_vertices_to_polygon(points_i[:, :2], hole_polygon_2d))
    #         # res.append(T[order[start:end]])
    #         # inverse_map.append(order[start:end])
    #         # values[i] = F[start]

    #     return is_inside, sq_distances

    
    
    # def distances_to_polygon(self, points: torch.Tensor, point_polygons: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """ same as above but uses a faster implementation"""
    #     points_2d = self.project_points_to_planes_3d_to_2d(points, point_polygons)
    #     polygon_vertices_2d = self.get_polygon_vertices_2d()
        
    #     sorted_vals, order = point_polygons.sort()
    #     sorted_points = points_2d[order]
    #     delta = sorted_vals[1:] != sorted_vals[:-1]
    #     cutpoints = delta.nonzero().squeeze(-1) + 1
        
    #     sq_distances = torch.zeros(len(points), dtype=torch.float32, device=points.device)

    #     if len(order) == 0:
    #         return sq_distances

    #     for start, end in zip([0] + list(cutpoints), list(cutpoints) + [len(point_polygons)]):
    #         assert point_polygons[order[start:end]].unique().shape[0] == 1
            
    #         i = point_polygons[order[start]]
            
    #         polygon_2d = polygon_vertices_2d[i]
    #         if len(polygon_2d) <= 2:
    #             continue

    #         points_i = sorted_points[start:end]
    #         sq_distances[order[start:end]] = compute_sq_dist_vertices_to_polygon(points_i[:, :2], polygon_2d)
            
    #         hole_polygons = self.polygon2inner[i.item()]
    #         for hole_polygon in hole_polygons:
    #             hole_polygon_2d = polygon_vertices_2d[hole_polygon]
    #             if len(hole_polygon_2d) <= 2:
    #                 continue
    #             sq_distances[order[start:end]] = torch.min(sq_distances[order[start:end]], compute_sq_dist_vertices_to_polygon(points_i[:, :2], hole_polygon_2d))

    #     return sq_distances


    def project_points_to_planes_3d_to_2d(self, points: torch.Tensor, point_polygons: List[int], planes: torch.Tensor) -> torch.Tensor:
        """
        Projects points onto the planes of their assigned polygons.
        
        Args:
            points: Tensor of shape (N, 3).
            point_polygons: List of polygon indices for each point of length N.
            planes: Tensor of shape (P, 4) where P is the number of polygons, each plane is defined by (a, b, c, d) in ax + by + cz + d = 0.
        
        Returns:
            projected_points: Tensor of shape (N, 3), projected points on polygon planes.
        """
        projected_points = torch.empty_like(points[:, :2])
        for i, plane in enumerate(planes):
            points_i = points[point_polygons == i]
            if len(points_i) == 0:
                continue
            projected_points[point_polygons == i] = project_points_3d_to_2d_plane_based_torch(points_i, plane)
        return projected_points
    
    
    @torch.no_grad()
    def triangulate_polygons(self, vertices_3d, planes, polygons_closed, is_outer_polygon, polygon2inner, original_polygon_colors):
        """Export the polygon set as a triangle mesh."""
        vertices_3d = vertices_3d.detach().cpu().numpy()
    
        vertices = []
        triangles = []
        triangle_polygons = []
        vertex_polygons = []
        polygon_planes = []
        polygon_colors = []

        for i, polygon in enumerate(polygons_closed):
            if not is_outer_polygon[i] or len(polygon) <= 2:
                continue
            plane = planes[i].detach().cpu().numpy()
            contours = [polygon] + [polygons_closed[p] for p in polygon2inner[i] if len(polygons_closed[p]) > 2]
            
            # vertices_2d = project_points_3d_to_2d_plane_based(vertices_3d[polygon], plane)
            # inner_vertices_2d = [project_points_3d_to_2d_plane_based(vertices_3d[polygons_closed[p]], plane) for p in polygon2inner[i] if len(polygons_closed[p]) > 2]
            # new_points_2d, triangles_2d, triangle_ixes = triangulate_polygon(vertices_2d, holes=inner_vertices_2d)
            # new_points_3d = project_points_2d_to_3d_plane_based(new_points_2d, plane)

            # triangle_ixes = np.array(triangle_ixes).astype(np.int32)

            new_points_3d, triangle_ixes = triangulate_3d_polygon((vertices_3d, plane, contours))

            triangles += list(triangle_ixes + len(vertices))
            vertices += list(new_points_3d)
            triangle_polygons += list(np.array([i] * len(triangle_ixes)))
            vertex_polygons += list(np.array([i] * len(new_points_3d)))
            polygon_planes.append(plane)
            polygon_colors.append(original_polygon_colors[i])

        mesh_vertices = torch.from_numpy(np.array(vertices)).float()
        mesh_triangles = torch.from_numpy(np.array(triangles)).long()
        mesh_triangle_polygons = torch.from_numpy(np.array(triangle_polygons)).long()
        mesh_vertex_polygons = torch.from_numpy(np.array(vertex_polygons)).long()
        mesh_polygon_planes = torch.from_numpy(np.array(polygon_planes)).float()
        mesh_polygon_colors = torch.stack(polygon_colors)

        assert mesh_triangles.max().item() < len(mesh_vertices), "The triangles contain invalid vertex indices."

        return mesh_triangles, mesh_vertices, mesh_triangle_polygons, mesh_vertex_polygons, mesh_polygon_colors, mesh_polygon_planes


    def retriangulate_polygons(self, vertices_3d : torch.Tensor = None, planes : torch.Tensor = None, _triangles : torch.Tensor = None, _triangle_polygons : torch.Tensor = None, delete_small=True):
        """Export the polygon set as a triangle mesh."""
        self.flip_misaligned_triangles()
        if vertices_3d is None:
            self.clear_cache()
            vertices_3d = self.get_vertices().detach().cpu().numpy()
        else:
            vertices_3d = vertices_3d.detach().cpu().numpy()
        if planes is None:
            planes = self.vertices.get_or_compute_planes().detach().cpu().numpy()
        else:
            planes = planes.detach().cpu().numpy()
        if _triangles is None:
            _triangles = self.triangles
        if _triangle_polygons is None:
            _triangle_polygons = self.triangle_polygons

        # if vertex_polygons is None:
        #     vertex_polygons = self.vertices.vertex_plane_assignments

        # contour_edges = self.get_contour_edges()

        # contour_edges_by_polygon = {}
        # for i, j in contour_edges.cpu().numpy():
        #     polygons = set(vertex_polygons[i].tolist()) & set(vertex_polygons[j].tolist()) - {-1}
        #     for p in polygons:
        #         if p not in contour_edges_by_polygon:
        #             contour_edges_by_polygon[p] = []
        #         contour_edges_by_polygon[p].append((i, j))

        # all_triangle_edges = torch.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=-1)
        # triangles = triangles.sort(dim=1).values
        # triangles = torch.unique(triangles, dim=0)

        n_polygons = _triangle_polygons.max().item() + 1

        vertices = []
        triangles = []
        triangle_polygons = []
        vertex_polygons = []
        colors = []
        classes = []
        plane_eqs = []
        for polygon in tqdm(range(n_polygons), desc="Retriangulating polygons", total=n_polygons):
            
            polygon_triangles = _triangles[_triangle_polygons == polygon]
            if len(polygon_triangles) == 0:
                continue
                
            plane_eq = planes[polygon].copy()

            new_points_3d, triangle_ixes = retriangulate_3d_triangles(vertices_3d, polygon_triangles, plane_eq, polygon, delete_small=delete_small)

            new_triangle_area = 0 if new_points_3d is None else sum([np.linalg.norm(np.cross(new_points_3d[triangle_ixes[i][1]] - new_points_3d[triangle_ixes[i][0]], new_points_3d[triangle_ixes[i][2]] - new_points_3d[triangle_ixes[i][0]])) for i in range(len(triangle_ixes))])
            old_triangle_area = sum([np.linalg.norm(np.cross(vertices_3d[polygon_triangles[i][1]] - vertices_3d[polygon_triangles[i][0]], vertices_3d[polygon_triangles[i][2]] - vertices_3d[polygon_triangles[i][0]])) for i in range(len(polygon_triangles))])
              
            
            if new_triangle_area  / (old_triangle_area + 1e-3) < 0.6 and new_triangle_area > 4 and len(polygon_triangles) < 20:
                print("Something odd happened.")
                
                vertices_2d = project_points_3d_to_2d_plane_based(vertices_3d[polygon_triangles.cpu().numpy()].reshape(-1, 3), plane_eq)
                vertices_2d = vertices_2d.reshape(-1, 3, 2)
                import matplotlib.pyplot as plt

                # vertices_2d = project_points_3d_to_2d_plane_based(vertices_3d[edges].reshape(-1, 3), plane_eq)
                # vertices_2d = vertices_2d.reshape(-1, 2, 2)
                # for edge in vertices_2d:
                #     plt.plot(edge[:, 0], edge[:, 1], 'k-')
                # plt.gca().set_aspect('equal', adjustable='box')
                # plt.savefig('tmp_triangulated.png')
                # plt.close()
                
                plt.figure(figsize=(10, 10))
                for i in range(len(vertices_2d)):
                    for j in range(3):
                        v1, v2 = vertices_2d[i, j], vertices_2d[i, (j + 1) % 3]
                        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig('tmp_triangulated1.png')
                plt.close()

                
                if new_points_3d is None or len(new_points_3d) == 0:
                    new_points_3d_in_2d = np.zeros((0, 2))
                else:
                    new_points_3d_in_2d = project_points_3d_to_2d_plane_based(new_points_3d, plane_eq)
                
                for triangle_ix in triangle_ixes:
                    triangle_edges = np.array([[triangle_ix[0], triangle_ix[1], triangle_ix[2]], [triangle_ix[2], triangle_ix[0], triangle_ix[1]]])
                    plt.plot(new_points_3d_in_2d[triangle_edges, 0], new_points_3d_in_2d[triangle_edges, 1], 'k-')
    
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title(f"Polygon {polygon}, new area: {new_triangle_area}, old area: {old_triangle_area}")
                plt.savefig('tmp_triangulated2.png')
                plt.close()  
                
                # all_vertices_2d = project_points_3d_to_2d_plane_based(vertices_3d, plane_eq)
                # # import matplotlib.pyplot as plt

                # plt.figure(figsize=(10, 10))
                # for _poly in contours:
                #     for i, j in zip(_poly, np.roll(_poly, -1)):
                #         v1, v2 = all_vertices_2d[i], all_vertices_2d[j]
                #         plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-')

                # plt.gca().set_aspect('equal', adjustable='box')
                # plt.savefig('tmp_triangulated2.png')
                # plt.close()
                
                new_points_3d, triangle_ixes = retriangulate_3d_triangles(vertices_3d, polygon_triangles, plane_eq, polygon, plot=True)

                # new_points_3d, triangle_ixes = triangulate_3d_polygon((vertices_3d, plane_eq, contours))

            if new_points_3d is None or len(triangle_ixes) == 0:
                print(f"Polygon {polygon} has no valid triangles.")
                continue
            assert triangle_ixes.max() < len(new_points_3d), "The triangles contain invalid vertex indices."
            
            triangles += list(triangle_ixes + len(vertices))
            vertices += list(new_points_3d)
            triangle_polygons += list(np.array([polygon] * len(triangle_ixes)))
            vertex_polygons += list(np.array([polygon] * len(new_points_3d)))
            colors += [self.polygon_colors[polygon]]
            if self.polygon_classes is not None:
                classes += [self.polygon_classes[polygon]]
            plane_eqs.append(planes[polygon])

        mesh_vertices = torch.from_numpy(np.array(vertices)).float()
        mesh_triangles = torch.from_numpy(np.array(triangles)).long()
        mesh_triangle_polygons = torch.from_numpy(np.array(triangle_polygons)).long()
        mesh_vertex_polygons = torch.from_numpy(np.array(vertex_polygons)).long()
        triangle_colors = torch.stack(colors, dim=0)
        if len(classes) > 0:
            polygon_classes = torch.stack(classes, dim=0)
        else:
            polygon_classes = None
        plane_eqs = torch.from_numpy(np.array(plane_eqs)).float()

        assert mesh_triangles.max().item() < len(mesh_vertices), "The triangles contain invalid vertex indices."

        self.update_triangles(mesh_triangles, mesh_vertices, mesh_triangle_polygons, mesh_vertex_polygons, triangle_colors, plane_eqs, polygon_classes)
        self.simplify_vertices(max_dist=self.merge_thresh)
        self.flip_misaligned_triangles()
        print(f"vertices before: {len(vertices_3d)}, vertices after: {len(mesh_vertices)}, n polygons: {len(plane_eqs)}")


    @torch.no_grad()
    def triangulate_polygons_parallel(self, vertices_3d, planes):
        """Export the polygon set as a triangle mesh. Uses python multiprocessing pool."""
        
        
        vertices_3d = vertices_3d.detach().cpu().numpy()
        args = []
        for i, polygon in enumerate(self.polygons_closed):
            if not self.is_outer_polygon[i] or len(polygon) <= 2:
                continue
            plane = planes[i].detach().cpu().numpy()
            inner_polygons = [self.polygons_closed[p] for p in self.polygon2inner[i] if len(self.polygons_closed[p]) > 2]
            args.append((vertices_3d, polygon, plane, inner_polygons))

        results = self.pool.map(triangulate_3d_polygon, args)

        vertices = []
        triangles = []
        triangle_polygons = []
        vertex_polygons = []

        for i, (new_points_3d, triangle_ixes) in enumerate(results):
            triangle_ixes = np.array(triangle_ixes).astype(np.int32)
            triangles += list(triangle_ixes + len(vertices))
            vertices += list(new_points_3d)
            triangle_polygons += list(np.array([i] * len(triangle_ixes)))
            vertex_polygons += list(np.array([i] * len(new_points_3d)))

        mesh_vertices = torch.from_numpy(np.array(vertices)).float()
        mesh_triangles = torch.from_numpy(np.array(triangles)).long()
        mesh_triangle_polygons = torch.from_numpy(np.array(triangle_polygons)).long()
        mesh_vertex_polygons = torch.from_numpy(np.array(vertex_polygons)).long()

        return mesh_triangles, mesh_vertices, mesh_triangle_polygons, mesh_vertex_polygons


    @torch.no_grad()
    def export_triangle_mesh(self, classes : torch.Tensor = None):
        """Export the polygon set as a triangle mesh."""
        vertices = self.get_vertices().detach().cpu().numpy()
        self.flip_misaligned_triangles()
        triangles = self.triangles.detach().cpu().numpy()
        triangle_polygons = self.triangle_polygons.detach().cpu().numpy()

        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()

        if classes is not None:
            if not isinstance(classes, torch.Tensor):
                classes = torch.tensor(classes)
            classes = classes.to(self.device)


        # duplicate vertices for each polygon so we get per-face colored mesh
        for polygon in np.unique(triangle_polygons):
            if polygon < 0:
                continue
            if classes is not None and self.polygon_classes is not None and not torch.isin(self.polygon_classes[polygon], classes):
                continue
            mask = triangle_polygons == polygon
            # triangles[mask] += len(vertices) * polygon
            triangles_subset = triangles[mask].copy()
            vertex_subset, vertex_remap = np.unique(triangles_subset, return_inverse=True)
            triangles_subset = vertex_remap.reshape(triangles_subset.shape)
            
            submesh = o3d.geometry.TriangleMesh()
            submesh.vertices = o3d.utility.Vector3dVector(vertices[vertex_subset].astype(np.float64))
            submesh.triangles = o3d.utility.Vector3iVector(triangles_subset.astype(np.int32))
            submesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(stable_plane_color(int(polygon))[None], len(vertex_subset), axis=0))
            mesh += submesh
        mesh.compute_vertex_normals()
        return mesh
    

    def find_mergeable_vertices(self, max_dist=0.01, k=10):
        """merges vertices that are closer than max_dist"""
        vertices = self.get_vertices().detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(vertices)
        distances, indices = nbrs.kneighbors(vertices)
        
        mergeable_pairs = {}
        mask = (distances < max_dist) & (np.arange(len(vertices))[:, None] != indices)
        i_indices, j_indices = np.where(mask)
        for i, j in zip(i_indices, j_indices):
            mergeable_pairs[tuple(sorted([i, indices[i, j]]))] = distances[i, j]

        mergeable_pairs = sorted(mergeable_pairs.keys(), key=lambda x: mergeable_pairs[x])
        return mergeable_pairs
    
    def get_vertex_knn_indices(self, vertices, k=10):
        from pytorch3d.ops.knn import knn_points, knn_gather
        lengths = (torch.ones_like(vertices[:, 0]) * len(vertices)).long()
        x_nn = knn_points(vertices.unsqueeze(0), vertices.unsqueeze(0), lengths1=lengths, lengths2=lengths, norm=2, K=k)
        # distances = x_nn.dists[0]  # (N, P1)
        idx = x_nn.idx[0] # (N, P1)
        return idx
    
    def find_mergeable_vertices_fast(self, max_dist=0.01, k=10):
        """returns vertex pairs that are closer than max_dist"""
        with torch.no_grad():
            vertices = self.get_vertices().detach()
            idx = self.get_vertex_knn_indices(vertices, k=k)
        
        pairs_flat = torch.stack([torch.arange(len(vertices), device=idx.device).unsqueeze(1).repeat(1, k), idx], dim=-1).reshape(-1, 2)
        pairs_flat = pairs_flat.sort(dim=-1).values
        unique_pairs = torch.unique(pairs_flat, dim=0)
        unique_pair_distances = (vertices[unique_pairs[:, 0]] - vertices[unique_pairs[:, 1]]).norm(dim=-1)
        mask  = (unique_pair_distances < max_dist) & (unique_pairs[:, 0] != unique_pairs[:, 1])
        unique_pairs_filtered = unique_pairs[mask]
        sort_index = unique_pair_distances[mask].argsort()
        unique_pairs_filtered = unique_pairs_filtered[sort_index]
        return unique_pairs_filtered
    

    @torch.no_grad()
    def simplify(self):
        # vertices = self.get_vertices().detach()
        # self.simplify_vertices(self.merge_thresh)
        # new_vertices = self.get_vertices().detach()
        # assert (vertices - new_vertices).norm(dim=-1).max() < 1e-2
        # self.simplify_vertices()
        self.simplify_edges()
        self.retriangulate_polygons()
        self.clear_cache()
        self.compute_adjacency()
        self.attach_vertices_to_close_lines()


    @torch.no_grad()
    def simplify_vertices(self, plot=False, max_dist=0.005, max_angle_to_90=70, recursive_depth=3):
        """Merges vertices that are close to each other. @TODO: add mutual constraints, check that adding the constraint does not move the vertex too much"""
        # vertex_merge_candidates = self.find_mergeable_vertices(max_dist=max_dist)
        
        print("Simplifying vertices")
        self.clear_cache()

        vertex_merge_candidates = self.find_mergeable_vertices_fast(max_dist=max_dist)
        
        vertices = self.get_vertices().detach()
        planes = self.get_planes().detach()
        n_planes_per_vertex = self.vertices.get_num_assigned_planes_per_vertex()

        # special_ix = 7822
        # print("Vertex", vertices[special_ix])
        # print("Planes", planes[self.vertices.vertex_plane_assignments[special_ix]])
        # print("Constraints", self.vertices.vertex_plane_assignments[special_ix])
        
        projection_dists = float("inf") * torch.zeros(len(vertex_merge_candidates), device=vertices.device)
        updated_constraints = torch.zeros((len(vertex_merge_candidates), 3), device=vertices.device, dtype=torch.long)

        vertex_constraints = self.vertices.vertex_plane_assignments.clone()
        _old_vertex_constraints_numpy = vertex_constraints.cpu().numpy()
        constraints_1 = vertex_constraints[vertex_merge_candidates[:, 0]]
        constraints_2 = vertex_constraints[vertex_merge_candidates[:, 1]]

        # constraints_match_matrix = (constraints_1[:, :, None] == constraints_2[:, None, :])
        # c1_subset_of_c2 = constraints_match_matrix[:, :, 0].sum(dim=-1) == (constraints_1 != -1).sum(dim=-1)
        # c2_subset_of_c1 = constraints_match_matrix[:, :, 1].sum(dim=-1) == (constraints_2 != -1).sum(dim=-1)
        # we don't merge if one constraint set is a subset of another and they are fixed points
        # no_subset_mask = (c1_subset_of_c2 | c2_subset_of_c1) & (~((n_planes_per_vertex[vertex_merge_candidates[:, 0]] == 3) & (n_planes_per_vertex[vertex_merge_candidates[:, 1]] == 3)))
        # vertex_merge_candidates = vertex_merge_candidates[c1_subset_of_c2 & c2_subset_of_c1]
        joint_constraints = torch.cat([constraints_1, constraints_2], dim=-1)
        joint_constraints = joint_constraints.sort(dim=-1).values

        n_joint_constraints = ((joint_constraints[:,1:] != joint_constraints[:,:-1]) & (joint_constraints[:,:-1] != -1)).sum(axis=1)+1

        first_has_more_planes = (n_planes_per_vertex[vertex_merge_candidates[:, 0]] > n_planes_per_vertex[vertex_merge_candidates[:, 1]]).long()
        vertex_with_max_planes = vertex_merge_candidates[torch.arange(len(vertex_merge_candidates), dtype=torch.long, device=vertex_merge_candidates.device), (1 - first_has_more_planes).long()]
        vertex_with_min_planes = vertex_merge_candidates[torch.arange(len(vertex_merge_candidates), dtype=torch.long, device=vertex_merge_candidates.device), first_has_more_planes.long()]
        assert (n_planes_per_vertex[vertex_with_max_planes] >= n_planes_per_vertex[vertex_with_min_planes]).all()
        assert (vertex_with_max_planes != vertex_with_min_planes).all()

        # compute the distances to the new fixed points
        fixed_point_candidates = (n_joint_constraints == 3)
        fp_constraints = joint_constraints[fixed_point_candidates]
        fp_constraints = fp_constraints[torch.cat([fp_constraints[:, :1] != -1, (fp_constraints[:, 1:] != fp_constraints[:, :-1])], dim=1)]
        c1, c2, c3 = fp_constraints.view(-1, 3).T
        assert (c1 != c2).all() and (c1 != c3).all() and (c2 != c3).all()
        assert (c1 != -1).all() and (c2 != -1).all() and (c3 != -1).all()
        
        p1, p2 = intersection_line_between_planes_batched(planes[c1], planes[c2])
        fixed_points = line_plane_intersection_batched(torch.cat([p1, p2], dim=-1), planes[c3])
        dist1 = (fixed_points - vertices[vertex_with_max_planes[fixed_point_candidates]]).norm(dim=-1)
        dist2 = (fixed_points - vertices[vertex_with_min_planes[fixed_point_candidates]]).norm(dim=-1)
        fixed_point_dist = torch.maximum(dist1, dist2)

        # check that the smallest angle between the planes is not too small 
        angle_to_90 = self.vertices.get_angle_to_orthogonal((planes[c1], planes[c2], planes[c3]))
        # we must have at most min_angle to 90 degrees
        fixed_point_dist[angle_to_90 > max_angle_to_90] = float("inf")
        # min_angle_deg = min_angle_ * 180 / np.pi
        # min_angle_deg = torch.minimum(min_angle_deg, 180 - min_angle_deg)
        # fixed_point_dist[min_angle_deg < min_angle] = float("inf")

        projection_dists[fixed_point_candidates] = fixed_point_dist
        updated_constraints[fixed_point_candidates] = torch.stack([c1, c2, c3], dim=-1)

        # compute the distances to the new line constraints
        line_candidates = (n_joint_constraints == 2)
        c1 = joint_constraints[line_candidates][:, -1]
        choose_second_or_third = ((joint_constraints[line_candidates][:, -2] != c1) & (joint_constraints[line_candidates][:, -2] != -1)) 
        c2 = joint_constraints[line_candidates][:, -2] * choose_second_or_third + (joint_constraints[line_candidates][:, -3] * (~choose_second_or_third))
        assert (c1 != c2).all()
        assert (c1 != -1).all() and (c2 != -1).all()
        p1, p2 = intersection_line_between_planes_batched(planes[c1], planes[c2])
        _, dist1 = project_points_to_lines_torch(vertices[vertex_with_max_planes[line_candidates]], torch.cat([p1, p2], dim=1))
        _, dist2 = project_points_to_lines_torch(vertices[vertex_with_min_planes[line_candidates]], torch.cat([p1, p2], dim=1))
        line_dist = torch.maximum(dist1, dist2)
        # check that the angle between the planes is not too small
        angle_to_90 = self.vertices.get_angle_to_orthogonal((planes[c1], planes[c2]))
        # angle_deg = torch.minimum(angle_deg, 180 - angle_deg)
        line_dist[angle_to_90 > max_angle_to_90] = float("inf")
        
        projection_dists[line_candidates] = torch.maximum(dist1, dist2)
        updated_constraints[line_candidates] = torch.stack([c1, c2, -torch.ones_like(c1)], dim=-1)
    
        # compute the distances to the new plane constraints (they are forcibly already on the same plane)
        plane_candidates = (n_joint_constraints == 1)
        plane_constraints = joint_constraints[plane_candidates][:, -1]
        projection_dists[plane_candidates] = 0
        updated_constraints[plane_candidates] = torch.stack([plane_constraints, -torch.ones_like(plane_constraints), -torch.ones_like(plane_constraints)], dim=-1)


        # old2new = {i: i for i in range(len(vertices))}
        old2new = torch.arange(len(vertices), dtype=torch.long, device=vertices.device)
        merged_already = torch.zeros(len(vertices), dtype=torch.bool)
        
        mask = (projection_dists < max_dist)
        vertex_merge_candidates = vertex_merge_candidates[mask]
        projection_dists = projection_dists[mask]
        updated_constraints = updated_constraints[mask]
        # angle_to_90 = angle_to_90[mask]
        n_joint_constraints = n_joint_constraints[mask]
        vertex_with_max_planes = vertex_with_max_planes[mask]
        vertex_with_min_planes = vertex_with_min_planes[mask]
        constraints_1_numpy = constraints_1[mask].cpu().numpy()
        constraints_2_numpy = constraints_2[mask].cpu().numpy()
        updated_constraints_numpy = updated_constraints.cpu().numpy()
        
        print(f"Found {len(vertex_merge_candidates)} mergeable vertex pairs.")

        for ix, (i, j) in enumerate(vertex_merge_candidates.tolist()):
            if merged_already[i] or merged_already[j]: # TODO: if the constraints are the same and the distance is small we can merge them
                continue
            if False:
                constraints_1 = self.vertices.get_vertex_constraints(i).tolist()
                constraints_2 = self.vertices.get_vertex_constraints(j).tolist()
                joint_constraints = list(set(list(constraints_1) + list(constraints_2)))
                
                vertex_with_max_planes = i if n_planes_per_vertex[i] > n_planes_per_vertex[j] else j
                vertex_to_merge = j if vertex_with_max_planes == i else i
                # if (set(constraints_1) == set(constraints_2)) & (len(set(constraints_1)) < 3):
                #     # don't merge. Edge merging will take care of this if needed
                #     continue
                assert n_joint_constraints[ix] == len(joint_constraints)
                if set(constraints_1).issubset(set(constraints_2)) or set(constraints_2).issubset(set(constraints_1)):
                    if not (len(constraints_1) == 3 and len(constraints_2) == 3): # exception for fixed points
                        # don't merge if one constraint set is a subset of another. Edge merging can fix this
                        continue
                if len(joint_constraints) > 3:
                    # probably not satisfiable
                    continue
                if len(joint_constraints) == 3:
                    # compute the intersection of the 3 planes
                    try:
                        p1, p2 = intersection_line_between_planes(planes[joint_constraints[0]], planes[joint_constraints[1]])
                        intersection_point = line_plane_intersection(torch.cat([p1, p2]), planes[joint_constraints[2]])
                        # check the distances
                        dist1 = torch.norm(vertices[i] - intersection_point)
                        dist2 = torch.norm(vertices[j] - intersection_point)
                        assert np.isclose(max(dist1.item(), dist2.item()), projection_dists[ix].item(), atol=1e-5, rtol=2*1e-3)
                        if dist1 > max_dist or dist2 > max_dist:
                            continue
                    except ValueError:
                        continue
                elif len(joint_constraints) == 2:
                    # compute the intersection of the 2 planes, project to the line
                    try:
                        line_eq = intersection_line_between_planes(planes[joint_constraints[0]], planes[joint_constraints[1]])
                        projected_p1, dist1 = project_points_to_lines_torch(vertices[i].unsqueeze(0), torch.cat(line_eq).unsqueeze(0))
                        projected_p2, dist2 = project_points_to_lines_torch(vertices[j].unsqueeze(0), torch.cat(line_eq).unsqueeze(0))
                        assert np.isclose(max(dist1.item(), dist2.item()), projection_dists[ix].item(), atol=1e-5)
                        if dist1.item() > max_dist or dist2.item() > max_dist:
                            continue
                    except ValueError:
                        continue
                else:
                    raise ValueError("This should not happen")
                
                assert set(updated_constraints[ix].tolist()) - {-1} == set(joint_constraints)
            
            # if projection_dists[ix] < max_dist:
                # if the updated constraint set is subset of either of the original constraint sets, merge directly
            constraint_set_1 = set(_old_vertex_constraints_numpy[i]) - {-1}
            constraint_set_2 = set(_old_vertex_constraints_numpy[j]) - {-1}
            new_constraint_set = set(updated_constraints_numpy[ix]) - {-1}
            if not (new_constraint_set.issubset(constraint_set_1) or new_constraint_set.issubset(constraint_set_2)):
                # double-check whether we can link those planes
                is_fixed_point = updated_constraints[ix, 2] != -1
                if is_fixed_point:
                    _angle_to_90 = self.vertices.get_angle_to_orthogonal((planes[updated_constraints[ix, 0]].unsqueeze(0), planes[updated_constraints[ix, 1]].unsqueeze(0), planes[updated_constraints[ix, 2]].unsqueeze(0)))
                    # assert _angle_to_90 == angle_to_90[ix]
                    if _angle_to_90 > max_angle_to_90:
                        continue
                else: # line point
                    _angle_to_90 = self.vertices.get_angle_to_orthogonal((planes[updated_constraints[ix, 0]].unsqueeze(0), planes[updated_constraints[ix, 1]].unsqueeze(0)))
                    # assert _angle_to_90 == angle_to_90[ix]
                    if _angle_to_90 > max_angle_to_90:
                        continue
            v1, v2 = vertex_with_max_planes[ix], vertex_with_min_planes[ix]
            # assert (v1 == i and v2 == j) or (v1 == j and v2 == i)
            old2new[v2] = v1
            merged_already[v1] = True
            merged_already[v2] = True
            if (updated_constraints[ix] != vertex_constraints[v1]).any():
                self.vertices.update_vertex_constraints(v1, updated_constraints[ix])
            if (updated_constraints[ix] != vertex_constraints[v2]).any():
                self.vertices.update_vertex_constraints(v2, updated_constraints[ix])
                # assert (self.vertices.vertex_plane_assignments[v1].sort().values == updated_constraints[ix].sort().values).all()
                # assert (self.vertices.vertex_plane_assignments[v2].sort().values == updated_constraints[ix].sort().values).all()
        
        new_verts = self.vertices.get()[0]
        if not (new_verts - vertices).norm(dim=-1).max() < max_dist + 1e-5:
            print("Warning! Merging vertices did not work as expected")
            raise ValueError("Merging vertices did not work as expected")
            for vertex in torch.where((new_verts - vertices).norm(dim=-1) >= max_dist)[0]:
                merged_already[vertex] = False
                # reset to old constraints
                old2new[vertex] = vertex
                self.vertices.update_vertex_constraints(vertex, _old_vertex_constraints_numpy[vertex])
        
        if not (new_verts - vertices).norm(dim=-1).max() < max_dist + 1e-5:
            raise ValueError("Merging vertices did not work as expected")
            print("Warning! Merging vertices did not work as expected")
            for vertex in torch.where((new_verts - vertices).norm(dim=-1) >= max_dist)[0]:
                if self.vertices.vertex_plane_assignments[vertex].min() != -1 and self.vertices.get_angle_to_orthogonal((planes[self.vertices.vertex_plane_assignments[vertex]].unsqueeze(1))) < max_angle_to_90:
                    self.vertices.vertex_plane_assignments[vertex, :] = -1 # let retriangulation fix this
                # reset to old constraints
                old2new[vertex] = vertex

        new_verts = self.vertices.get()[0]
        planes = self.get_planes()
        # print("New vertices", new_verts[special_ix])
        # print("New planes", planes[self.vertices.vertex_plane_assignments[special_ix]])
        # print("New constraints", self.vertices.vertex_plane_assignments[special_ix])

            
        # if not ((new_verts - vertices).norm(dim=-1).max() < max_dist + 1e-5):
        #     raise ValueError("Merging vertices did not work as expected")
            # print(f"WARNING!! Merging vertices did not work as expected. Max distance: {(new_verts - vertices).norm(dim=-1).max().item()}")
            # print(f"Merging vertices {i} and {j}. Constraints of i: {constraints_1}, constraints of j: {constraints_2}, joint constraints: {joint_constraints}")


        
        if plot:
            mesh = self.export_triangle_mesh()
            import open3d as o3d
            # paint the merge candidates red and the others blue
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.get_vertices().detach().cpu().numpy())
            colors = np.zeros((len(point_cloud.points), 3))
            for i, j in vertex_merge_candidates:
                colors[i] = [1, 0, 0]
                colors[j] = [0, 1, 0]
            colors[merged_already] = [0, 0, 1]
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([mesh, point_cloud])

        # reassign the triangles to the merged vertices
        self.triangles = old2new[self.triangles]

        
        # max_dot_product = 0
        # max_dot_product_index = -1
        # planes_1 = None
        # planes_2 = None
        # constraints_ = None
        # for i in range(len(self.vertices.vertex_plane_assignments)):
        #     constraints = self.vertices.vertex_plane_assignments[i]
        #     if constraints[-1] == -1 and constraints[1] != -1:
        #         dot_product = (planes[constraints[0]][:3] @ planes[constraints[1]][:3]).item()
        #         if dot_product > max_dot_product:
        #             max_dot_product_index = i
        #             max_dot_product = dot_product
        #             planes_1 = planes[constraints[0]][:3]
        #             planes_2 = planes[constraints[1]][:3]
        #             constraints_ = constraints


        # print(f"Max dot product: {max_dot_product}, index: {max_dot_product_index}, planes: {planes_1}, {planes_2}, constraints: {constraints_}")
        
        # # break up vertices that have constraints that are too close to 90 degrees
        # constraints = self.vertices.vertex_plane_assignments
        # fixed_point_mask = constraints[:, 2] != -1
        # non_valid_mask = self.vertices.get_angle_to_orthogonal((planes[constraints[fixed_point_mask, 0]], planes[constraints[fixed_point_mask, 1]], planes[constraints[fixed_point_mask, 2]])) > max_angle_to_90
        # # constraints[fixed_point_mask, -1][non_valid_mask] = -1

        # line_mask = (constraints[:, 2] == -1) & (constraints[:, 1] != -1)
        # non_valid_mask = self.vertices.get_angle_to_orthogonal((planes[constraints[line_mask, 0]], planes[constraints[line_mask, 1]])) > max_angle_to_90




        # remove triangles with less than 3 unique vertices
        # new_triangles = []
        # new_triangle_polygons = []
        # for triangle, polygon in zip(self.triangles, self.triangle_polygons):
        #     new_triangle = []
        #     for vertex in triangle:
        #         new_triangle.append(old2new[vertex.item()])
        #     if len(set(new_triangle)) == 3:
        #         new_triangles.append(new_triangle)
        #         new_triangle_polygons.append(polygon.item())

        # clean up: remove vertices that are not in any triangle
        self.clean()


        n_changed = len(vertices) - len(self.vertices.vertices)
        # n_changed = (old2new != torch.arange(len(vertices), device=vertices.device)).sum().item()
        if n_changed > 0 and recursive_depth > 0:
            print(f"Merged {n_changed} vertices. N remaining vertices: {len(self.get_vertices())}")
            self.simplify_vertices(plot=plot, max_dist=max_dist, max_angle_to_90=max_angle_to_90, recursive_depth=recursive_depth-1)


    @torch.no_grad()
    def attach_vertices_to_close_lines(self, max_dist=0.01, max_angle_to_90=70):
        # attach points that are on lines to the line
        print("Attaching vertices to lines")
        old_constraints = self.vertices.vertex_plane_assignments.clone()
        vertices = self.get_vertices().detach()
        contour_edges = self.get_contour_edges()
        contour_edge_has_attachment = torch.zeros(len(contour_edges), dtype=torch.bool, device=vertices.device)
        # contour_edges = torch.cat([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]], dim=0)
        all_edges = torch.cat([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]], dim=0)
        all_edges = torch.sort(all_edges, dim=1)[0]
        planes = self.get_planes()
        k = 20
        vertex_edge_indices, vertex_edge_distances = get_closest_edges_to_point_sq_dist(vertices, vertices[contour_edges], k=k)
        vertex_edge_distances = vertex_edge_distances.sqrt()
        # vertex_edge_polygons = self.vertices.vertex_plane_assignments[contour_edges[vertex_edge_indices]]
        # vertex_polygons = self.vertices.vertex_plane_assignments.unsqueeze(1).repeat(1, k, 1)
        # candidates = torch.arange(len(vertices), device=vertices.device).unsqueeze(1).repeat(1, k)
        # candidates = torch.stack([candidates, vertex_edge_indices], dim=-1)
        # candidates = candidates[vertex_edge_distances < max_dist]
        # candidates = candidates[candidates[:, 0] != candidates[:, 1]]
        # candidates = torch.unique(candidates, dim=0)
        # candidate_constraints = self.vertices.vertex_plane_assignments[candidates[0]]
        
        vertex_ixes = torch.arange(len(vertices), device=vertices.device).unsqueeze(1).repeat(1, k).unsqueeze(-1)
        edge_vertices = contour_edges[vertex_edge_indices]
        vertex_triplets = torch.concatenate([vertex_ixes, edge_vertices], dim=-1)
        vertex_triplets = vertex_triplets[vertex_edge_distances < max_dist]
        vertex_triplets = vertex_triplets[(vertex_triplets[:, 0] != vertex_triplets[:, 1]) & (vertex_triplets[:, 0] != vertex_triplets[:, 2]) & (vertex_triplets[:, 1] != vertex_triplets[:, 2])]
        vertex_triplets = torch.unique(vertex_triplets, dim=0)


        verts_to_be_attached = []
        vert_ixes_to_be_attached = []
        # for i, (vertex, edges, distances) in enumerate(zip(vertices, vertex_edge_indices, vertex_edge_distances)):
        #     constraints_1 = self.vertices.vertex_plane_assignments[i].tolist()
        #     candidates = (distances < max_dist)
        #     for edge in zip(edges[candidates]):
        #         i1, i2 = contour_edges[edge]
        #         if i1 == i or i2 == i:
        #             continue
        contour_adjacency = self.contour_adjacency_list

        for i, i1, i2 in vertex_triplets.tolist():
                constraints_1 = self.vertices.vertex_plane_assignments[i].tolist()
                constraints_2 = self.vertices.vertex_plane_assignments[i1].tolist()
                constraints_3 = self.vertices.vertex_plane_assignments[i2].tolist()
                edge_constraints_2_3 = set(constraints_2).intersection(set(constraints_3))
                new_constraints = list(set(constraints_1).union(edge_constraints_2_3) - {-1})
                if len(new_constraints) == 3:
                    max_angle = self.vertices.get_angle_to_orthogonal((planes[new_constraints[0]].unsqueeze(0), planes[new_constraints[1]].unsqueeze(0), planes[new_constraints[2]].unsqueeze(0)))
                    # p1, p2 = intersection_line_between_planes(planes[new_constraints[0]], planes[new_constraints[1]])
                    # fixed_point = line_plane_intersection(torch.cat([p1, p2]), planes[new_constraints[2]])
                    # if (fixed_point - vertex).norm() > max_dist:
                    #     continue
                elif len(new_constraints) == 2:
                    max_angle = self.vertices.get_angle_to_orthogonal((planes[new_constraints[0]].unsqueeze(0), planes[new_constraints[1]].unsqueeze(0)))
                    # p1, p2 = intersection_line_between_planes(planes[new_constraints[0]], planes[new_constraints[1]])
                    # projected_p1, dist1 = project_points_to_lines_torch(vertices[i].unsqueeze(0), torch.cat([p1, p2]).unsqueeze(0))
                    # if dist1 > max_dist:
                    #     continue
                    # new_constraints.append(-1)
                else:
                    # print(f"pair {i1}-{i2} has no common constraint: {new_constraints}, {len(new_constraints)}")
                    continue
                if max_angle < max_angle_to_90:
                    new_constraints = torch.tensor(new_constraints, dtype=torch.long, device=vertices.device)
                    self.vertices.update_vertex_constraints(i, new_constraints)
                    verts_to_be_attached.append(vertices[i].cpu().numpy())
                    vert_ixes_to_be_attached.append(i)
                    # print(f"Attaching vertex {i} to edge {i1}-{i2} with constraints {new_constraints}")
                    # contour_edge_has_attachment[edge] = True

                    # also attach the other edge to the line between the planes
                    if len(edge_constraints_2_3) == 1:
                        neighbors = contour_adjacency[i]
                        if len(neighbors) == 0:
                            continue
                        line = vertices[[i1, i2]].view(-1).unsqueeze(0).repeat(len(neighbors), 1)
                        projected_neighbors, dists = project_points_to_lines_torch(vertices[neighbors], line)
                        if dists.min() < max_dist:
                            neighbor_on_line = neighbors[dists.argmin()]
                            edge_constraints_1_n = set(self.vertices.vertex_plane_assignments[neighbor_on_line].tolist()).intersection(set(constraints_1))
                            line_constraints = list(set(edge_constraints_1_n).union(edge_constraints_2_3) - {-1})
                            
                            new_constraints_2 = list(set(constraints_2).union(line_constraints) - {-1})
                            if len(new_constraints_2) == 3:
                                max_angle_2 = self.vertices.get_angle_to_orthogonal((planes[new_constraints_2[0]].unsqueeze(0), planes[new_constraints_2[1]].unsqueeze(0), planes[new_constraints_2[2]].unsqueeze(0)))
                            elif len(new_constraints_2) == 2:
                                max_angle_2 = self.vertices.get_angle_to_orthogonal((planes[new_constraints_2[0]].unsqueeze(0), planes[new_constraints_2[1]].unsqueeze(0)))
                            else:
                                max_angle_2 = float("inf")
                            if max_angle_2 < max_angle_to_90:
                                new_constraints_2 = torch.tensor(new_constraints_2, dtype=torch.long, device=vertices.device)
                                self.vertices.update_vertex_constraints(i1, new_constraints_2)
                            
                            new_constraints_3 = list(set(constraints_3).union(line_constraints) - {-1})
                            if len(new_constraints_3) == 3:
                                max_angle_3 = self.vertices.get_angle_to_orthogonal((planes[new_constraints_3[0]].unsqueeze(0), planes[new_constraints_3[1]].unsqueeze(0), planes[new_constraints_3[2]].unsqueeze(0)))
                            elif len(new_constraints_3) == 2:
                                max_angle_3 = self.vertices.get_angle_to_orthogonal((planes[new_constraints_3[0]].unsqueeze(0), planes[new_constraints_3[1]].unsqueeze(0)))
                            else:
                                max_angle_3 = float("inf")
                            if max_angle_3 < max_angle_to_90:
                                new_constraints_3 = torch.tensor(new_constraints_3, dtype=torch.long, device=vertices.device)
                                self.vertices.update_vertex_constraints(i2, new_constraints_3)

                else:
                    # print(f"Angle too large: {max_angle.item()}")
                    pass
        
        # create pcd
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # # pcd.points = o3d.utility.Vector3dVector(verts_to_be_attached
        # pcd.points = o3d.utility.Vector3dVector(vertices[torch.unique(vertex_triplets.reshape(-1))].cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((len(vertices), 3)))
        # o3d.io.write_point_cloud("/mnt/usb_ssd/bieriv/layout-estimation-outputs/01-20-dslr-mesh-fitting/scannet++/5ee7c22ba0/all-classes/fitted_mesh_vertices.ply", pcd)

        print(f"Attached {len(verts_to_be_attached)} vertices to lines.")

        # check that they haven't moved too much
        new_vertices = self.get_vertices().detach()
        moved_too_much_mask = (new_vertices - vertices).norm(dim=-1) > 2*max_dist
        self.vertices.vertex_plane_assignments[moved_too_much_mask] = old_constraints[moved_too_much_mask]

        print(f"Number of vertices that moved too much: {moved_too_much_mask.sum().item()}. Gotta figure this out!")

        # attached_vertices = np.array(verts_to_be_attached)
        # vertices = self.get_vertices().detach().cpu().numpy()
        # new_attached_vertices = vertices[np.array(vert_ixes_to_be_attached)]
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # all_vertices = np.concatenate([new_attached_vertices, attached_vertices])
        # all_vertex_colors = np.concatenate([np.array([1, 0,0]) * np.ones((len(vertices), 3)), np.array([0, 1, 0]) * np.ones((len(attached_vertices), 3))])
        # # pcd.points = o3d.utility.Vector3dVector(attached_vertices)
        # pcd.points = o3d.utility.Vector3dVector(all_vertices)
        # pcd.colors = o3d.utility.Vector3dVector(all_vertex_colors)
        # o3d.io.write_point_cloud("/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/meshes/scannet++_debug/opengs/run23_openseg/all-classes/attachment.ply", pcd)
        # mesh = self.export_triangle_mesh()
        # o3d.io.write_triangle_mesh("/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/meshes/scannet++_debug/opengs/run23_openseg/all-classes/attachment_mesh.ply", mesh)

        # fixed_points = self.vertices.vertex_plane_assignments[:, 2] != -1
        # new_vertices = self.get_vertices()
        # max_fixed_point_dist = 0
        # for i in range(3):
        #     planes_i = self.vertices.planes[self.vertices.vertex_plane_assignments[:, i]]
        #     projected_pts, dists = project_points_3d_to_3d_plane(new_vertices[fixed_points], planes_i[fixed_points])
        #     max_fixed_point_dist = max(max_fixed_point_dist, dists.max().item())
        
        # print(f"Max fixed point distance: {max_fixed_point_dist}")

        # line_points = (self.vertices.vertex_plane_assignments[:, 2] == -1) & (self.vertices.vertex_plane_assignments[:, 1] != -1)
        # max_line_point_dist = 0
        # for i in range(2):
        #     planes_i = self.vertices.planes[self.vertices.vertex_plane_assignments[:, i]]
        #     projected_pts, dists = project_points_3d_to_3d_plane(new_vertices[line_points], planes_i[line_points])
        #     max_line_point_dist = max(max_line_point_dist, dists.max().item())
        # print(f"Max line point distance: {max_line_point_dist}")

        # # fixed_points_numpy = new_vertices[fixed_points].detach().cpu().numpy()
        # # pcd = o3d.geometry.PointCloud()
        # # pcd.points = o3d.utility.Vector3dVector(fixed_points_numpy)
        # # o3d.io.write_point_cloud("/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/meshes/scannet++_debug/opengs/run23_openseg/all-classes/fixed_points.ply", pcd)



    def export_color_coded_vertex_cloud(self):
        # # export a point cloud where the mesh vertices are colored according to the number of constraints
        import open3d as o3d
        vertices = self.get_vertices().detach().cpu().numpy()
        vertex_constraints = self.vertices.vertex_plane_assignments
        n_constraints = 3 - (vertex_constraints != -1).sum(dim=-1)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        vertex_colors = colors[n_constraints.cpu().numpy()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
        return pcd



    def get_non_watertight_edges(self, return_if_attached_to_edge=True):
        """Returns all edges that are part of only one triangle"""
        triangles = self.triangles
        edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        unique_edges, edge_count = torch.unique(edges, return_counts=True, dim=0)
        non_watertight_edges = unique_edges[edge_count == 1]
        if not return_if_attached_to_edge:
            edge_constraints = self.vertices.vertex_plane_assignments[non_watertight_edges]
            two_constraints_match = ((edge_constraints[:, 0, None, :] == edge_constraints[:, 1, :, None]) & (edge_constraints[:, 0, None, :] != -1) & (edge_constraints[:, 1, :, None] != -1)).sum(dim=-1).sum(dim=-1) >= 2
            non_watertight_edges = non_watertight_edges[~two_constraints_match]
        return non_watertight_edges
    
    def get_watertight_edges(self):
        """Returns all edges that are part of two triangles"""
        triangles = self.triangles

        edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        unique_edges, edge_count = torch.unique(edges, return_counts=True, dim=0)
        watertight_edges = unique_edges[edge_count > 1]
        return watertight_edges
    
    # def get_contour_edges(self):
    #     edges = torch.cat([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]], dim=0)
    #     edges = torch.sort(edges, dim=1)[0]
    #     unique_edges = torch.unique(edges, dim=0)
    #     return unique_edges

    def get_contour_edges_with_cache(self, return_non_watertight=True):
        contour_edges_cached = getattr(self, "contour_edges", None)
        if contour_edges_cached is None:
            self.frozen_vertex_plane_assignments_contour_edges = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_contour_edges = self.triangles.detach().clone()
            self.contour_edges = self.get_contour_edges()
            self.contour_edges_without_non_watertight = self.get_contour_edges(return_non_watertight=False)
        elif (len(self.frozen_triangles_contour_edges) != len(self.triangles)) or (len(self.frozen_vertex_plane_assignments_contour_edges) != len(self.vertices.vertex_plane_assignments)) or not (self.frozen_triangles_contour_edges == self.triangles).all() or not (self.frozen_vertex_plane_assignments_contour_edges == self.vertices.vertex_plane_assignments).all():
            self.frozen_vertex_plane_assignments_contour_edges = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_contour_edges = self.triangles.detach().clone()
            self.contour_edges = self.get_contour_edges()
            self.contour_edges_without_non_watertight = self.get_contour_edges(return_non_watertight=False)
        if return_non_watertight:
            return self.contour_edges
        else:
            return self.contour_edges_without_non_watertight
        

    def get_inner_edge_mask_with_cache(self):
        if not hasattr(self, "inner_edge_mask"):
            self.frozen_vertex_plane_assignments_inner_edges = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_inner_edges = self.triangles.detach().clone()
            self.inner_edge_mask = self.get_inner_edge_mask()
        elif (len(self.frozen_triangles_inner_edges) != len(self.triangles)) or (len(self.frozen_vertex_plane_assignments_inner_edges) != len(self.vertices.vertex_plane_assignments)) or not (self.frozen_triangles_inner_edges == self.triangles).all() or not (self.frozen_vertex_plane_assignments_inner_edges == self.vertices.vertex_plane_assignments).all():
            self.frozen_vertex_plane_assignments_inner_edges = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_inner_edges = self.triangles.detach().clone()
            self.inner_edge_mask = self.get_inner_edge_mask()
        return self.inner_edge_mask
    

    def get_inner_edge_mask(self):
        """Returns a mask that is one for all edges that are not on the contour
        Returns:
            is_inner_edge: torch.Tensor of shape (n_triangles, 3) indicating whether the edges (t[0], t[1]), (t[1], t[2]), (t[2], t[0]) lie on the contour for each triangle t
        """
        contour_edges = self.get_contour_edges_with_cache()
        contour_edge_set = set(map(lambda x: tuple(sorted(x)), contour_edges.cpu().numpy()))

        is_inner_edge = torch.ones((len(self.triangles), 3), dtype=torch.bool, device=self.device)
        for i, triangle in enumerate(self.triangles.cpu().numpy()):
            for j, edge in enumerate([triangle[[0, 1]], triangle[[1, 2]], triangle[[2, 0]]]):
                if tuple(sorted(edge)) in contour_edge_set:
                    is_inner_edge[i, j] = False
        return is_inner_edge


    


    @torch.no_grad()
    def get_contour_edges(self, return_non_watertight=True):
        """Returns all edges that are either
            1) part of two triangles belonging to different polygons or
            2) part of only one triangle (non-watertight)
        """
        # only mentioned once
        edges = torch.cat([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        unique_edges, edge_count = torch.unique(edges, return_counts=True, dim=0)
        non_watertight_edges = unique_edges[edge_count == 1]

        # part of two triangles belonging to different polygons
        watertight_edges = unique_edges[(edge_count > 1) & (self.vertices.vertex_plane_assignments[unique_edges[:, 0], 1] != -1) & (self.vertices.vertex_plane_assignments[unique_edges[:, 1], 1] != -1)]
        unique_edge_constraints = self.vertices.vertex_plane_assignments[watertight_edges] # shape (n_edges, 2, 3)
        shared_edges = []
        for edge, edge_constraints in zip(watertight_edges.cpu().numpy(), unique_edge_constraints.cpu().numpy()):
            constraints1 = set(edge_constraints[0].tolist())
            constraints2 = set(edge_constraints[1].tolist())
            # if they intersect with at least two constraints then they are on a contour
            if len(constraints1.intersection(constraints2) - {-1}) >= 2:
                shared_edges.append(edge)

        shared_edges = torch.from_numpy(np.array(shared_edges).astype(int)).to(edges.device)

        if return_non_watertight:
            return torch.cat([non_watertight_edges, shared_edges], dim=0)
        else:
            return shared_edges
    

    def get_inner_edges(self):
        """Returns all edges that are not contour edges"""
        edges = torch.cat([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        unique_edges, edge_count = torch.unique(edges, return_counts=True, dim=0)
        planes_1 = self.vertices.vertex_plane_assignments[unique_edges[:, 0]]
        planes_2 = self.vertices.vertex_plane_assignments[unique_edges[:, 1]]
        equals = ((planes_1[:, None] == planes_2[:, :, None]) * (planes_1[:, :, None] != -1)).sum(dim=-1).sum(dim=-1) >= 2
        belongs_to_only_one_plane = (planes_1[:, 1:] == -1).all(dim=-1) & (planes_2[:, 1:] == -1).all(dim=-1)
        inner_edges = unique_edges[(edge_count > 1) & (equals | belongs_to_only_one_plane)]
        # inner_edges = unique_edges[(edge_count > 1) & (self.vertices.vertex_plane_assignments[unique_edges[:, 0], 1] == self.vertices.vertex_plane_assignments[unique_edges[:, 1], 1])]
        return inner_edges
    

    def find_edge_pairs_with_cache(self, max_n_neighbors=5):
        if not hasattr(self, "edge_pairs"):
            self.frozen_vertex_plane_assignments_edge_pairs = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_edge_pairs = self.triangles.detach().clone()
            self.edge_pairs = self.find_edge_pairs(max_n_neighbors=max_n_neighbors)
        elif (len(self.frozen_triangles_edge_pairs) != len(self.triangles)) or (len(self.frozen_vertex_plane_assignments_edge_pairs) != len(self.vertices.vertex_plane_assignments)) or not (self.frozen_triangles_edge_pairs == self.triangles).all() or not (self.frozen_vertex_plane_assignments_edge_pairs == self.vertices.vertex_plane_assignments).all():
            self.frozen_vertex_plane_assignments_edge_pairs = self.vertices.vertex_plane_assignments.detach().clone()
            self.frozen_triangles_edge_pairs = self.triangles.detach().clone()
            self.edge_pairs = self.find_edge_pairs(max_n_neighbors=max_n_neighbors)
        return self.edge_pairs


    
    def find_edge_pairs(self, max_n_neighbors=5):
        """Returns all pairs of edges that share a vertex"""
        triplets = []
        for v in range(len(self.contour_adjacency_list)):
            neighbors = self.contour_adjacency_list[v]
            if len(neighbors) < 2:
                continue
            if len(neighbors) > max_n_neighbors:
                neighbors = neighbors[torch.randperm(len(neighbors))[:max_n_neighbors]]
            comb = torch.combinations(self.adjacency_list[v], 2)
            new_triplets = torch.cat([comb, torch.ones(len(comb), 1, dtype=torch.long) * v], dim=-1)
            new_triplets = torch.roll(new_triplets, -1, dims=-1)
            triplets.append(new_triplets)
        triplets = torch.cat(triplets, dim=0)
        triplets = torch.unique(triplets, dim=0)
        return triplets.long()
    

    @torch.no_grad()
    def simplify_edges(self, plot=False, max_deg=15, dist_thresh=0.005, recursive_depth=1, simplify_rectangles=False):
        self.clear_cache()        
        contour_edges = self.get_contour_edges()
        vertices = self.get_vertices()
        triangles_sorted = self.triangles.clone().sort(dim=-1).values

        contour_edges_by_vertex = {}
        for edge in contour_edges.cpu().numpy():
            if edge[0] not in contour_edges_by_vertex:
                contour_edges_by_vertex[edge[0]] = []
            if edge[1] not in contour_edges_by_vertex:
                contour_edges_by_vertex[edge[1]] = []
            contour_edges_by_vertex[edge[0]].append(edge[1])
            contour_edges_by_vertex[edge[1]].append(edge[0])

        self.clear_cache()
        # mesh = self.export_triangle_mesh()
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(vertices[contour_edges.view(-1)].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([mesh, pcd])

        deleted = torch.zeros(len(vertices), dtype=torch.bool)
        must_not_be_deleted = torch.zeros(len(vertices), dtype=torch.bool)

        if not simplify_rectangles:
            unique_triangles, num_triangles_per_polygon = torch.unique(self.triangle_polygons, return_counts=True)
            vertices_with_four_edges = torch.unique(self.triangles[unique_triangles[num_triangles_per_polygon == 2]].view(-1))
            must_not_be_deleted[vertices_with_four_edges] = True
        # old2new = torch.arange(len(vertices), dtype=torch.long, device=self.get_vertices().device)

        new_triangles = []
        new_triangle_polygons = []

        contour_edges_by_vertex_ = torch.tensor([[v[0], i2, v[1]] for i2, v in contour_edges_by_vertex.items() if len(v) == 2]).to(self.device)
        p1_vec, p2_vec, p3_vec = vertices[contour_edges_by_vertex_].unbind(dim=1)
        v1 = p1_vec - p2_vec
        v2 = p3_vec - p2_vec
        angles = torch.acos(torch.clamp(torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8), -1 + 1e-8, 1 - 1e-8)) * 180 / np.pi
        angles = 180 - angles
        edge_lengths = torch.norm(p1_vec - p2_vec, dim=-1)
        constraints_2 = self.vertices.vertex_plane_assignments[contour_edges_by_vertex_[:, 1]]
        n_constraints_2 = constraints_2.shape[1] - (constraints_2 == -1).sum(dim=-1)

        projected_points, dists = project_points_to_lines_torch(p2_vec, torch.cat([p1_vec, p3_vec], dim=-1))
        too_close = torch.isclose(p1_vec, p3_vec, atol=1e-3).all(dim=-1)
        projected_points[too_close] = p1_vec[too_close]

        mask = (n_constraints_2 == 1) & ((angles < max_deg) | (edge_lengths < dist_thresh))

        contour_edges_by_vertex_ = contour_edges_by_vertex_[mask]
        projected_points = projected_points[mask].detach()
        
        for ix, (i1, i2, i3) in enumerate(contour_edges_by_vertex_):
            if must_not_be_deleted[i2]:
                continue
            self.vertices.vertices.data[i2] = projected_points[ix]
            deleted[i2] = True
            must_not_be_deleted[i1] = True
            must_not_be_deleted[i3] = True

        

        
        


        # _contour_edges_by_vertex = _contour_edges_by_vertex[mask]
        





        # for i2 in tqdm(contour_edges_by_vertex, desc="Simplifying edges"):
        #     if len(contour_edges_by_vertex[i2]) == 2:
        #         i1, i3 = contour_edges_by_vertex[i2]
        #         # if any of its neighbors have been deleted, skip
        #         if must_not_be_deleted[i2]:
        #             continue
        #         p1, p2, p3 = vertices[[i1, i2, i3]]
        #         v1 = p1 - p2
        #         v2 = p3 - p2
        #         angles = torch.acos(torch.clamp(torch.sum(v1 * v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8), -1 + 1e-8, 1 - 1e-8)) * 180 / np.pi
        #         angles = 180 - angles
        #         edge_lengths = torch.norm(p1 - p2)
        #         if (angles < max_deg) | (edge_lengths < dist_thresh):
        #             constraints_1 = self.vertices.get_vertex_constraints(i1).tolist()
        #             constraints_2 = self.vertices.get_vertex_constraints(i2).tolist()
        #             constraints_3 = self.vertices.get_vertex_constraints(i3).tolist()
        #             # if the vertex is a corner, skip
        #             if len(constraints_2) == 3:
        #                 continue
        #             # if the vertex is only on one plane, merge
        #             elif len(constraints_2) == 1:
        #                 deleted[i2] = True 
        #                 # old2new[i2] = i3
        #                 must_not_be_deleted[i1] = True
        #                 must_not_be_deleted[i3] = True
        #                 # if torch.tensor(sorted([i1, i2, i3])).to(triangles_sorted.device) not in triangles_sorted:
        #                 #     new_triangles.append([i1, i2, i3])
        #                 #     new_triangle_polygons.append(constraints_1[0])
        #                 # self.vertices.vertices[i2]


        #                 # project the vertex to the line: once it is "inside" the polygon it will be ignored during retriangulation
        #                 p1, p2 = vertices[[i1, i3]]
        #                 if not torch.isclose(p1, p2, atol=1e-5).all():
        #                     projected_p, _ = project_points_to_lines_torch(vertices[i2].unsqueeze(0), torch.cat([p1, p2]).unsqueeze(0))
        #                     # if (projected_p - vertices[i2]).norm() < dist_thresh * 2:
        #                     self.vertices.vertices.data[i2] = projected_p
        #                     assert not torch.isnan(self.vertices.vertices[i3]).any()
                        # i1_i2_triangle = self.triangles[(self.triangles == i2).any(dim=-1) & (self.triangles == i1).any(dim=-1)]
                        
                        # self.vertices.verte
                    # if the vertex and both its neighbors are on the same line, merge
                    # elif len(constraints_2) == 2 and (constraints_1 == constraints_2) and (constraints_1 == constraints_3):
                    #     deleted[i2] = True
                    #     # old2new[i2] = i1
                    #     must_not_be_deleted[i1] = True
                    #     must_not_be_deleted[i3] = True
                        # new_triangles.append([i1, i2, i3])
                        # new_triangle_polygons.append(constraints_1[0])
                        # remove the line constraint from i2: it now belongs to i1
                        # there is nothing to do: this will be simplified by retriangulation
                        # self.vertices.update_vertex_constraints(i1, torch.tensor([constraints_1[0], -1, -1], dtype=torch.long, device=vertices.device))


                    
        # self.clear_cache()
        # mesh = self.export_triangle_mesh()
        # # visualize the deleted vertices
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(vertices[deleted].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([mesh, pcd])

        # self.triangles = old2new[self.triangles]
        new_triangles = torch.tensor(new_triangles, dtype=torch.long, device=vertices.device)
        self.triangles = torch.cat([self.triangles, new_triangles], dim=0)
        new_triangle_polygons = torch.tensor(new_triangle_polygons, dtype=torch.long, device=vertices.device)
        self.triangle_polygons = torch.cat([self.triangle_polygons, new_triangle_polygons], dim=0)
        
        # self.clear_cache()
        self.clean()

        if deleted.sum() > 0:
            print(f"Deleted {deleted.sum()} vertices. N remaining vertices: {len(self.get_vertices())}")
            if recursive_depth > 0:
                self.simplify_edges(plot=plot, max_deg=max_deg, dist_thresh=dist_thresh, recursive_depth=recursive_depth-1)
                
        

        
            
        # edge_merge_candidates = self.find_mergeable_edges()

        # if plot:
        #     mesh = self.export_triangle_mesh()
        #     import open3d as o3d
        #     # paint the merge candidates red and the others blue
        #     point_cloud = o3d.geometry.PointCloud()
        #     point_cloud.points = o3d.utility.Vector3dVector(self.get_vertices().detach().cpu().numpy())
        #     colors = np.zeros((len(point_cloud.points), 3))
        #     colors[edge_merge_candidates] = [1, 0, 0]
        #     colors[~edge_merge_candidates] = [0, 0, 1]
        #     point_cloud.colors = o3d.utility.Vector3dVector(colors)
        #     o3d.visualization.draw_geometries([mesh, point_cloud])

        # old_polygons = self.polygons_closed
        # new_points = []
        # vertices = self.get_vertices()
        # n_polygons_per_vertex = self.vertices.get_num_assigned_planes_per_vertex()
        # deleted = torch.zeros(len(vertices), dtype=torch.bool)
        # replacements = torch.arange(len(vertices), dtype=torch.long)

        # shared_edges_to_check = {} # this maps (vertex on plane) -> mergeable neighbor on line

        # for i, polygon in enumerate(old_polygons):
        #     new_polygon = []
        #     for j, vertex in enumerate(polygon):
        #         if not edge_merge_candidates[vertex]:
        #             new_polygon.append(vertex)
        #             new_points.append(vertices[vertex].cpu().detach().numpy())
        #         else:
        #             # if any neighbor has been deleted skip
        #             if deleted[polygon[(j - 1) % len(polygon)]] or deleted[polygon[(j + 1) % len(polygon)]]:
        #                 # new_polygon.append(vertex)
        #                 # new_points.append(vertices[vertex].detach().numpy())
        #                 continue
        #             else:
        #                 # if the vertex is a corner, skip
        #                 if n_polygons_per_vertex[vertex] == 3:
        #                     continue
        #                 # if the vertex is on a line:
        #                 elif n_polygons_per_vertex[vertex] == 2:
        #                     # 1) if both its neighbors are on the line, delete
        #                     if n_polygons_per_vertex[polygon[(j - 1) % len(polygon)]] >= 2 and n_polygons_per_vertex[polygon[(j + 1) % len(polygon)]] >= 2:
        #                         deleted[vertex] = True
        #                     # 2) if it has at least one neighbor that is not on the line, check if we can add the neighbor to the line
        #                     # elif n_polygons_per_vertex[polygon[(j + 1) % len(polygon)]] == 1:
        #                     #     # deleted[polygon[(j + 1) % len(polygon)]] = True
        #                     #     # shared_edges_to_check[polygon[(j + 1) % len(polygon)]] = vertex
        #                     #     continue
        #                     # elif n_polygons_per_vertex[polygon[(j - 1) % len(polygon)]] == 1:
        #                     #     # deleted[polygon[(j - 1) % len(polygon)]] = True
        #                     #     # shared_edges_to_check[polygon[(j - 1) % len(polygon)]] = vertex
        #                     #     continue
        #                     # else: # its neighbors are
        #                     #     raise ValueError("This should not happen")
        #                 # if the vertex is on a plane: delete
        #                 elif n_polygons_per_vertex[vertex] == 1:
        #                     deleted[vertex] = True
            # new_old_polygons.append(new_polygon)

        # create a dictionary (vertex on line) -> list of candidates to merge
        # merge_candidates = {k: [] for k in shared_edges_to_check.values()}
        # for i, j in shared_edges_to_check.items():
        #     merge_candidates[j].append(i)

        # for vertex_on_line, candidates in merge_candidates.items():
        #     vertex_constraints = self.vertices.get_vertex_constraints(vertex_on_line)
        #     plane_eq1, plane_eq2 = self.planes[vertex_constraints[0]], self.planes[vertex_constraints[1]]
        #     line_eq = intersection_line_between_planes(plane_eq1, plane_eq2)
        #     candidates_projected_to_line = project_points_to_lines_torch(vertices[candidates], torch.cat(line_eq).unsqueeze(0))[0]
        #     signed_dist_to_line = torch.norm(candidates_projected_to_line - vertices[vertex_on_line], dim=1) * torch.sign(torch.dot(candidates_projected_to_line - vertices[vertex_on_line], line_eq[1] - line_eq[0]))
        #     left_side_candidates = candidates[signed_dist_to_line < 0]
        #     right_side_candidates = candidates[signed_dist_to_line > 0]

        #     if len(left_side_candidates) >= 2:
        #         # merge the shortest one
        #         dists = torch.norm(vertices[vertex_on_line] - vertices[left_side_candidates], dim=1)
        #         candidate_with_min_dist = left_side_candidates[torch.argmin(dists)]
        #         if dists.min() < add_to_line_max_dist:
        #             # insert the candidate into the polygon that doesn't contain it
        #             polygon_1 = old_polygons[vertex_constraints[0]]
        #             polygon_2 = old_polygons[vertex_constraints[1]]
        #             insert_into_polygon = polygon_1 if vertex_on_line in polygon_2 else polygon_2
        #             # check whether to insert it before or after the vertex_on_line
        #             insert_into_polygon.insert(insert_into_polygon.index(vertex_on_line) + 1, candidate_with_min_dist)
        #     elif len(right_side_candidates) >= 2:
        #         dists = torch.norm(vertices[vertex_on_line] - vertices[right_side_candidates], dim=1)
        #         candidate_with_min_dist = right_side_candidates[torch.argmin(dists)]
        #         if dists.min() < add_to_line_max_dist:
        #             # insert the candidate into the polygon that doesn't contain it
        #             polygon_1 = old_polygons[vertex_constraints[0]]
        #             polygon_2 = old_polygons[vertex_constraints[1]]
        #             insert_into_polygon = polygon_1 if vertex_on_line in polygon_2 else polygon_2
        #             insert_into_polygon.insert(insert_into_polygon.index(vertex_on_line) + 1, candidate_with_min_dist)

        # new_polygons = []
        # for i, polygon in enumerate(old_polygons):
        #     new_polygon = []
        #     for j, vertex in enumerate(polygon):
        #         if not deleted[vertex]:
        #             new_polygon.append(vertex)
        #     new_polygons.append(new_polygon)
        
        # self.polygons_closed = new_polygons
        #     # break

        # self.polygons_closed = old_polygons
        
        # if plot:
        #     simplified_mesh = self.export_triangle_mesh()

        #     new_point_cloud = o3d.geometry.PointCloud()
        #     new_point_cloud.points = o3d.utility.Vector3dVector(new_points)
        #     # colors = np.zeros((len(new_point_cloud.points), 3))
        #     # colors[deleted] = [1, 0, 0]

        #     o3d.visualization.draw_geometries([simplified_mesh, new_point_cloud])

        # self.clean()
            
        # if len(vertices) > len(self.get_vertices()):
        #     print(f"Deleted {deleted.sum()} vertices. N remaining vertices: {len(vertices) - deleted.sum()}")
        #     self.simplify_edges()
        
    # @torch.no_grad()
    # def find_mergeable_edges(self, max_deg=20, dist_thresh=0.005):
    #     """Merge parallel edges of a polygon. Checks each vertex whether the angle between the two edges is less than max_deg"""
    #     with torch.no_grad():
    #         vertices_3d = self.get_vertices()
    #     # vertex_polygon_assignments = self.get_vertex_polygons()
    #     # is_candidate = torch.ones(len(vertices_3d), dtype=torch.bool)
    #     # # replacements = []

    #     # for j, vertices in enumerate(self.polygons_closed):
    #     #     polygon_vertices_2d = self.project_points_to_planes_3d_to_2d(vertices_3d[vertices], torch.Tensor([j]).repeat(len(vertices)))
    #     #     # polygon_replacements = torch.arange(len(vertices), dtype=torch.long)
    #     #     for i in range(len(vertices)):
    #     #         p1 = polygon_vertices_2d[i]
    #     #         p2 = polygon_vertices_2d[(i + 1) % len(vertices)]
    #     #         p3 = polygon_vertices_2d[(i + 2) % len(vertices)]
    #     #         v1 = p1 - p2
    #     #         v2 = p3 - p2
    #     #         angle = torch.acos(torch.clamp(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8), -1 + 1e-8, 1 - 1e-8)) * 180 / np.pi
    #     #         angle = 180 - angle
    #     #         if not(angle < max_deg or torch.linalg.norm(p1 - p2) < dist_thresh):
    #     #             is_candidate[vertices[(i + 1) % len(vertices)]] = False

    #     # is_candidate = is_candidate.to(vertices_3d.device)

    #     # Vectorized version
    #     is_candidate_vectorized = torch.ones(len(vertices_3d), dtype=torch.bool, device=vertices_3d.device)
    #     polygon_vertices_2d = self.get_polygon_vertices_2d()
    #     for i, vertices in enumerate(self.polygons_closed):
    #         p1 = polygon_vertices_2d[i]
    #         p2 = torch.roll(p1, -1, dims=0)
    #         p3 = torch.roll(p1, -2, dims=0)
    #         v1 = p1 - p2
    #         v2 = p3 - p2
    #         angles = torch.acos(torch.clamp(torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8), -1 + 1e-8, 1 - 1e-8)) * 180 / np.pi
    #         angles = 180 - angles
    #         edge_lengths = torch.norm(p1 - p2, dim=1)
    #         mask = (angles < max_deg) | (edge_lengths < dist_thresh)
            
    #         vertices = torch.Tensor(vertices).long().to(vertices_3d.device)
    #         vertices = torch.roll(vertices, -1, dims=0)
    #         is_candidate_vectorized[vertices[~mask]] = False
            
    #     # assert torch.all(is_candidate == is_candidate_vectorized), f"Vectorized and non-vectorized version do not agree"
    #     is_candidate = is_candidate_vectorized
    #     return is_candidate
    

    def flip_misaligned_triangles(self):
        # flip triangles if the normal doesn't agree with the (original) plane normal
        v0, v1, v2 = self.get_vertices()[self.triangles].unbind(1)
        triangle_normals = (v1 - v0).cross(v2 - v1, dim=1)
        triangle_normals = triangle_normals / triangle_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=1e-14
        )
        triangle_plane_normals = self.vertices.get_or_compute_planes().detach()[self.triangle_polygons][:, :3]
        # triangle_plane_normals = self.polygon_plane_eqs[self.triangle_polygons][:, :3]
        switch_mask = (triangle_normals * triangle_plane_normals).sum(dim=1) < 0
        self.triangles[switch_mask] = torch.stack([self.triangles[switch_mask][:, 0], self.triangles[switch_mask][:, 2], self.triangles[switch_mask][:, 1]], dim=1)


    def delete_triangles(self, deletion_mask):
        self.triangles = self.triangles[~deletion_mask]
        self.triangle_polygons = self.triangle_polygons[~deletion_mask]
        
        
    def compute_triangle_areas(self):
        polygon_vertices = self.get_vertices()
        triangle_vertices_3d = polygon_vertices[self.triangles]
        triangle_polygons = self.triangle_polygons.unsqueeze(-1).repeat(1, 3)
        planes = self.get_planes().detach()
        triangle_vertices_2d = self.project_points_to_planes_3d_to_2d(triangle_vertices_3d.reshape(-1, 3), triangle_polygons.reshape(-1), planes).reshape(-1, 3, 2)
        triangle_areas = 0.5 * torch.abs(
            triangle_vertices_2d[:, 0, 0] * (triangle_vertices_2d[:, 1, 1] - triangle_vertices_2d[:, 2, 1]) +
            triangle_vertices_2d[:, 1, 0] * (triangle_vertices_2d[:, 2, 1] - triangle_vertices_2d[:, 0, 1]) +
            triangle_vertices_2d[:, 2, 0] * (triangle_vertices_2d[:, 0, 1] - triangle_vertices_2d[:, 1, 1])
        )
        return triangle_areas


    def clean(self, retriangulate=False):
        """deletes all vertices that are not part of a polygon and reindexes the polygon"""
        self.clear_cache()
        
        old_vertices = self.get_vertices()
        
        # delete triangles with only one inner edge and small area
        # inner_edges = self.get_inner_edge_mask_with_cache()
        # triangles_with_one_inner_edge = inner_edges.sum(dim=1) == 1
        # triangle_areas = self.compute_triangle_areas()
        # small_triangles = triangle_areas < 0.0001
        # polygon_spike_mask = small_triangles

        
        # duplicate_vertex_mask = (self.triangles[:, 0] == self.triangles[:, 1]) | (self.triangles[:, 1] == self.triangles[:, 2]) | (self.triangles[:, 0] == self.triangles[:, 2])
        # self.triangles = self.triangles[~duplicate_vertex_mask) & (~polygon_spike_mask)]
        # self.triangle_polygons = self.triangle_polygons[(~duplicate_vertex_mask) & (~polygon_spike_mask)]
        # unique_triangles, indices = torch.unique(torch.cat([self.triangles.sort(dim=1).values, self.triangle_polygons[:, None]], dim=1), return_inverse=True, dim=0)
        # if len(unique_triangles) < len(self.triangles):
        #     print(f"Warning! Found {len(self.triangles) - len(unique_triangles)} duplicate triangles")
        #     self.triangles = unique_triangles[:, :3]
        #     self.triangle_polygons = unique_triangles[:, 3]

        mentioned_vertices = torch.zeros(len(old_vertices), dtype=torch.bool, device=old_vertices.device)
        mentioned_vertices[self.triangles] = True
        old2new = self.vertices.filter_mask_based(mentioned_vertices)
        self.clear_cache()

        self.triangles = old2new[self.triangles]
        self.flip_misaligned_triangles()
        if retriangulate:
            self.retriangulate_polygons()
        self.clear_cache()
        self.compute_adjacency()



    def compute_length_of_polygons(self, rootlen=False, downweight_shared=True):
        """Compute the length of a polygon in torch given its vertices"""
        length = 0
        all_vertices = self.get_vertices()
        planes_per_vertex = self.vertices.get_num_assigned_planes_per_vertex()
        for ixes in self.polygons_closed:
            vertices = all_vertices[ixes] 
            edges = vertices - torch.roll(vertices, 1, dims=0)
            if downweight_shared:
                min_planes_per_edge = torch.min(planes_per_vertex[ixes], torch.roll(planes_per_vertex[ixes], 1, dims=0)).to(all_vertices.device)
                edges = edges * torch.pow(0.1, min_planes_per_edge - 1)[:, None]
                # edges = edges * (min_planes_per_edge == 1)[:, None]
            if rootlen:
                length += torch.sum(torch.sqrt(torch.norm(edges, dim=1) + 1e-8))
            else:
                length += 0.1 * torch.sum(torch.norm(edges, dim=1))
        return length
    

    def vertex_magnetism_loss(self, threshold=0.0025):
        """close vertices attract each other"""
        vertices = self.get_vertices()
        vertices_2d = vertices[:, :2]
        dists = (vertices_2d[:, None] - vertices_2d[None, :]).norm(dim=-1)
        print("Median number of inliers/vertex:", torch.median((dists < threshold).sum(dim=1)))
        dists = dists * (dists < threshold)
        return torch.mean(torch.sqrt(dists + 1e-8))
    

    def vertex_magnetism_loss_knn(self, k=5):
        """close vertices attract each other. Vertices on multiple planes attract stronger"""
        vertices = self.get_vertices()
        n_assigned_planes = self.vertices.get_num_assigned_planes_per_vertex()
        vertices_2d = vertices[:, :2]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(vertices_2d.detach().cpu().numpy())
        _, indices = nbrs.kneighbors(vertices_2d.detach().cpu().numpy())
        neigbors = vertices_2d[indices]
        dists = (neigbors - vertices_2d[:, None, :]).norm(dim=-1)
        weights = n_assigned_planes[indices].float()
        dists = dists * weights
        return torch.mean(torch.sqrt(dists + 1e-8))


    # def edge_magnetism_loss(self, threshold=0.1, batch_size=1000):
    #     """edges attract vertices"""
    #     vertices = self.get_vertices()
    #     edges = []
    #     for polygon in self.polygons_closed:
    #         edges.append(torch.stack([vertices[polygon], torch.roll(vertices[polygon], 1, dims=0)], dim=1))

    #     edges = torch.cat(edges)

    #     edge_magnetism = 0
    #     for i in tqdm(range(0, len(edges), batch_size)):
    #         edge_batch = edges[i:i+batch_size]
    #         distances = distances_to_line_segment_3d(edge_batch[:, 0], edge_batch[:, 1], vertices)
    #         edge_magnetism += torch.sum(distances[distances < threshold])
    #     return edge_magnetism
    # def plane_magnetism_loss(self, threshold=0.05):
    #     """planes attract vertices"""
    #     vertices = self.get_vertices()
    #     planes = self.planes
    #     dist = 0
    #     from tqdm import tqdm
    #     n_inliers = []
    #     for plane in tqdm(planes):
    #         _, dists = project_points_3d_to_3d_plane(vertices,  plane[None, :] * torch.ones(len(vertices))[:, None])
    #         dist += torch.sum(dists[dists < threshold])
    #         n_inliers.append((dists < threshold).sum())
    #     print("Median number of inliers/vertex:", torch.median(torch.Tensor(n_inliers)))
    #     return torch.sum(dists)
    
    # def get_regularization_loss(self):
    #     len_loss = self.compute_length_of_polygons()
    #     # vertex_magnetism_loss = self.vertex_magnetism_loss()
    #     # constraint_loss = self.vertices.get_constraint_loss()
    #     # plane_magnetism = self.plane_magnetism_loss()
    #     vertex_magnetism_loss_knn = self.vertex_magnetism_loss_knn()
    #     return 0.2 * len_loss + vertex_magnetism_loss_knn  # + 10 * vertex_magnetism_loss # + plane_magnetism
    
    def get_regularization_loss(self):
        contour_edges = self.get_contour_edges()


    def get_params(self):
        return self.vertices.get_params()

    def get_triangle_colors(self):
        return self.polygon_colors[self.triangle_polygons]


    def clear_cache(self):
        if hasattr(self, 'vertices_cache'):
            del self.vertices_cache
        if hasattr(self, 'polygon_vertices_cache'):
            del self.polygon_vertices_cache



    def get_plane_merge_candidate_pairs(self, k=10, max_dist = 0.01, max_angle=45, only_same_class=False):
        """Gets polygon instances that we could merge: Gets the knn pairs with distinct polygons"""
        
        vertices = self.get_vertices()
        if True:
            planes = self.get_planes()
            triangle_vertices = vertices[self.triangles]
            triangle_normals = planes[self.triangle_polygons][:, :3]
            face_indices, vertex_face_distances, closest_points = find_k_closest_triangles(vertices, triangle_vertices, triangle_normals, k=10)

            N = len(vertices)  
            MAX_N_PLANES_PER_VERTEX = self.vertices.vertex_plane_assignments.shape[-1]          
            vertex_polygons = self.vertices.vertex_plane_assignments.unsqueeze(1).repeat(1, k, 1)
            face_polygons = self.triangle_polygons[face_indices].unsqueeze(-1).repeat(1, 1, MAX_N_PLANES_PER_VERTEX)
            pairs = torch.stack([vertex_polygons, face_polygons], dim=-1) # shape (N, k, MAX_N_PLANES_PER_VERTEX, 2)

            close_enough = vertex_face_distances < max_dist
            no_unassigned = (pairs[:, :, :, 0] != -1) & (pairs[:, :, :, 1] != -1)
            not_same = pairs[:, :, :, 0] != pairs[:, :, :, 1]
            pairs = pairs[close_enough.unsqueeze(-1) & no_unassigned & not_same]
            angle = torch.acos(torch.clamp(torch.sum(planes[pairs[:, 0]][:, :3] * planes[pairs[:, 1]][:, :3], dim=-1), -1.0, 1.0)) * 180 / np.pi
            # angle = torch.minim< um(angle, 180 - angle)
            pairs = pairs[angle < max_angle]
            if only_same_class:
                pairs = pairs[self.polygon_classes[pairs[:, 0]] == self.polygon_classes[pairs[:, 1]]]
                
            pairs = pairs.sort(dim=-1)[0]
            polygon_pairs = torch.unique(pairs, dim=0)
            
            # neighbor_ixes = self.get_vertex_knn_indices(vertices, k=k)
            # # polygons = self.get_vertex_polygons()
            # vertex_polygons = self.vertices.vertex_plane_assignments
            # neighbor_polygons = self.vertices.vertex_plane_assignments[neighbor_ixes]

            # polygon_pairs = torch.stack([vertex_polygons[:, None, :].repeat(1, k, 1), neighbor_polygons], dim=-1)
            # polygon_pairs = polygon_pairs[(vertices[:, None, :] - vertices[neighbor_ixes]).norm(dim=-1) < max_dist]
            # MAX_N_PLANES = vertex_polygons.shape[-1]
            # # polygon_pairs has shape (n_vertices, k, MAX_N_PLANES, 2). We now need to form all combinations of plane assigments
            # polygon_pairs_flat = []
            # for i in range(MAX_N_PLANES):
            #     for j in range(MAX_N_PLANES):
            #         polygon_pairs_flat.append(torch.stack([polygon_pairs[:, i, 0], polygon_pairs[:, j, 1]], dim=-1))
            # polygon_pairs = torch.cat(polygon_pairs_flat, dim=0).reshape(-1, 2)
            # # filter out the pairs where the polygons are the same or -1 is contained
            # mask = (polygon_pairs[:, 0] != polygon_pairs[:, 1]) & (polygon_pairs[:, 0] != -1) & (polygon_pairs[:, 1] != -1)
            # polygon_pairs = polygon_pairs[mask]
            # polygon_pairs = polygon_pairs.sort(dim=-1)[0]
            # polygon_pairs = torch.unique(polygon_pairs, dim=0)
        else:
            vertex_merge_candidates = self.find_mergeable_vertices_fast(max_dist=max_dist)
            candidate_planes = self.vertices.vertex_plane_assignments[vertex_merge_candidates]
            MAX_N_PLANES = candidate_planes.shape[-1]
            # polygon_pairs has shape (n_vertices, k, MAX_N_PLANES, 2). We now need to form all combinations of plane assigments
            polygon_pairs_flat = []
            for i in range(MAX_N_PLANES):
                for j in range(MAX_N_PLANES):
                    polygon_pairs_flat.append(torch.stack([candidate_planes[:, 0, i], candidate_planes[:, 1, j]], dim=-1))

            pairs = torch.cat(polygon_pairs_flat, dim=0).reshape(-1, 2)
            # pairs = torch.cat([self.vertices.vertex_plane_assignments[:, [0, 1]], self.vertices.vertex_plane_assignments[:, [1, 2]], self.vertices.vertex_plane_assignments[:, [2, 0]]], dim=0)
            pairs = torch.sort(pairs, dim=1)[0]
            pairs, counts = torch.unique(pairs, return_counts=True, dim=0)
            mask = (counts > 1) & (pairs[:, 0] != pairs[:, 1]) & (pairs[:, 0] != -1) & (pairs[:, 1] != -1)
            polygon_pairs = pairs[mask] 
        return polygon_pairs


    
    def plane_error(self, plane_eq, vertex_coords, vertex_normals):
        # projection error
        plane_eq = plane_eq.clone() / (torch.norm(plane_eq[:3]) + 1e-10)
        _, projection_error = project_points_3d_to_3d_plane(vertex_coords, plane_eq[None, :].repeat(len(vertex_coords), 1))
        projection_error = torch.mean(torch.abs(projection_error))
        # normal error
        plane_normal = plane_eq[:3]
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-10)
        dot_products = torch.matmul(vertex_normals, plane_normal)
        angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        angles_deg = angles * 180 / np.pi
        # angles_deg = torch.min(angles_deg, 180 - angles_deg)
        normal_error = torch.mean(angles_deg)
        # feature error (cosine similarity)
        # feature_error = torch.mean(1 - torch.matmul(vertex_features, plane_feature) / (torch.norm(vertex_features, dim=1) * torch.norm(plane_feature) + 1e-8))

        return projection_error, normal_error
        # print(f"Projection Error: {lambda_projection * projection_error}, Normal Error: {lambda_normal * normal_error}, Feature Error: {lambda_feature * feature_error}")
        # return lambda_projection * projection_error + lambda_normal * normal_error + lambda_feature * feature_error


    @torch.no_grad()
    def align_plane_normals_with_target_points(self, target_point_face_ids, target_normals, plane_ids=None):
        """Iterates over all planes and aligns the normal with the target normal: that is, flips it if more than half of the target points point the other way"""
        assert len(target_point_face_ids) == len(target_normals)
        assert target_point_face_ids.max() < len(self.triangle_polygons)
        plane_eqs = self.vertices.get_or_compute_planes().clone()
        target_point_plane_ids = self.triangle_polygons[target_point_face_ids]
        n_flip = 0
        if plane_ids is None:
            plane_ids = range(len(plane_eqs))
        for i in plane_ids:
            target_point_ids = target_point_plane_ids == i
            if target_point_ids.sum() == 0:
                continue
            target_normals_i = target_normals[target_point_ids]
            dot_products = torch.matmul(target_normals_i, plane_eqs[i, :3])
            if (dot_products < 0).sum() > len(dot_products) / 2: # flip by majority vote
                plane_eqs[i] = -plane_eqs[i]
                n_flip += 1
        self.vertices.planes.data = plane_eqs
        print(f"Flipped {n_flip} normals")

        
    

    def merge_planes(self, target_points, target_point_face_ids, target_normals, target_features, max_projection_cost=0.5, max_normal_cost=0.2, split_vertices=False, plot=False, only_merge_same_class=False, align_normals=True, recompute_plane_eqs=False):

        if split_vertices:
            planes = self.vertices.get_or_compute_planes()
            # split the polygons
            old_vertices = self.get_vertices()
            vertices = []
            vertex_polygons = []
            triangles = []
            triangle_polygons = []
            colors = []
            plane_eqs = []
            polygon_classes = []
            for polygon in range(self.vertices.n_planes):
                polygon_triangles = self.triangles[self.triangle_polygons == polygon]
                polygon_vertices = old_vertices[polygon_triangles]
                unique_vertices, reindexed_triangles = torch.unique(polygon_vertices.view(-1, 3), return_inverse=True, dim=0)
                reindexed_triangles = reindexed_triangles.view(-1, 3)

                triangles += list(reindexed_triangles + len(vertices))
                vertices += list(unique_vertices)
                triangle_polygons += list(torch.tensor([polygon] * len(reindexed_triangles)))
                vertex_polygons += list(torch.tensor([polygon] * len(unique_vertices)))
                colors += [self.polygon_colors[polygon]]
                plane_eqs.append(planes[polygon])
                polygon_classes.append(self.polygon_classes[polygon])
            # assert reindexed_triangles.max() < len(vertices), f"Reindexed triangles max: {reindexed_triangles.max()}, len(vertices): {len(vertices)}"

            mesh_vertices = torch.stack(vertices, dim=0)
            mesh_triangles = torch.stack(triangles, dim=0)
            mesh_triangle_polygons = torch.stack(triangle_polygons, dim=0)
            mesh_vertex_polygons = torch.stack(vertex_polygons, dim=0)
            triangle_colors = torch.stack(colors, dim=0)
            plane_eqs = torch.stack(plane_eqs, dim=0)
            polygon_classes = torch.stack(polygon_classes, dim=0)

            self.update_triangles(mesh_triangles, mesh_vertices, mesh_triangle_polygons, mesh_vertex_polygons, triangle_colors, plane_eqs, polygon_classes=polygon_classes)

        
        
        # max_dist = 0.1
        # projection_baseline_error = 0.08 # roughly 10% of what we expect
        # normal_baseline_error = 0.5 # roughly 10% of what we expect
        # feature_baseline_error = 0.1
        # max_angle = 45

        print("Merging planes")

        max_dist = self.config.max_plane_merge_dist
        projection_baseline_error = self.config.projection_baseline_error
        normal_baseline_error = self.config.normal_baseline_error
        max_angle = self.config.max_plane_merge_angle
        

        candidates = self.get_plane_merge_candidate_pairs(max_dist=max_dist, max_angle=max_angle, only_same_class=only_merge_same_class)
        target_point_polygon_ids = self.triangle_polygons[target_point_face_ids]
        target_point_polygon_ids[target_point_face_ids == -1] = -1

        vertices_per_polygon = {}
        normals_per_polygon = {}
        features_per_polygon = {}
        for i in range(self.vertices.n_planes):
            vertices_per_polygon[i] = target_points[target_point_polygon_ids == i]
            normals_per_polygon[i] = target_normals[target_point_polygon_ids == i]
            features_per_polygon[i] = target_features[target_point_polygon_ids == i]

        merge_costs = {}

        if recompute_plane_eqs:
            new_plane_eqs = torch.zeros((self.vertices.n_planes, 4), device=self.device, dtype=self.vertices.planes.dtype)
            for i in range(len(new_plane_eqs)):
                vertices, normals = vertices_per_polygon[i], normals_per_polygon[i]
                new_plane_eqs[i] = fit_plane_torch(vertices, normals)
            self.vertices.planes.data = new_plane_eqs
        
        planes = self.vertices.get_or_compute_planes()

        for (i, j) in candidates.cpu().numpy():
            vertices_1, vertices_2 = vertices_per_polygon[i], vertices_per_polygon[j]
            normals_1, normals_2 = normals_per_polygon[i], normals_per_polygon[j]
            # features_1, features_2 = features_per_polygon[i], features_per_polygon[j]

            
            if len(vertices_1) < 3 or len(vertices_2) < 3:
                continue

            plane_eq_1 = fit_plane_torch(vertices_1, normals_1)
            plane_eq_2 = fit_plane_torch(vertices_2, normals_2)
            # plane_eq_1 = self.vertices.get_or_compute_planes()[i]
            # plane_eq_2 = self.vertices.get_or_compute_planes()[j]

            # plane_feature_1 = torch.mean(vertices_1, dim=0)
            # plane_feature_2 = torch.mean(vertices_2, dim=0)

            weight_1 = len(vertices_1) / (len(vertices_1) + len(vertices_2))
            weight_2 = 1 - weight_1


            projection_error_1, normal_error_1 = self.plane_error(plane_eq_1, vertices_1, normals_1)
            projection_error_2, normal_error_2 = self.plane_error(plane_eq_2, vertices_2, normals_2)
            # old_error = error_1 + error_2

            new_plane_eq = fit_plane_torch(torch.cat([vertices_1, vertices_2]), torch.cat([normals_1, normals_2]))
            # new_plane_feature = weight_1 * plane_feature_1 + weight_2 * plane_feature_2
            new_color = weight_1 * self.polygon_colors[i] + weight_2 * self.polygon_colors[j]
            new_projection_error_1, new_normal_error_1 = self.plane_error(new_plane_eq, vertices_1, normals_1)
            new_projection_error_2, new_normal_error_2 = self.plane_error(new_plane_eq, vertices_2, normals_2)
            # new_error = new_error_1 + new_error_2
            
            # also test whether we can simply project one onto the other
            projection_error_1_, normal_error_1_ = self.plane_error(self.vertices.get_or_compute_planes()[j], vertices_1, normals_1)
            new_projection_error_1 = min(new_projection_error_1, projection_error_1_)
            new_normal_error_1 = min(new_normal_error_1, normal_error_1_)

            projection_error_2_, normal_error_2_ = self.plane_error(self.vertices.get_or_compute_planes()[i], vertices_2, normals_2)
            new_projection_error_2 = min(new_projection_error_2, projection_error_2_)
            new_normal_error_2 = min(new_normal_error_2, normal_error_2_)
            

            # flip the new normal if it points in the wrong direction
            if torch.dot(new_plane_eq[:3], plane_eq_1[:3]) < 0:
                new_plane_eq = -new_plane_eq

            def cost(old_error, new_error, baseline_error): 
                return (new_error - old_error) / (max(old_error, baseline_error))
            
            # merge_cost = (new_error - old_error) / old_error
            # merge_cost = max((new_error_1 - error_1) / error_1, (new_error_2 - error_2) / error_2)
            projection_cost = max(cost(projection_error_1, new_projection_error_1, projection_baseline_error), cost(projection_error_2, new_projection_error_2, projection_baseline_error))
            normal_cost = max(cost(normal_error_1, new_normal_error_1, normal_baseline_error), cost(normal_error_2, new_normal_error_2, normal_baseline_error))
            # feature_cost = 0 * max(cost(feature_error_1, new_plane_feature_error_1, feature_baseline_error), cost(feature_error_2, new_plane_feature_error_2, feature_baseline_error))
            merge_cost = max(projection_cost, normal_cost)
            print(f"({i}, {j})Merge cost: {merge_cost}, projection cost: {projection_cost}, normal cost: {normal_cost}")
                            
            color_i, color_j = self.polygon_colors[i].clone(), self.polygon_colors[j].clone()
            # reassign red and blue
            # import open3d as o3d
            # self.polygon_colors[i] = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
            # self.polygon_colors[j] = torch.tensor([0.0, 0.0, 1.0]).to(self.device)
            # mesh = self.export_triangle_mesh()
            # o3d.io.write_triangle_mesh(f"/mnt/usb_ssd/bieriv/layout-estimation-outputs/01-20-dslr-mesh-fitting/scannet++/f3d64c30f8/all-classes/fitted_mesh_merge_{i}_{j}.ply", mesh)
            # self.polygon_colors[i] = color_i
            # self.polygon_colors[j] = color_j
            assert not torch.isinf(merge_cost), "Merge cost is inf"
            assert not torch.isnan(merge_cost), "Merge cost is nan"
            assert not torch.isinf(projection_cost), "Projection cost is inf"
            assert not torch.isnan(projection_cost), "Projection cost is nan"
            assert not torch.isinf(normal_cost), "Normal cost is inf"
            assert not torch.isnan(normal_cost), "Normal cost is nan"

            
            if projection_cost < max_projection_cost and normal_cost < max_normal_cost:
                bigger_plane = i if len(vertices_1) > len(vertices_2) else j
                smaller_plane = j if bigger_plane == i else i
                merge_costs[(bigger_plane, smaller_plane)] = {"merge_cost": merge_cost, "plane_eq": new_plane_eq, "color": new_color.tolist()}



        # sort the merge candidates, mark already merged planes
        merge_costs = dict(sorted(merge_costs.items(), key=lambda item: item[1]["merge_cost"]))
        merged_already = torch.zeros(self.vertices.n_planes, dtype=torch.bool, device=self.device)
        deleted = torch.zeros(self.vertices.n_planes, dtype=torch.bool, device=self.device)
        n_merged = 0
        for (i, j), merge_info in merge_costs.items():
            if merged_already[i] or merged_already[j]:
                continue
            else:
                if plot:
                    color_i, color_j = self.polygon_colors[i].clone(), self.polygon_colors[j].clone()
                    # reassign red and blue
                    import open3d as o3d
                    self.polygon_colors[i] = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
                    self.polygon_colors[j] = torch.tensor([0.0, 0.0, 1.0]).to(self.device)
                    mesh = self.export_triangle_mesh()
                    o3d.io.write_triangle_mesh(f"merge_{i}_{j}.ply", mesh)
                    self.polygon_colors[i] = color_i
                    self.polygon_colors[j] = color_j


                merged_already[i] = True
                merged_already[j] = True
                planes[i] = torch.tensor(merge_costs[(i, j)]["plane_eq"])
                planes[j] = torch.tensor(merge_costs[(i, j)]["plane_eq"])
                for vertex in torch.where((self.vertices.vertex_plane_assignments == j).any(dim=-1))[0]:
                    joint_constraints = set(self.vertices.get_vertex_constraints(vertex).tolist())
                    joint_constraints.remove(j)
                    joint_constraints.add(i)
                    self.vertices.update_vertex_constraints(vertex, torch.tensor(list(joint_constraints)))
                # self.polygon_colors[i] = torch.tensor([1.0, 0.0, 0.0])  # Red
                # self.polygon_colors[j] = torch.tensor([0.0, 0.0, 1.0])  # Blue
                self.triangle_polygons[self.triangle_polygons == j] = i
                deleted[j] = True
                self.polygon_colors[i] = torch.tensor(merge_costs[(i, j)]["color"])
                self.polygon_colors[j] = torch.tensor(merge_costs[(i, j)]["color"])
                n_merged += 1
                print(f"Merged planes {i} and {j} with cost {merge_info['merge_cost']}")

        self.vertices.planes.data = planes
        print(f"Merged {n_merged} planes. N remaining planes: {self.vertices.n_planes - n_merged}")
        # self.clear_cache()
        # self.simplify_vertices()
        # self.flip_misaligned_triangles()
        # self.attach_vertices_to_close_lines()
        # self.compute_adjacency()
        
        assert not torch.isin(torch.where(deleted)[0], self.triangle_polygons).any(), "Deleted a polygon that is still in use"
        assert not torch.isin(torch.where(deleted)[0], self.vertices.vertex_plane_assignments).any(), "Deleted a polygon that is still in use"
        
        self.align_plane_normals_with_target_points(target_point_face_ids, target_normals, plane_ids=torch.where(merged_already)[0])



    def find_holes(self, max_contour_length=1, max_dist=0.01):
        """Find holes in the mesh defined by connected paths along non watertight edges"""
        vertices = self.get_vertices()
        non_watertight_edges = self.get_non_watertight_edges()
        contours = triangulate_polygon_from_edges2(non_watertight_edges.detach().cpu().numpy())
        # clean the contours: at least 3 edges 
        contours = [contour for contour in contours if len(contour) > 2]
        # contour length < max_contour_length. Compute by summing the edge lengths
        # contours = [contour for contour in contours if sum([torch.norm(vertices[contour[i]] - vertices[contour[(i + 1) % len(contour)]]).item() for i in range(len(contour))]) < max_contour_length]


        fixed_planar_contours = []
        fixed_line_contours = []

        for contour in contours:
            edge_polygons = []
            for i in range(len(contour)):
                polygons = self.vertices.vertex_plane_assignments[[contour[i], contour[(i + 1) % len(contour)]]]
                edge_polygons += list(set(polygons.view(-1).tolist()) - {-1})
            
            unique_polygons, unique_counts = torch.unique(torch.tensor(edge_polygons), return_counts=True)
            unique_polygons = unique_polygons[unique_counts > 1] # if it's only one edge we can ignore it: projecting to the line is probably a good choice
            print(f"Contour: {contour}, Polygon counts: {list(zip(unique_polygons.tolist(), unique_counts.tolist()))}")
            if len(unique_polygons) <= 1:
                # # compute area
                # vertices2d = vertices[contour]
                # remove hole: project all the points to the mean
                center_point = torch.mean(vertices[contour], dim=0)
                self.vertices.vertices.data[contour] = center_point
                fixed_planar_contours.append(contour)

            # if len(unique_polygons) == 2:
            #     # remove hole: project all the points to the line
            #     p1, p2 = self.vertices.get_or_compute_planes()[unique_polygons]
            #     angle_between_planes = torch.acos(torch.clamp(torch.dot(p1[:3], p2[:3]), -1.0 + 1e-8, 1.0 - 1e-8)) * 180 / np.pi
            #     if angle_between_planes < 30:
            #         continue
            #     line_eq = intersection_line_between_planes(p1, p2)
            #     projected_points = project_points_to_lines_torch(vertices[contour], torch.cat(line_eq).unsqueeze(0))[0]
            #     self.vertices.vertices.data[contour] = projected_points
            #     fixed_line_contours.append(contour)

            # if len(unique_polygons) == 3:
            #     # remove hole: project all the points to the fixed point
            #     p1, p2 = self.vertices.get_or_compute_planes()[unique_polygons[:2]]
            #     line_eq = intersection_line_between_planes(p1, p2)
            #     fixed_point = line_plane_intersection( torch.cat(line_eq), self.vertices.get_or_compute_planes()[unique_polygons[2]])
            #     # only project if close
            #     self.vertices.vertices.data[contour][(self.vertices.vertices.data[contour] - fixed_point).norm(dim=-1) < max_dist] = fixed_point

        self.simplify_vertices()
        self.clean()

        return fixed_planar_contours, fixed_line_contours
    

    def add_triangles_to_polygons(self, triangle_vertices: torch.Tensor, triangle_polygons: torch.Tensor):
        """Adds triangles to the mesh and assigns them to polygons"""
        assert triangle_polygons.shape == (len(triangle_vertices),)
        assert triangle_vertices.shape[1:] == (3, 3)

        if len(triangle_polygons) == 0:
            return

        n_existing_vertices = len(self.get_vertices())

        new_triangles = []
        new_vertices = []
        new_vertex_polygons = []
        for i, vertices in enumerate(triangle_vertices.tolist()):
            new_vertices.extend(vertices)
            new_triangles.append(n_existing_vertices + torch.arange(len(new_vertices) - 3, len(new_vertices)).long())
            new_vertex_polygons += [triangle_polygons[i]] * 3


        new_triangles = torch.stack(new_triangles, dim=0).to(self.device)
        new_vertices = torch.tensor(new_vertices).to(self.device)
        new_triangle_polygons = triangle_polygons.to(self.device)
        new_vertex_polygons = torch.stack(new_vertex_polygons, dim=0).to(self.device)
        self.triangles = torch.cat([self.triangles, new_triangles], dim=0)
        self.triangle_polygons = torch.cat([self.triangle_polygons, new_triangle_polygons], dim=0)
        new_vertex_constraints = -1 * torch.ones((len(new_vertices), self.vertices.vertex_plane_assignments.shape[1]), dtype=torch.long, device=self.device)
        new_vertex_constraints[:, 0] = new_vertex_polygons
        self.vertices.add_vertices(new_vertices=new_vertices, new_vertex_constraints=new_vertex_constraints)
    

    def delete_triangles(self, deletion_mask):
        self.triangles = self.triangles[~deletion_mask]
        self.triangle_polygons = self.triangle_polygons[~deletion_mask]
    
    @torch.no_grad()
    def find_nearest_polygon_belonging_to_class(self, query_points: torch.Tensor, class_ix: int, max_dist : float = -1):
        """Finds the nearest polygon belonging to a certain class"""
        vertices = self.get_vertices()
        mask = self.polygon_classes[self.triangle_polygons] == class_ix
        class_triangles = self.triangles[mask]
        class_triangle_polygon_ids = self.triangle_polygons[mask]

        if len(class_triangles) == 0:
            return torch.tensor([-1] * len(query_points)).to(self.device)
        # get nearest triangles
        triangle_ixes, _ = compute_closest_triangle_to_points_o3d(vertices, class_triangles, query_points)
        return class_triangle_polygon_ids[triangle_ixes]
    
    
    @torch.no_grad()
    def find_next_lower_polygon_belonging_to_class(self, query_points: torch.Tensor, class_ix: int, up_vector: torch.Tensor, margin : float = -5 * 1e-2, max_dist : float = 3):
        """Finds the nearest polygon belonging to a certain class"""
        vertices = self.get_vertices()
        class_polygons = torch.where(self.polygon_classes == class_ix)[0]
        plane_eqs = self.get_planes()
        # class_triangles = self.triangles[mask]
        # class_triangle_polygon_ids = self.triangle_polygons[mask]

        # polygon_plane_eqs = self.get_planes()[mask]
        # polygon_midpoint = torch.mean(vertices[class_triangles], dim=1)

        # midpoint_height = torch.matmul(polygon_midpoint, up_vector)


        closest_polygons = -1 * torch.ones(len(query_points), device=self.device, dtype=torch.long)
        closest_distances = float('inf') * torch.ones(len(query_points), device=self.device)

        for polygon_ix in class_polygons:
            polygon_triangles = self.triangles[self.triangle_polygons == polygon_ix]
            if len(polygon_triangles) == 0:
                continue
            polygon_vertices = vertices[polygon_triangles]
            polygon_midpoint = torch.mean(polygon_vertices, dim=0).mean(dim=0)
            query_point_is_above_plane_eq = torch.matmul(query_points - polygon_midpoint[None], up_vector) > margin # give it some slack
            above_ixes = torch.where(query_point_is_above_plane_eq)[0]
            query_points_above = query_points[query_point_is_above_plane_eq]
            triangle_ixes, _ = compute_closest_triangle_to_points_o3d(vertices, polygon_triangles, query_points_above)
            triangle_normals = plane_eqs[polygon_ix, :3].unsqueeze(0).repeat(len(query_points_above), 1)
            distances, _ = point_triangle_distance_vectorized.compute_distances(query_points_above, polygon_vertices[triangle_ixes], triangle_normals)
            close_enough = distances < closest_distances[above_ixes]
            closest_above_ixes = above_ixes[close_enough]
            closest_polygons[closest_above_ixes] = polygon_ix
            closest_distances[closest_above_ixes] = distances[close_enough]

        # color points by closest below polygon
        # out_dir = "/mnt/usb_ssd/bieriv/tmp/matterport_mesh_annot/WYY7iVyf5p8"
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(query_points.detach().cpu().numpy())
        # polygon_colors = torch.rand((len(self.vertices.planes) + 1, 3), device=self.device)
        # point_colors = polygon_colors[closest_polygons].detach().cpu().numpy()
        # pcd.colors = o3d.utility.Vector3dVector(point_colors)
        # o3d.io.write_point_cloud(f"{out_dir}/closest_below_polygon.ply", pcd)

        # original_colors = self.polygon_colors.clone()
        # self.polygon_colors = polygon_colors
        # self.polygon_colors[~torch.isin(torch.arange(len(polygon_colors)).to(self.device), torch.unique(closest_polygons))] = torch.tensor([0.0, 0.0, 0.0]).to(self.device)        
        # mesh = self.export_triangle_mesh()
        # o3d.io.write_triangle_mesh(f"{out_dir}/class_{class_ix}.ply", mesh)
        # self.polygon_colors = original_colors

        if max_dist is not None:
            mask = closest_distances > max_dist
            closest_polygons[mask] = -1

        return closest_polygons
    


    def count_ray_intersections(self, ray_origins, ray_dests, vertices, triangles, margin=0.01):
        """Counts the number of intersections of rays with the mesh"""
        does_intersect, primitive_ids, intersection_points, len_of_overlap = compute_ray_mesh_intersections_ray_tracing(vertices, triangles, ray_origins, ray_dests, margin=margin)
        return does_intersect.sum()
    

    @torch.no_grad()
    def try_extend_edge_to_line(self, edge_vertices, line_eq, ray_origins, ray_dests, ray_intersection_margin=0.01):
        """Tries to extend an edge to a plane: Tests how many intersections the change would generate"""
        edge_vertices_on_line, _ = project_points_to_lines_torch(edge_vertices, line_eq.unsqueeze(0))
        vertices_of_added_segment = torch.concatenate([edge_vertices, edge_vertices_on_line], dim=0)
        added_triangles = torch.tensor([[0, 1, 2], [1, 3, 2]])
        n_intersections = self.count_ray_intersections(ray_dests=ray_dests, ray_origins=ray_origins, vertices=vertices_of_added_segment, triangles=added_triangles, margin=ray_intersection_margin)
        added_triangle_vertices = vertices_of_added_segment[added_triangles]
        added_area = 0.5 * torch.abs(
            added_triangle_vertices[:, 0, 0] * (added_triangle_vertices[:, 1, 1] - added_triangle_vertices[:, 2, 1]) +
            added_triangle_vertices[:, 1, 0] * (added_triangle_vertices[:, 2, 1] - added_triangle_vertices[:, 0, 1]) +
            added_triangle_vertices[:, 2, 0] * (added_triangle_vertices[:, 0, 1] - added_triangle_vertices[:, 1, 1])
        ).sum()
        return vertices_of_added_segment, added_triangles, added_area, n_intersections

    @torch.no_grad()
    def get_polygon_edges_facing_direction(self, direction, polygon, plot=False):
        """Interprets the direction as a ray. 
        Classifies the edges into edges facing the ray (if we intersected with a parallel ray, we enter the triangle)
        and edges facing away from the ray (if we intersected with a parallel ray, we exit the triangle)"""
        vertices = self.get_vertices()
        plane = self.get_planes()[polygon]

        direction = direction.float()
        orthogonal_direction = torch.cross(plane[:3], direction)

        def compute_is_lower_edge(triangle_vertex_ids, direction, orthogonal_direction, is_inner_edge):
            triangle_vertices = vertices[triangle_vertex_ids]
            dot_products = torch.sum(triangle_vertices * direction[None, :], dim=-1)
            # sort: start with the vertex that is "closest" to a ray we shoot in this direction
            order = torch.argsort(dot_products)
            sorted_triangle_vertices = triangle_vertices[order]
            sorted_triangle_indices = triangle_vertex_ids[order]

            edge_1 = sorted_triangle_vertices[1] - sorted_triangle_vertices[0]
            edge_2 = sorted_triangle_vertices[2] - sorted_triangle_vertices[0]

            edge_1_direction = edge_1 / (torch.norm(edge_1, dim=-1, keepdim=True) + 1e-8)
            edge_2_direction = edge_2 / (torch.norm(edge_2, dim=-1, keepdim=True) + 1e-8)

            edge_1_to_orthogonal = torch.sum(edge_1_direction * orthogonal_direction, dim=-1)
            edge_2_to_orthogonal = torch.sum(edge_2_direction * orthogonal_direction, dim=-1)

            if torch.sign(edge_1_to_orthogonal) != torch.sign(edge_2_to_orthogonal):
                # if they disagree we can take both
                candidates = [sorted_triangle_indices[[0,1]], sorted_triangle_indices[[0,2]]]
            else:
                # if they agree we only take the one with the smaller angle to orthogonal
                if torch.abs(edge_1_to_orthogonal) > torch.abs(edge_2_to_orthogonal):
                    candidates = [sorted_triangle_indices[[0,1]]]
                else:
                    candidates = [sorted_triangle_indices[[0,2]]]
            
            # filter out inner edges
            non_inner_edges = set()
            for i in range(3):
                if not is_inner_edge[i]:
                    non_inner_edges.add(tuple(sorted(triangle_vertex_ids[[i, (i + 1) % 3]].tolist())))

            non_inner_lower_edges = [edge for edge in candidates if tuple(sorted(edge.tolist())) in non_inner_edges]
            return non_inner_lower_edges

            

        is_inner_edge = self.get_inner_edge_mask_with_cache()
        lower_edges = []
        
        triangle_mask = self.triangle_polygons == polygon
        triangles = self.triangles[triangle_mask]
        is_inner_edge = is_inner_edge[triangle_mask]
        for i, triangle_vertex_ids in enumerate(triangles):
            lower_edges.extend(compute_is_lower_edge(triangle_vertex_ids, direction, orthogonal_direction, is_inner_edge[i]))

        if len(lower_edges) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=self.device)
        
        contour_edges = self.get_contour_edges_with_cache()
        assert torch.isin(torch.cat(lower_edges), contour_edges).all(), "Lower edges are also contour edges"
        lower_edges = torch.stack(lower_edges) 

        if plot:
            points = []
            lines = []
            for edge in lower_edges:
                v1, v2 = vertices[edge].detach().numpy()
                points.append(v1)
                points.append(v2)
                lines.append([len(points) - 2, len(points) - 1])


            lines = np.stack(lines)
            points = np.stack(points)

            import open3d as o3d
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            mesh = self.export_triangle_mesh()
            o3d.visualization.draw_geometries([mesh, line_set])

        return lower_edges
        


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.update({
            'triangles': self.triangles,
            'triangle_polygons': self.triangle_polygons,
            'polygon_colors': self.polygon_colors,
            'polygon_classes': self.polygon_classes,
            'vertices.vertex_plane_assignments': self.vertices.vertex_plane_assignments,
            'class_names': self.class_names,
            'config': self.config
        })
        return state


    def load_state_dict(self, state_dict, strict=True):
        self.triangles = state_dict.pop('triangles', torch.zeros((0, 3), dtype=torch.long)).to(self.device)
        self.triangle_polygons = state_dict.pop('triangle_polygons', torch.zeros(0, dtype=torch.long)).to(self.device)
        self.polygon_colors = state_dict.pop('polygon_colors', torch.zeros(0, 3)).to(self.device)
        self.polygon_classes = state_dict.pop('polygon_classes', torch.zeros(0, dtype=torch.long)).to(self.device)
        self.vertices.vertex_plane_assignments = state_dict.pop('vertices.vertex_plane_assignments', torch.zeros(0, 3)).to(self.device)
        self.class_names = state_dict.pop('class_names', [])

        self.vertices.vertices = nn.Parameter(state_dict.pop('vertices.vertices', torch.zeros(0, 3)).to(self.device))
        self.vertices.planes = nn.Parameter(state_dict.pop('vertices.planes', torch.zeros(0, 4)).to(self.device))
        
        self.config = state_dict.pop('config', None)
        # super().load_state_dict(state_dict, strict=strict)
        
        
