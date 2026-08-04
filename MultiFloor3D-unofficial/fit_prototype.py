import json
import os
import pickle
import random
from typing import Dict, List, Optional
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
from mesh_fitting_3D.geometry_utils import compute_distance_to_mesh, compute_closest_triangle_to_points_o3d, compute_ray_mesh_intersections_ray_tracing
from mesh_fitting_3D.polygon_fitting_config import PolygonFittingConfig, ScannetppConfig, MatterportConfig
from mesh_fitting_3D.point_triangle_distance_vectorized import find_k_closest_triangles
try:
    import pytorch3d
except ImportError:
    print("Pytorch3d not installed. Please install it by running 'pip install pytorch3d'")

import time 

import torch

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import lstsq
from tqdm import tqdm


from differentiable_3D_polygon_stuctures import PolygonSet3D, intersection_line_between_planes, project_points_3d_to_3d_plane, project_points_to_lines_torch
from differentiable_mesh_sampling import sample_points_from_mesh
from chamfer_distance import chamfer_distance_indices, mutual_nearest_neighbors
from merge_split_util import _point_edge_sq_distance_along_last_dimension, get_closest_face_per_point, get_closest_edges_to_point_sq_dist, point_mesh_face_distance, point_to_face_distance, point_to_triangle_edges_sq_dist, point_to_mesh_distance



def split_into_windows(points: List[torch.Tensor], thresh_xs: List, thresh_ys : List, thresh_zs : List):
    """Splits the points into eight windows. The split is performed at thresh_x, thresh_y, thresh_z"""
    windows = []
    ixes = []
    for i in range(len(thresh_zs) + 1):
        left_bound = -float("inf") if i == 0 else thresh_zs[i - 1]
        right_bound = float("inf") if i == len(thresh_zs) else thresh_zs[i]
        z_mask = (points[:, 2] >= left_bound) & (points[:, 2] < right_bound)
        for j in range(len(thresh_ys) + 1):
            left_bound = -float("inf") if j == 0 else thresh_ys[j - 1]
            right_bound = float("inf") if j == len(thresh_ys) else thresh_ys[j]
            y_mask = (points[:, 1] >= left_bound) & (points[:, 1] < right_bound)
            for k in range(len(thresh_xs) + 1):
                left_bound = -float("inf") if k == 0 else thresh_xs[k - 1]
                right_bound = float("inf") if k == len(thresh_xs) else thresh_xs[k]
                x_mask = (points[:, 0] >= left_bound) & (points[:, 0] < right_bound)
                mask = x_mask & y_mask & z_mask
                windows.append(points[mask])
                ixes.append(torch.where(mask)[0])
    return windows, ixes

def split_into_eight_windows(points: List[torch.Tensor], thresh_x, thresh_y, thresh_z):
    """Splits the points into eight windows. The split is performed at thresh_x, thresh_y, thresh_z"""
    x_mask = points[:, 0] < thresh_x
    y_mask = points[:, 1] < thresh_y
    z_mask = points[:, 2] < thresh_z
    windows = []
    ixes = []
    for l1 in [True, False]:
        for l2 in [True, False]:
            for l3 in [True, False]:
                mask = x_mask if l1 else ~x_mask
                mask = mask & (y_mask if l2 else ~y_mask)
                mask = mask & (z_mask if l3 else ~z_mask)
                windows.append(points[mask])
                ixes.append(torch.where(mask)[0])
    return windows, ixes


def windowed_mutual_nearest_neighbors(src_points, target_points, window_split_x, window_split_y, window_split_z):
        device = src_points.device

        windowed_src, windowed_src_ixes = split_into_windows(src_points, window_split_x, window_split_y, window_split_z)
        windowed_targets, windowed_target_ixes = split_into_windows(target_points, window_split_x, window_split_y, window_split_z)
        assert sum(map(len, windowed_src)) == len(src_points)
        assert sum(map(len, windowed_targets)) == len(target_points)


        # pad the points and targets to max length
        point_lengths = torch.tensor([len(p) for p in windowed_src]).to(device)
        max_len = point_lengths.max()
        windowed_src = torch.stack([torch.cat((p, torch.zeros(max_len - len(p), 3).to(p.device))) for p in windowed_src])
        windowed_src_ixes = torch.stack([torch.cat((ix, torch.zeros(max_len - len(ix), dtype=torch.long).to(ix.device))) for ix in windowed_src_ixes])
        point_mask = torch.arange(max_len, device=device) < point_lengths[:, None]
        # assert len(windowed_src[point_mask]) == len(src_points)
        

        target_lengths = torch.tensor([len(p) for p in windowed_targets]).to(device)
        max_len = target_lengths.max()
        windowed_targets = torch.stack([torch.cat((p, torch.zeros(max_len - len(p), 3).to(p.device))) for p in windowed_targets])
        windowed_target_ixes = torch.stack([torch.cat((ix, torch.zeros(max_len - len(ix), dtype=torch.long).to(ix.device))) for ix in windowed_target_ixes])
        target_mask = torch.arange(max_len, device=device) < target_lengths[:, None]
        assert len(windowed_targets[target_mask]) == len(target_points)

        closest_src_rel, closest_trg_rel = mutual_nearest_neighbors(windowed_src, windowed_targets, point_lengths, target_lengths)
        
        # remap the relative indices to the original indices
        closest_trg = torch.zeros_like(closest_trg_rel)
        closest_src = torch.zeros_like(closest_src_rel)
        for i in range(len(windowed_src_ixes)):
            closest_trg[i] = windowed_src_ixes[i][closest_trg_rel[i]]
            closest_src[i] = windowed_target_ixes[i][closest_src_rel[i]]

        closest_src = closest_src.squeeze(-1)[point_mask].squeeze(-1)
        closest_trg = closest_trg.squeeze(-1)[target_mask].squeeze(-1)
        windowed_src_ixes = windowed_src_ixes[point_mask].squeeze(-1)
        windowed_target_ixes = windowed_target_ixes[target_mask].squeeze(-1)

        # sort back into order
        closest_src = closest_src[windowed_src_ixes]
        closest_trg = closest_trg[windowed_target_ixes]

        return closest_trg, closest_src



def visualize_polygon_info(mesh: o3d.geometry.TriangleMesh, polygon_info: dict):

    colors = np.ones((len(mesh.vertices), 3)) * 0.5 # vertices are gray by default
    colors[polygon_info["vertices"]] = [0, 1, 0] # inliers are green
    # # color code: concave shared edges red, concave contour orange, convex contour light blue, convex shared edges dark blue
    # colors[polygon_info["contours"][0]] = [1, 0, 0] #  contour vertices are red
    colors[polygon_info["convex_edges"]] = [0, 0, 1] # convex contour vertices are blue
    # shared_edges = polygon_info.get("shared_edges", {})
    # for edge in shared_edges.values():
    #     colors[edge] = [1, 0.5, 0] # concave shared edges are orange
    #     convex_vertices = set(polygon_info["convex_edges"])
    #     concave_vertices = {e for e in edge if e not in convex_vertices}
    #     convex_vertices = convex_vertices.intersection(edge)
    #     if len(convex_vertices) > 0:
    #         colors[list(convex_vertices)] = [1, 1, 0] # convex shared edges are yellow
    #     if len(concave_vertices) > 0:
    #         colors[list(concave_vertices)] = [1, 0, 1] # concave shared edges are purple

    is_colored = (colors != 0.5).any(axis=1)

    points = np.array(mesh.vertices)[is_colored]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[is_colored])
    o3d.visualization.draw_geometries([mesh, pcd])



def compute_signed_distances_to_3D_plane(points, plane_eq):
    """Compute the signed distances of the points to the plane. Generated by Copilot."""
    [a, b, c, d] = plane_eq
    plane_origin = np.array([0, 0, -d/(c + 1e-10)], dtype=np.float32)
    plane_normal = np.array([a, b, c], dtype=np.float32)
    vec = points - plane_origin
    return np.dot(vec, plane_normal)



def compute_contour_convexity(mesh: o3d.geometry.TriangleMesh, polygon_info : Dict, thresh=0.05) -> dict:
    """Computes for every polygon the inliers, the 3D contour, its convex and concave parts, its adjacency to other instances, the vertices it shares with other instances, as well as 'corners' (vertices it shares with more than one other plane)"""
    mesh = o3d.geometry.TriangleMesh(mesh)
    polygon_info = deepcopy(polygon_info)


    mesh.compute_adjacency_list()
    vertices = np.array(mesh.vertices)
    global_adjacency_list = np.array(mesh.adjacency_list, dtype=object)

    for k, polygon in polygon_info.items():
        if not (np.array(polygon["plane_eq"]) == 0).all():     
            convex_contour_vertices = set()
            for contour_vertices in polygon["contours"]:
                # todo: we could do this based on plane equations of neighbors
                for vertex in contour_vertices:
                    neighbors = np.array(list(global_adjacency_list[vertex]))
                    neighbor_vertices = vertices[neighbors]
                    distances = compute_signed_distances_to_3D_plane(neighbor_vertices, polygon["plane_eq"])
                    if any(distances < -(thresh ** 2)):
                        convex_contour_vertices.add(vertex)
            
            polygon_info[k]["convex_edges"] = list(convex_contour_vertices)

    return polygon_info


# def approximate_polygon_rdp(points, epsilon=0.0005, thresh=0.001):
#     projected_contour_int =(points/ thresh).astype(int)
#     peri = cv2.arcLength(projected_contour_int, True)
#     return cv2.approxPolyDP(projected_contour_int, epsilon * peri, True)[:,0,:] * thresh

def filter_polygons_class_based(polygon_info: Dict, keep_classes: List[str]) -> Dict:
    """Filters the polygons based on their class."""
    polygon_info = deepcopy(polygon_info)
    for i, polygon in polygon_info.items():
        polygon["shared_edges"] = {k : v for k, v in polygon["shared_edges"].items() if k in polygon_info and polygon_info[k]["class"] in keep_classes}

    polygon_info = {k : polygon_info[k] for k in polygon_info if polygon_info[k]["class"] in keep_classes}
    return polygon_info


def delete_small_holes(mesh : o3d.geometry.TriangleMesh, polygon_info: Dict, min_area: float) -> Dict:
    """Deletes small holes from the polygon info."""
    polygon_info = deepcopy(polygon_info)

    contour_instances = {v : set() for v in range(len(mesh.vertices))}
    for k, polygon in polygon_info.items():
        for contour in polygon["contours"]:
            for v in contour:
                contour_instances[v].add(k)

    for k, polygon in polygon_info.items():
        new_contours = [polygon["contours"][0]]
        new_contour_areas = [polygon["contour_areas_original"][0]]
        new_contour_masks_rdp = [polygon["contour_masks_rdp"][0]]
        for i, contour in enumerate(polygon["contours"][1:], 1):
            n_adjacent_polygons = len({p for v in contour for p in contour_instances[v] if p != k})
            if polygon["contour_areas_original"][i] < min_area and n_adjacent_polygons == 0:
                continue
            new_contours.append(contour)
            new_contour_areas.append(polygon["contour_areas_original"][i])
            new_contour_masks_rdp.append(polygon["contour_masks_rdp"][i])

        polygon["contours"] = new_contours
        polygon["contour_areas_original"] = new_contour_areas
        polygon["contour_masks_rdp"] = new_contour_masks_rdp
                
    return polygon_info


def delete_holes_with_no_convex_contours(mesh : o3d.geometry.TriangleMesh, polygon_info: Dict, min_n_convex_edge_vertices=5) -> Dict:
    """Delete all holes that do not have any convex vertex."""
    polygon_info = deepcopy(polygon_info)
    
    contour_instances = {v : set() for v in range(len(mesh.vertices))}
    for k, polygon in polygon_info.items():
        for contour in polygon["contours"]:
            for v in contour:
                contour_instances[v].add(k)

    for k, polygon in polygon_info.items():
        new_contours = [polygon["contours"][0]]
        new_contour_areas = [polygon["contour_areas_original"][0]]
        new_contour_masks_rdp = [polygon["contour_masks_rdp"][0]]
        for i, contour in enumerate(polygon["contours"][1:], 1):
            n_adjacent_polygons = len({p for v in contour for p in contour_instances[v] if p != k})
            if n_adjacent_polygons == 0 and len(set(contour).intersection(polygon["convex_edges"])) < min_n_convex_edge_vertices:
                continue
            new_contours.append(contour)
            new_contour_areas.append(polygon["contour_areas_original"][i])
            new_contour_masks_rdp.append(polygon["contour_masks_rdp"][i])
    return polygon_info


def transform_mesh(mesh: o3d.geometry.TriangleMesh, nerfstudio_transforms_file : str) -> o3d.geometry.TriangleMesh:
    """Transforms the mesh with the given nerf studio transforms file."""

    if not os.path.exists(nerfstudio_transforms_file):
        print(f"File {nerfstudio_transforms_file} does not exist. Skipping transformation.")
        return None
    with open(nerfstudio_transforms_file, 'r') as f:
        transforms = json.load(f)
    
    transform_matrix = np.array(transforms['transform'])
    scale = transforms['scale']
    
    print(f"Transform matrix: {transform_matrix}")
    print(f"Scale: {scale}")
    # Transform the mesh and save the transformed version
    vertices = np.asarray(mesh.vertices).copy()
    vertices /= scale

    inv_transform_matrix = np.linalg.inv(
        np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
    )
    vertices = np.dot(inv_transform_matrix[:3, :3], vertices.T).T + inv_transform_matrix[:3, 3]
    copied_mesh = o3d.geometry.TriangleMesh(mesh)
    copied_mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(vertices.astype(np.float64)))
    return copied_mesh


def truncated_chamfer_distance(src, trg, src_norm, target_norm, max_dist=0.1, max_angle=45, include_normals=True, include_src_to_trg=True):
    x_lengths = torch.tensor([len(src)]).to(device)
    y_lengths = torch.tensor([len(trg)]).to(device)
    if include_normals:
        closest_src, closest_trg = mutual_nearest_neighbors(torch.cat([src, src_norm], dim=1).unsqueeze(0), torch.cat([trg, target_norm], dim=1).unsqueeze(0), x_lengths, y_lengths)
    else:
        closest_src, closest_trg = mutual_nearest_neighbors(src.unsqueeze(0), trg.unsqueeze(0), x_lengths, y_lengths)
    closest_src = closest_src.view(-1)
    closest_trg = closest_trg.view(-1)

    normal_mask = (src[closest_trg] * trg).sum(dim=-1) > np.cos(np.deg2rad(max_angle))
    max_distance_mask = (src[closest_trg] - trg).norm(dim=-1) < max_dist
    mask = normal_mask & max_distance_mask
    # mask = torch.ones_like(closest_trg, dtype=torch.bool)
    dist_to_closest_src = (src[closest_trg] - trg).norm(dim=-1)
    loss_to_closest_src = (dist_to_closest_src)
    # normal_loss += (src_norm[closest_trg] - target_norm).norm(dim=-1)[mask].mean()

    normal_mask = (trg[closest_src] * src).sum(dim=-1) > np.cos(np.deg2rad(max_angle))
    max_distance_mask = (trg[closest_src] - src).norm(dim=-1) < max_dist
    mask = normal_mask & max_distance_mask
    mask = torch.ones_like(closest_src, dtype=torch.bool)
    dist_to_closest_trg = (trg[closest_src] - src).norm(dim=-1)
    loss_to_closest_trg = (dist_to_closest_trg)[mask]

    return loss_to_closest_src, loss_to_closest_trg
    # if not include_src_to_trg:
    #     return loss_to_closest_trg
    # else:
    #     return loss_to_closest_src + loss_to_closest_trg


def ray_tracing_distance_loss(vertices: torch.Tensor, triangles: torch.Tensor, triangle_plane_eqs : torch.Tensor, cover_points: torch.Tensor, ray_dests: torch.Tensor, ray_origins: torch.Tensor, max_dist=0.1, max_angle=45, include_normals=True, margin=0.1, is_inner_edge_mask=None, plot=False, polygon_areas_per_triangle : torch.Tensor = None):
    loss = 0

    t1 = time.time()
    
    does_intersect, triangle_ids, intersection_points, len_of_overlap = compute_ray_mesh_intersections_ray_tracing(vertices, triangles, ray_origins, ray_dests, margin=margin)

    t2 = time.time()
    # the triangles should not obstruct the view
    # the ones that intersect should be moved to the border of the triangle it intersects with
    intersected_triangles = triangles[triangle_ids[does_intersect]]
    assert intersected_triangles.shape == (does_intersect.sum(), 3)
    intersected_triangle_vertices = vertices[intersected_triangles]
    assert intersected_triangle_vertices.shape == (does_intersect.sum(), 3, 3)
    intersection_points = intersection_points[does_intersect]

    if len(intersection_points) > 0:
        # distance_to_edges_of_triangle = _point_edge_sq_distance_along_last_dimension(intersection_points, edges)# compute min distance to all three edges of the relevant triangle
        distance_to_edges_of_triangle = point_to_triangle_edges_sq_dist(intersection_points, intersected_triangle_vertices)
        assert distance_to_edges_of_triangle.shape == (len(intersection_points), 3)
        if is_inner_edge_mask is not None:
            # compute distance to closest border, not closest inner edge
            intersected_triangle_is_inner_edge_mask = is_inner_edge_mask[triangle_ids[does_intersect]]
            distance_to_edges_of_triangle[intersected_triangle_is_inner_edge_mask] = (distance_to_edges_of_triangle.max() + 1).detach()
        dist_to_border = torch.min(distance_to_edges_of_triangle, dim=-1)[0]
        # crop outliers
        close_to_border_mask = dist_to_border < margin

        # we should move the closest border towards the intersection point
        loss += dist_to_border[close_to_border_mask].sum() / len(ray_dests)

        # we should move the plane towards the target for those that are not close to the border
        intersected_plane_eqs = triangle_plane_eqs[triangle_ids[does_intersect]]
        targets_of_intersection_points = ray_dests[does_intersect]
        _, inlier_distances_to_planes = project_points_3d_to_3d_plane(targets_of_intersection_points, intersected_plane_eqs)
        not_too_far_mask = inlier_distances_to_planes < max_dist
        if polygon_areas_per_triangle is not None:
            inlier_distances_to_planes /= 1e6 * (1e-3 + polygon_areas_per_triangle[triangle_ids[does_intersect]])
        else:
            inlier_distances_to_planes /= len(ray_dests)
        
        loss += 10 * inlier_distances_to_planes[(~close_to_border_mask)].sum()

    # far_intersection_points = intersection_points[~close_to_border_mask]
    # pytorch3d_meshes = pytorch3d.structures.Meshes(verts=vertices.unsqueeze(0), faces=triangles.unsqueeze(0))
    # pytorch3d_pcds = pytorch3d.structures.Pointclouds(points=far_intersection_points.unsqueeze(0))
    # loss -= point_to_mesh_distance(pytorch3d_meshes, pytorch3d_pcds).sum() / len(cover_points)


    if plot: 
        # plot with o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())

        mesh_colors = np.ones((len(vertices), 3)) * 0.5
        mesh_colors[intersected_triangles] = [1, 0, 0]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
       
        # red for where the ray intersects
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(intersection_points.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(intersection_points)))

        rays_that_intersect = torch.cat([ray_origins[does_intersect], ray_dests[does_intersect]], dim=0)
        line_set_o3d = o3d.geometry.LineSet()
        line_set_o3d.points = o3d.utility.Vector3dVector(rays_that_intersect.detach().cpu().numpy())
        line_set_o3d.lines = o3d.utility.Vector2iVector(np.stack([np.arange(len(rays_that_intersect)//2), np.arange(len(rays_that_intersect)//2) + len(rays_that_intersect)//2], axis=0).T)

        # green for where the ray should end
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(ray_dests[does_intersect].detach().cpu().numpy())
        target_pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]] * len(ray_dests[does_intersect])))
        o3d.visualization.draw_geometries([mesh, pcd, target_pcd, line_set_o3d])

        # create a blue pcd from the argmax of the intersected triangle vertices
        points = []
        lines = []
        for dist, triangle_vertices in zip(distance_to_edges_of_triangle, intersected_triangle_vertices):
            argmin = dist.argmin()
            points.append(triangle_vertices[argmin])
            points.append(triangle_vertices[(argmin + 1) % 3])
            lines.append([len(points) - 2, len(points) - 1])

        pcd_2 = o3d.geometry.LineSet()
        pcd_2.points = o3d.utility.Vector3dVector(torch.stack(points).detach().cpu().numpy())
        pcd_2.lines = o3d.utility.Vector2iVector(lines)
        
        o3d.visualization.draw_geometries([mesh, pcd, target_pcd, pcd_2])
        
    t3 = time.time()

    # all points must be covered: assign to closest triangle
    # closest_triangle_ids, is_inside = compute_closest_triangle_to_points_o3d(vertices, triangles, cover_points)

    t35 = time.time()

    # CASE 1: It's inside the triangle
    # # the plane should fit the inliers. The polygon vertices are not affected
    # inside_triangle_ids = closest_triangle_ids[is_inside]
    # inside_plane_eqs = triangle_plane_eqs[inside_triangle_ids]
    # inside_points = cover_points[is_inside]
    # _, inlier_distances_to_planes = project_points_3d_to_3d_plane(inside_points, inside_plane_eqs)
    # # crop outliers
    # inlier_distances_to_planes = inlier_distances_to_planes[inlier_distances_to_planes < 0.05]

    # # we should move the plane towards the point
    # loss += 100 * inlier_distances_to_planes.sum() / len(cover_points)
    
    if plot:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())

        random_point_colors = np.random.rand(len(inside_points), 3)
        mesh_colors = np.ones((len(vertices), 3)) * 0.5

        # choose a subset of inlier points: mark the assigned triangles with red as well as the points
        randix = torch.randperm(len(inside_points))[:10]
        for triangle_id, rand_color in zip(inside_triangle_ids[randix], random_point_colors[randix]):
            mesh_colors[triangles[triangle_id]] = rand_color
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inside_points[randix].detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(random_point_colors[randix])
        o3d.visualization.draw_geometries([mesh, pcd])

    # line_set_o3d = o3d.geometry.LineSet()
    # line_set_o3d.points = o3d.utility.Vector3dVector(torch.cat([cover_points, closest_points.squeeze(1)], dim=0).detach().cpu().numpy())
    # line_set_o3d.lines = o3d.utility.Vector2iVector(np.stack([np.arange(len(cover_points)), np.arange(len(cover_points)) + len(cover_points)], axis=0).T)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
    # mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
    # # o3d.visualization.draw_geometries([mesh, line_set_o3d])
    
    # out_dir = "/mnt/usb_ssd/bieriv/layout-estimation-outputs/01-20-dslr-mesh-fitting/scannet++/f2dc06b1d2/all-classes/fitted_mesh_XXX.ply"
    # o3d.io.write_line_set(out_dir, line_set_o3d)
    # o3d.io.write_triangle_mesh(out_dir.replace("XXX", "mesh"), mesh)


    # t4 = time.time()
    pytorch3d_meshes = pytorch3d.structures.Meshes(verts=vertices.unsqueeze(0), faces=triangles.unsqueeze(0))
    pytorch3d_pcds = pytorch3d.structures.Pointclouds(points=cover_points.unsqueeze(0))
    point_to_mesh_dist = point_to_mesh_distance(pytorch3d_meshes, pytorch3d_pcds)
    mask = point_to_mesh_dist < max_dist
    loss += point_to_mesh_dist[mask].sum() / (1e-5 + mask.sum())
    # CASE 2: It's outside the triangle
    # the triangle should be moved closer to the point
    # outside_triangle_ids = closest_triangle_ids[~is_inside]
    # outside_triangle_vertices = vertices[triangles[outside_triangle_ids]]
    # cover_points_projected_to_surface = project_points_3d_to_3d_plane(cover_points[~is_inside], triangle_plane_eqs[outside_triangle_ids].detach())[0]

    # distances_of_projected = compute_distance_to_mesh(vertices, triangles, cover_points_projected_to_surface)

    # mask = (distances_of_projected > 1e-3) & (distances_of_projected < max_dist)
    # cover_points_projected_to_surface = cover_points_projected_to_surface[mask]
    # outside_triangle_vertices = outside_triangle_vertices[mask]

    # distance_to_edges_of_triangle = point_to_triangle_edges_sq_dist(cover_points_projected_to_surface, outside_triangle_vertices)
    # # distances_to_corners_of_triangle = ((outside_triangle_vertices - cover_points_projected_to_surface[:, None]) ** 2).sum(dim=-1)
    # # min_distance_to_edges = distances_to_corners_of_triangle.min(dim=-1)[0]
    # # assert (distance_to_edges_of_triangle.min(dim=-1)[0] <= distances_to_corners_of_triangle.min(dim=-1)[0] + 1e-5).all()
    
    # min_distance_to_edges = torch.min(distance_to_edges_of_triangle, dim=-1)[0]
    # # crop outliers
    # min_distance_to_edges = 5 * min_distance_to_edges[min_distance_to_edges < 0.05]

    # # we should move the closest border towards where the ray ends
    # loss += 50 * min_distance_to_edges.sum() / len(cover_points)

    # print(f"Number of points outside the triangle: {len(cover_points_projected_to_surface)}, Number of ray intersections: {does_intersect.sum()}, Number of points inside the triangle: {len(inside_points)}")

    t5 = time.time()
    # print(f"Time for ray tracing: {t2 - t1}, time for ray tracing loss {t3 - t2}, time for inside triangle loss {t4 - t3}, time for closest triangle to point {t35 - t3}, time for outside triangle loss {t5 - t4}, total time: {t5 - t1}")

    if plot:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cover_points_projected_to_surface.detach().cpu().numpy())
        o3d.visualization.draw_geometries([mesh, pcd])



    return loss, triangle_ids[does_intersect]


# def walls_must_reach_the_floor_loss(vertices: torch.Tensor, contour_edges: torch.Tensor, vertex_constraints : torch.Tensor, floor_eq : float, ceiling_eq : float, is_inner_edge_mask : torch.Tensor = None):
#     # must be near-orthogonal to up-axis
#     # up_vector = floor_eq[:3]
#     # up_vector /= torch.norm(up_vector)
#     # triangle_plane_eqs = triangle_plane_eqs.clone() / torch.norm(triangle_plane_eqs[:, :3], dim=-1, keepdim=True)
#     # triangle_angle_to_ups = torch.acos(torch.clamp((up_vector * triangle_plane_eqs[:, :3]).sum(dim=-1), -1 + 1e-6, 1 - 1e-6))
#     # triangle_angle_to_ups = torch.rad2deg(triangle_angle_to_ups)
#     # triangle_angle_to_vertical = torch.abs(triangle_angle_to_ups - 90)
    
#     contour_edge_vetices = vertices[contour_edges]
#     contour_edge_vectors = contour_edge_vetices[:, 1] - contour_edge_vetices[:, 0]
#     contour_edge_angle_to_up = torch.acos(torch.clamp((contour_edge_vectors * floor_eq[:3]).sum(dim=-1) / (contour_edge_vectors.norm(dim=-1) * floor_eq[:3].norm()), -1 + 1e-6, 1 - 1e-6))
#     contour_edge_angle_to_up = torch.rad2deg(contour_edge_angle_to_up)
#     mask = (contour_edge_angle_to_up < 10) | (contour_edge_angle_to_up > 170)
#     contour_edges = contour_edges[mask]
#     contour_edge_vertices = contour_edge_vetices[mask]

#     constrained_vertices = (vertex_constraints[contour_edges] != -1).sum(dim=-1) != 1
#     contour_edge_vertices[constrained_vertices] = float('inf')


#     _, vertex_dist_to_floor = project_points_3d_to_3d_plane(contour_edge_vertices, floor_eq.unsqueeze(0))
#     _, vertex_dist_to_ceiling = project_points_3d_to_3d_plane(contour_edge_vertices, ceiling_eq.unsqueeze(0))

#     return vertex_dist_to_floor.sum()
#     # min_dist_to_good = torch.min(vertex_dists_to_floor, vertex_dist_to_ceiling)
#     # the vertices should be moved to the floor
#     # return min_dist_to_good.sum() / len(triangle_vertices)


def project_objects_to_floor(polygon_set_3d : PolygonSet3D, floor_class_ix: int, object_vertices: torch.Tensor, object_triangles: torch.Tensor, floor_must_be_below: bool = False, up_vector: torch.Tensor = None):
    if floor_must_be_below:
        vertex_polygon_ixes = polygon_set_3d.find_next_lower_polygon_belonging_to_class(object_vertices, floor_class_ix, up_vector)
    else:
        vertex_polygon_ixes = polygon_set_3d.find_nearest_polygon_belonging_to_class(object_vertices, floor_class_ix)

    # show the floor class ix polygons
    # old_colors = polygon_set_3d.polygon_colors.clone()
    # polygon_colors = polygon_set_3d.polygon_colors.clone()
    # polygon_colors[vertex_polygon_ixes[0]] = torch.tensor([1.0, 0., 0.])
    # polygon_set_3d.polygon_colors = polygon_colors
    # o3d.visualization.draw_geometries([polygon_set_3d.export_triangle_mesh()])

    # only project the triangles that all belong to the same floor
    triangle_polygon_ixes = vertex_polygon_ixes[object_triangles]
    same_polygon_mask = triangle_polygon_ixes.max(dim=-1)[0] == triangle_polygon_ixes.min(dim=-1)[0]
    new_triangles = object_triangles[same_polygon_mask]
    new_triangle_vertices = object_vertices[new_triangles]
    new_triangle_polygons = triangle_polygon_ixes[same_polygon_mask, 0]

    polygon_set_3d.add_triangles_to_polygons(new_triangle_vertices, new_triangle_polygons)
    

def extend_walls_to_floor(polygon_set_3d : PolygonSet3D, wall_class_ix: int, floor_class_ix: int, up_vector: torch.Tensor, target_ray_origins: torch.Tensor, target_ray_dests: torch.Tensor, vertex_plane_assignments : torch.Tensor, max_intersects_per_area_per_polygon : float, must_be_below=False, plot=False, margin=0.1, max_angle_to_vertical=10):
    
    planes = polygon_set_3d.get_planes()

    for polygon in torch.where(polygon_set_3d.polygon_classes == floor_class_ix)[0]:
        angle_to_up = torch.acos(torch.clamp((planes[polygon, :3] * up_vector).sum(), -1 + 1e-6, 1 - 1e-6))
        angle_to_up = torch.rad2deg(angle_to_up)
        #zero out the floor/ceiling probability of near-vertical faces
        if angle_to_up > 30:
            polygon_set_3d.polygon_classes[polygon] = -1

    wall_polygons = torch.where(polygon_set_3d.polygon_classes == wall_class_ix)[0]
    for wall_polygon_id in wall_polygons:
        # check that angle to up is close to 90 deg
        angle_to_up = torch.acos(torch.clamp((planes[wall_polygon_id, :3] * up_vector).sum(), -1 + 1e-6, 1 - 1e-6))
        angle_to_up = torch.rad2deg(angle_to_up)
        if angle_to_up < 90 - max_angle_to_vertical or angle_to_up > 90 + max_angle_to_vertical:
            print(f"Wall polygon {wall_polygon_id} is not vertical enough: {angle_to_up}")
            # polygon_set_3d.polygon_classes[wall_polygon_id] = -1
            continue
        lower_edges = polygon_set_3d.get_polygon_edges_facing_direction(direction=up_vector, polygon=wall_polygon_id)
        
        n_lower = len(lower_edges)
        
        # only take near-horizontal edges
        vertices = polygon_set_3d.get_vertices()
        edge_vectors = vertices[lower_edges[:, 1]] - vertices[lower_edges[:, 0]]
        edge_vectors /= edge_vectors.norm(dim=-1, keepdim=True)
        edge_angles = torch.acos(torch.clamp((edge_vectors * up_vector).sum(dim=-1), -1 + 1e-6, 1 - 1e-6))
        edge_angles = torch.rad2deg(edge_angles)
        edge_len = edge_vectors.norm(dim=-1)
        mask = (edge_len > 0.2)
        mask &= torch.min(edge_angles, 180 - edge_angles) > max_angle_to_vertical # don't take near-vertical edges

        n_angle_and_orientation = mask.sum()
        
        lower_edges_filtered = lower_edges[mask]

        # find the closest floor plane

        first_is_lower = (vertices[lower_edges_filtered[:, 0]] @ up_vector[:, None]).sum(dim=1) < (vertices[lower_edges_filtered[:, 1]] @ up_vector[:, None]).sum(dim=1)
        edge_lower_point = torch.where(first_is_lower[:, None], vertices[lower_edges_filtered[:, 0]], vertices[lower_edges_filtered[:, 1]])
        if must_be_below:
            closest_floor_plane = polygon_set_3d.find_next_lower_polygon_belonging_to_class(edge_lower_point, floor_class_ix, up_vector, max_dist=3)
        else:
            closest_floor_plane = polygon_set_3d.find_nearest_polygon_belonging_to_class(edge_lower_point, floor_class_ix, max_dist=3)
        # mesh = polygon_set_3d.export_triangle_mesh()
        # new_vertices = []
        # new_triangles = []
        n_no_constraints_match = 0
        new_triangle_vertices = []
        for lower_edge, floor_polygon in zip(lower_edges_filtered, closest_floor_plane):
            if floor_polygon == -1:
                continue
            edge_constraints = vertex_plane_assignments[lower_edge]
            two_constraints_match = ((edge_constraints[0, None, :] == edge_constraints[1, :, None]) & (edge_constraints[0, None, :] != -1) & (edge_constraints[1, :, None] != -1)).sum(dim=-1).sum(dim=-1) >= 2
            if two_constraints_match:
                continue
            n_no_constraints_match += 1
            # try to extend the wall to the floor
            intersection_line = intersection_line_between_planes(planes[wall_polygon_id], planes[floor_polygon])
            added_vertices, added_triangles, added_area, n_intersections = polygon_set_3d.try_extend_edge_to_line(vertices[lower_edge], torch.cat(intersection_line), ray_origins=target_ray_origins, ray_dests=target_ray_dests, ray_intersection_margin=margin)
            max_intersects = max_intersects_per_area_per_polygon.get(wall_polygon_id.item(), 0)
            if n_intersections / added_area < max_intersects:
                # new_triangles.extend(added_triangles + len(new_vertices))
                # new_vertices.extend(added_vertices)
                new_triangle_vertices.append(added_vertices[added_triangles])

        if plot:
            try:
                # color the current polygon red
                original_colors = polygon_set_3d.polygon_colors.clone()
                new_colors = original_colors.clone()
                new_colors[wall_polygon_id] = torch.tensor([1.0, 0., 0.]).cuda()
                new_colors[torch.unique(closest_floor_plane)] = torch.rand((len(torch.unique(closest_floor_plane)), 3)).cuda()

                # for each edge midpoint: color the edge midpoint and the closest floor with the same color
                edge_midpoints = []
                midpoint_colors = []
                for lower_edge, floor_polygon in zip(lower_edges_filtered, closest_floor_plane):
                    if floor_polygon == -1:
                        continue
                    edge_midpoints.append((vertices[lower_edge[0]] + vertices[lower_edge[1]]) / 2)
                    midpoint_colors.append(new_colors[floor_polygon])
                
                polygon_set_3d.polygon_colors = new_colors
                out_dir = f"/mnt/usb_ssd/bieriv/tmp/debug/{wall_polygon_id}"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/mesh.ply", mesh)
                polygon_set_3d.polygon_colors = original_colors
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(torch.stack(edge_midpoints).detach().cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(torch.stack(midpoint_colors).detach().cpu().numpy())
                o3d.io.write_point_cloud(f"{out_dir}/midpoints.ply", pcd)
            except Exception as e:
                print(f"Error: {e}")
                pass

            if len(new_triangle_vertices) > 0:
                # also show the added triangles
                new_triangle_verts= np.concatenate([v.detach().cpu().numpy() for v in new_triangle_vertices]).reshape(-1, 3)
                mesh_new = o3d.geometry.TriangleMesh()
                mesh_new.vertices = o3d.utility.Vector3dVector(new_triangle_verts)
                mesh_new.triangles = o3d.utility.Vector3iVector(np.arange(len(new_triangle_verts)).reshape(-1, 3).astype(int))
                o3d.io.write_triangle_mesh(f"{out_dir}/new_triangles.ply", mesh_new)

            


        
        print(f"Wall polygon {wall_polygon_id} has {len(new_triangle_vertices) / 2} / {len(lower_edges) * 2} new triangles (LOWER: {n_lower}, ANGLE AND ORIENTATION: {n_angle_and_orientation}, NO CONSTRAINTS MATCH: {n_no_constraints_match})")
        if len(new_triangle_vertices) > 0:
            new_triangle_vertices = torch.cat(new_triangle_vertices)
            polygon_set_3d.add_triangles_to_polygons(new_triangle_vertices, torch.tensor([wall_polygon_id] * len(new_triangle_vertices)))
        



def fit_polygon_collection(config : PolygonFittingConfig, rectified_mesh: o3d.geometry.TriangleMesh, target_pcd: o3d.geometry.PointCloud, polygon_info : Dict, out_dir: str,device : str, nerfstudio_transforms_file: Optional[str] = None, target_pcd_ray_origins: Optional[np.ndarray] = None, target_ray_dests: Optional[np.ndarray] = None, target_vertex_classes: Optional[np.ndarray] = None, target_vertex_class_names: Optional[np.ndarray] = None, object_mesh: Optional[o3d.geometry.TriangleMesh] = None) -> None:
    """Fits the polygon collection to the rectified mesh."""

    print(f"Number of polygon contours initially: {sum(map(len, [polygon['contours'] for polygon in polygon_info.values()]))}")

    # keep_classes: List[str] = ["wall", "ceiling", "floor", "door", "window", "curtain", "blind", "door way", "column", "stairs"]
    # if filter_classes:
    #     polygon_info = filter_polygons_class_based(polygon_info, keep_classes)
    
    # print(f"Number of polygon contours after filtering: {sum(map(len, [polygon['contours'] for polygon in polygon_info.values()]))}")
    # # polygon_info = delete_small_holes(rectified_mesh, polygon_info, 0.1)

    # print(f"Number of polygons after hole deletion: {sum(map(len, [polygon['contours'] for polygon in polygon_info.values()]))}")
    # # polygon_info = compute_contour_convexity(rectified_mesh, polygon_info)
    # # polygon_info = delete_holes_with_no_convex_contours(rectified_mesh, polygon_info)
    # print(f"Number of polygons after non-convex hole deletion: {sum(map(len, [polygon['contours'] for polygon in polygon_info.values()]))}")
    # polygon_info = assign_features_to_polygons(rectified_mesh, polygon_info, openseg_features)
    # polygon_info = classify_polygons(polygon_info)


    
    

    # for k in list(polygon_info.keys())[::100]:
    #     visualize_polygon_info(rectified_mesh, polygon_info[k])
    
    # for k in polygon_info:
    #     polygon_info[k]["id"] = k

    # # convert to sorted list
    polygon_info = sorted(polygon_info.values(), key=lambda x: len(x["vertices"]), reverse=True)
    points3D = torch.from_numpy(np.array(rectified_mesh.vertices)).float()
    # point_inlier_assigmnent = [set() for _ in range(len(rectified_mesh.vertices))]
    # point_outlier_assignment = [set() for _ in range(len(rectified_mesh.vertices))]
    # point_border_assignment = [set() for _ in range(len(rectified_mesh.vertices))]
    # for i, polygon in enumerate(polygon_info):
    #     # if i == 0:
    #         # # plot the first polygon
    #         # polygon_vertices_3d =  np.array(rectified_mesh.vertices)[polygon["convex_edges"]]
    #         # inliers_3d = np.array(rectified_mesh.vertices)[polygon["vertices"]]
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(111, projection='3d')
    #         # ax.scatter(polygon_vertices_3d[:, 0], polygon_vertices_3d[:, 1], polygon_vertices_3d[:, 2], c='r', marker='o')
    #         # ax.scatter(inliers_3d[:, 0], inliers_3d[:, 1], inliers_3d[:, 2], c='g', marker='o', s=0.1)
    #         # ax.set_aspect('equal')
    #         # ax.set_xlabel('X Label')
    #         # ax.set_ylabel('Y Label')
    #         # ax.set_zlabel('Z Label')
    #         # plt.show()
    #     for v in polygon["vertices"]:
    #         point_inlier_assigmnent[v].add(i)
    #     for v in polygon["convex_edges"]:
    #         point_border_assignment[v].add(i)

    # max_n_inlier_assignments = max(len(a) for a in point_inlier_assigmnent)
    # max_n_outlier_assignments = max(len(a) for a in point_outlier_assignment)
    # max_n_border_assignments = max(len(a) for a in point_border_assignment)

    # create padded arrays
    # inlier_assignmens = torch.Tensor([list(a) + [-1] * (max_n_inlier_assignments - len(a)) for a in point_inlier_assigmnent]).to(int).to(device)
    # outlier_assignmens = torch.Tensor([list(a) + [-1] * (max_n_outlier_assignments - len(a)) for a in point_outlier_assignment]).to(int).to(device)
    # border_assignments = torch.Tensor([list(a) + [-1] * (max_n_border_assignments - len(a)) for a in point_border_assignment]).to(int).to(device)
    points3D = points3D.to(device)

    target_coords = torch.from_numpy(np.array(target_pcd.points)).float().to(device)
    target_normals = torch.from_numpy(np.array(target_pcd.normals)).float().to(device)
    target_colors = torch.from_numpy(np.array(target_pcd.colors)).float().to(device)

    target_ray_origins = torch.from_numpy(target_pcd_ray_origins).float().to(device)
    target_ray_dests = torch.from_numpy(target_ray_dests).float().to(device)
    assert target_ray_origins.shape == target_ray_dests.shape

    print(f"Number of polygons: {len(polygon_info)}")
    print(f"Number of vertices: {len(points3D)}")
    print(f"Number of target points: {len(target_coords)}")
    print(f"Number of rays: {len(target_ray_origins)}")
    
    ray_line_set = o3d.geometry.LineSet()
    ray_line_set.points = o3d.utility.Vector3dVector(np.concatenate([target_ray_origins.detach().cpu().numpy(), target_ray_dests.detach().cpu().numpy()], axis=0))
    ray_line_set.lines = o3d.utility.Vector2iVector(np.stack([np.arange(len(target_pcd_ray_origins)), np.arange(len(target_pcd_ray_origins)) + len(target_pcd_ray_origins)], axis=0).T)
    print(f"saved ray line set to {os.path.join(out_dir, 'ray_line_set.ply')}")
    o3d.io.write_line_set(os.path.join(out_dir, "ray_line_set.ply"), ray_line_set)

    # window_split_x = [torch.median(target_coords[:, 0])]
    # window_split_y = [torch.median(target_coords[:, 1])]
    # window_split_z = [torch.median(target_coords[:, 2])]
    # window_split_x = [torch.quantile(target_coords[:, 0], q) for q in [0.33, 0.66]]
    # window_split_y = [torch.quantile(target_coords[:, 1], q) for q in [0.33, 0.66]]
    # window_split_z = [torch.quantile(target_coords[:, 2], q) for q in [0.33, 0.66]]

    
    project_objects_to_floor_at_step = config.project_objects_to_floor_at_step
    simplification_steps = config.simplification_steps
    plane_merge_steps = config.plane_merge_steps
    # extend_walls_to_floor_steps = config.extend_walls_to_floor_steps
    extend_walls_to_floor_steps = [s + 1 for s in simplification_steps[::3] if s > project_objects_to_floor_at_step]
    recompute_interval = config.recompute_interval
    save_interval = config.save_interval
    iterations = config.iterations
    
    max_dist = config.max_dist
    magnetism_threshold = config.magnetism_threshold
    max_angle = config.max_angle
    vertex_merge_thresh = config.vertex_merge_thresh
    ray_intersection_margin = config.ray_intersection_margin

    warmup_steps = config.warmup_steps
    window = config.window
    chamfer_distance = config.chamfer_distance
    point_triangle_distance = config.point_triangle_distance
    ray_tracing_distance = config.ray_tracing_distance
    include_normals = config.include_normals

    merge_planes = config.merge_planes
    simplify = config.simplify
    do_extend_walls_to_floor = config.do_extend_walls_to_floor
    do_extend_walls_to_ceiling = config.do_extend_walls_to_ceiling
    regularize = config.regularize
    regularize_wall_reach_ceiling = config.regularize_wall_reach_ceiling
    regularize_non_watertight = config.regularize_non_watertight
    regularize_edge_magnetism = config.regularize_edge_magnetism
    regularize_vertex_magnetism = config.regularize_vertex_magnetism
    regularize_face_magnetism = config.regularize_face_magnetism
    regularize_area = config.regularize_area
    regularize_straightness = config.regularize_straightness
    regularize_rectangularity = config.regularize_rectangularity
    regularize_manhattan = config.regularize_manhattan    

    
    # project_objects_to_floor_at_step = 600
    # simplification_steps = [50, 100, 150, 225, 300, 375, 450, 525, 600, 675, 750, 825, 900, 975, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    # plane_merge_steps = [90, 200, 275, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2500, 3000, 3500, 4000, 4500, 5000]
    # extend_walls_to_floor_steps = [s + 1 for s in simplification_steps[::3] if s > project_objects_to_floor_at_step]
    # simplification_stets = []

    # mesh = polygon_set_3d.export_triangle_mesh()
    # o3d.visualization.draw_geometries([mesh])
    


    # warmup_steps = 50
    # window = True
    # chamfer_distance = False
    # point_triangle_distance = False
    # ray_tracing_distance = True
    # include_normals = False

    # merge_planes = True
    # simplify = True
    # do_extend_walls_to_floor = True
    # do_extend_walls_to_ceiling = True
    # regularize = True
    # regularize_wall_reach_ceiling = False
    # regularize_non_watertight = True
    # regularize_edge_magnetism = False
    # regularize_vertex_magnetism = True
    # regularize_face_magnetism = True
    # regularize_area = False
    # regularize_straightness = False
    # regularize_rectangularity = False
    # regularize_manhattan = True
    up_vector = torch.tensor([0, 1, 0] if config.up_vector.lower() == "y" else [0, 0, 1], device=device).float()

    if regularize_wall_reach_ceiling:
        # place floor plane at .99percentile of the target points
        floor_eq = torch.tensor([0, 1, 0, torch.quantile(target_coords[:, 1], 0.99)], device=device)
        ceiling_eq = torch.tensor([0, 1, 0, torch.quantile(target_coords[:, 1], 0.01)], device=device)
        print(f"Floor equation: {floor_eq}, Ceiling equation: {ceiling_eq}")

    
    num_samples = 1_000_000 if chamfer_distance else 100_000

    # lr = lambda step : 5 * 1e-4 # if step < 300 else 1e-4
    # lr = lambda step : (1e-4 if step < warmup_steps else 1e-4)
    lr = lambda step: config.lr_start if step < 200 else config.lr_normal

    lr_0 = lr(0)
    
    polygon_set_3d = PolygonSet3D.from_polygon_info(polygon_info, points3D, device=device, merge_thresh=vertex_merge_thresh, config=config)

    mesh = polygon_set_3d.export_triangle_mesh()
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "fitted_mesh_00.ply"), mesh)
    # o3d.visualization.draw_geometries([mesh])

    vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr_0)
    plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr_0 / config.plane_lr_downscale)



    with torch.no_grad():
        if target_vertex_classes is not None:
            assert len(target_coords) == len(target_vertex_classes)
            target_vertex_classes = torch.from_numpy(target_vertex_classes).to(device).float()
            polygon_set_3d.class_names = target_vertex_class_names

            mesh = polygon_set_3d.export_triangle_mesh()
            # target_point_pcd = o3d.geometry.PointCloud()
            # target_point_pcd.points = o3d.utility.Vector3dVector(target_coords.cpu().numpy())
            # target_point_pcd.colors = o3d.utility.Vector3dVector(target_colors.cpu().numpy())
            # o3d.visualization.draw_geometries([mesh, target_point_pcd])


            point_faces, _ = compute_closest_triangle_to_points_o3d(polygon_set_3d.get_vertices(), polygon_set_3d.triangles, target_coords)
            point_polygons = polygon_set_3d.triangle_polygons[point_faces]
            polygon_classes_soft = torch.zeros((polygon_set_3d.vertices.n_planes, len(target_vertex_class_names)), device=device, dtype=torch.float)
            plane_eqs = polygon_set_3d.get_planes()

            wall_class_ix = np.where(target_vertex_class_names == "wall")[0][0]
            ceiling_class_ix = np.where(target_vertex_class_names == "ceiling")[0][0]
            floor_class_ix = np.where(target_vertex_class_names == "floor")[0][0]
            door_class_ix = np.where(target_vertex_class_names == "door")[0][0]
            surface_class_ix = np.where(target_vertex_class_names == "surface")[0][0] if "surface" in target_vertex_class_names else np.where(target_vertex_class_names == "cupboard")[0][0]
            stair_class_ix = np.where(target_vertex_class_names == "stairs")[0][0] if "stairs" in target_vertex_class_names else np.where(target_vertex_class_names == "stair")[0][0]

            for polygon in range(polygon_set_3d.vertices.n_planes):
                # face_points = points[point_faces == face]
                mask = point_polygons == polygon
                if mask.sum() > 0:
                    face_point_classes = target_vertex_classes[point_polygons == polygon]
                    polygon_classes_soft[polygon, [wall_class_ix, ceiling_class_ix, floor_class_ix, door_class_ix, surface_class_ix, stair_class_ix]] = face_point_classes.mean(dim=0)[[wall_class_ix, ceiling_class_ix, floor_class_ix, door_class_ix, surface_class_ix, stair_class_ix]]
                    
                    
                    
            for polygon in range(polygon_set_3d.vertices.n_planes):
                    # zero out the wall probability of near-horizontal faces
                    angle_to_up = torch.acos(torch.clamp((plane_eqs[polygon, :3] * up_vector).sum(), -1 + 1e-6, 1 - 1e-6))
                    angle_to_up = torch.rad2deg(angle_to_up)
                    max_angle = 10
                    # cannot be a wall if almost vertical
                    if min(angle_to_up, 180 - angle_to_up) < 90 - max_angle:
                        polygon_classes_soft[polygon, wall_class_ix] = 0
                    #zero out the floor/ceiling probability of near-vertical faces
                    if angle_to_up > 10:
                        polygon_classes_soft[polygon, floor_class_ix] = 0
                    if 180 - angle_to_up > max_angle:
                        polygon_classes_soft[polygon, ceiling_class_ix] = 0


            polygon_set_3d.flip_misaligned_triangles()

            # color by classes
            min_proba = 0.2
            polygon_classes_hard = polygon_classes_soft.argmax(dim=-1)
            polygon_classes_hard[(polygon_classes_soft == 0).all(dim=-1) | (polygon_classes_soft.max(dim=-1).values < min_proba)] = -1
            polygon_set_3d.polygon_classes = polygon_classes_hard
            # delete everything that is not wall or ceiling or floor
            polygon_not_assigned = polygon_classes_hard == -1
            if config.delete_polygons_not_aligned_with_semantics:
                polygon_set_3d.delete_triangles(polygon_not_assigned[polygon_set_3d.triangle_polygons])
            # polygon_set_3d.polygon
            # colors = torch.ones((polygon_set_3d.vertices.n_planes, 3), device=device)
            # colors[polygon_classes_hard == wall_class_ix] = torch.tensor([1, 0, 0], device=device).float()
            # colors[polygon_classes_hard == ceiling_class_ix] = torch.tensor([0, 1, 0], device=device).float()
            # colors[polygon_classes_hard == floor_class_ix] = torch.tensor([0, 0, 1], device=device).float()
            # polygon_set_3d.polygon_colors = colors
            # o3d.visualization.draw_geometries([polygon_set_3d.export_triangle_mesh(), object_mesh])

            

        
        # self = polygon_set_3d
        
        # planes = self.vertices.get_or_compute_planes()
        # # split the polygons
        # old_vertices = self.get_vertices()
        # vertices = []
        # vertex_polygons = []
        # triangles = []
        # triangle_polygons = []
        # colors = []
        # plane_eqs = []
        # for polygon in range(self.vertices.n_planes):
        #     polygon_triangles = self.triangles[self.triangle_polygons == polygon]
        #     polygon_vertices = old_vertices[polygon_triangles]
        #     unique_vertices, reindexed_triangles = torch.unique(polygon_vertices.view(-1, 3), return_inverse=True, dim=0)
        #     reindexed_triangles = reindexed_triangles.view(-1, 3)

        #     triangles += list(reindexed_triangles + len(vertices))
        #     vertices += list(unique_vertices)
        #     triangle_polygons += list(torch.tensor([polygon] * len(reindexed_triangles)))
        #     vertex_polygons += list(torch.tensor([polygon] * len(unique_vertices)))
        #     colors += [self.polygon_colors[polygon]]
        #     plane_eqs.append(planes[polygon])

        # # assert reindexed_triangles.max() < len(vertices), f"Reindexed triangles max: {reindexed_triangles.max()}, len(vertices): {len(vertices)}"

        # mesh_vertices = torch.stack(vertices, dim=0)
        # mesh_triangles = torch.stack(triangles, dim=0)
        # mesh_triangle_polygons = torch.stack(triangle_polygons, dim=0)
        # mesh_vertex_polygons = torch.stack(vertex_polygons, dim=0)
        # triangle_colors = torch.stack(colors, dim=0)
        # plane_eqs = torch.stack(plane_eqs, dim=0)

        # self.update_triangles(mesh_triangles, mesh_vertices, mesh_triangle_polygons, mesh_vertex_polygons, triangle_colors, plane_eqs)

        # self.simplify_vertices(merge_thresh)
        # # polygon_set_3d.clear_cache()
        # # mesh = polygon_set_3d.export_triangle_mesh()
        # # o3d.io.write_triangle_mesh(os.path.join(out_dir, f"fitted_mesh_{step}_after_merge.ply"), mesh)
        # # optimizer = torch.optim.Adam(polygon_set_3d.get_params(), lr=lr(step))
        # vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr_0)
        # plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr_0 / config.plane_lr_downscale)
    # polygon_set_3d.clear_cache()
    # mesh = polygon_set_3d.export_triangle_mesh()
    # o3d.io.write_triangle_mesh(os.path.join(out_dir, "initial_mesh.ply"), mesh)

    # with torch.no_grad():
    #     target_point_faces = get_closest_face_per_point(polygon_set_3d.triangles, polygon_set_3d.get_vertices(), target_coords, max_dist=max_dist)
    #     polygon_set_3d.merge_planes(target_coords, target_point_faces, target_normals, torch.ones_like(target_colors))
    #     polygon_set_3d.clear_cache()
    #     mesh = polygon_set_3d.export_triangle_mesh()
    #     o3d.io.write_triangle_mesh(os.path.join(out_dir, "initial_mesh.ply"), mesh)
        #polygon_set_3d.retriangulate_polygons()

    print(f"Number of triangles: {len(polygon_set_3d.triangles)}")

    
    for step in tqdm(range(iterations)):
        # if step == iterations // 2:
        #     # find free-floating walls 
        #     wall_polygons = torch.where(polygon_set_3d.polygon_classes == wall_class_ix)[0]
        #     for wall_polygon in wall_polygons:
        #         # only if the wall is not too large
        #         if area_per_polygon[wall_polygon] > 2:
        #             continue
        #         # get all vertices of the wall
        #         triangle_mask = polygon_set_3d.triangle_polygons == wall_polygon
        #         wall_vertices = polygon_set_3d.triangles[triangle_mask]
        #         wall_vertex_planes = polygon_set_3d.vertices.vertex_plane_assignments[wall_vertices.view(-1)]
        #         wall_vertex_planes = wall_vertex_planes.view(-1)
        #         wall_vertex_planes = wall_vertex_planes[wall_vertex_planes != -1]
        #         if torch.unique(wall_vertex_planes).shape[0] == 1:
        #             print(f"Wall polygon {wall_polygon} is free-floating")
        #             polygon_set_3d.delete_triangles(triangle_mask)
                    

        # if step > max(simplification_steps + plane_merge_steps):
        #     max_dist = 0.02
        if step % config.recompute_polygon_areas_every == 0: # todo: implement this faster and recompute
            # intersects_per_polygon = {p.item() : c.item() for p, c in zip(unique_polygons_hit, counts)}
            triangle_areas = polygon_set_3d.compute_triangle_areas()
            area_per_polygon_dict = {p.item() : triangle_areas[polygon_set_3d.triangle_polygons == p].sum().item() for p in torch.unique(polygon_set_3d.triangle_polygons)}
            area_per_polygon = torch.zeros(max(area_per_polygon_dict) + 1, device=device, dtype=torch.float)
            for k, v in area_per_polygon_dict.items():
                area_per_polygon[k] = v


        if step == project_objects_to_floor_at_step:
            mesh_vertices = torch.from_numpy(np.array(object_mesh.vertices)).float().to(polygon_set_3d.device)
            mesh_triangles = torch.from_numpy(np.array(object_mesh.triangles)).long().to(polygon_set_3d.device)
            with torch.no_grad():
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_before_project_objects_to_floor.ply", mesh)
                project_objects_to_floor(polygon_set_3d, floor_class_ix, object_vertices=mesh_vertices, object_triangles=mesh_triangles, up_vector=up_vector, floor_must_be_below=config.multi_floor)
                
                # also project surface triangles to the floor
                # triangle_classes = polygon_set_3d.polygon_classes[polygon_set_3d.triangle_polygons]
                # surface_triangles = polygon_set_3d.triangles[triangle_classes == surface_class_ix]
                # project_objects_to_floor(polygon_set_3d, floor_class_ix, object_vertices=polygon_set_3d.get_vertices().detach(), object_triangles=surface_triangles, up_vector=up_vector, floor_must_be_below=config.multi_floor)
                polygon_set_3d.retriangulate_polygons()
                polygon_set_3d.clean()

                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_after_project_objects_to_floor.ply", mesh)
                vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr(step))
                plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr(step) / config.plane_lr_downscale)

        if do_extend_walls_to_floor and step in extend_walls_to_floor_steps:
            with torch.no_grad():
                # assert step in simplification_steps
                
                triangle_areas = polygon_set_3d.compute_triangle_areas()
                unique_polygons_hit, counts = torch.unique(polygon_set_3d.triangle_polygons[intersected_triangles], return_counts=True)
                # max_intersects_per_area_per_polygon = {p : 1.5 * intersects_per_polygon.get(p, 0) / (area_per_polygon[p] + 0.5) for p in area_per_polygon}
                # median_intersects_per_area = np.median(list(max_intersects_per_area_per_polygon.values()))
                # max_intersects_per_area_per_polygon = {p : 1000000.0 for p in area_per_polygon}

                extension_margin = ray_intersection_margin if step < iterations // 2 else 2 * ray_intersection_margin
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_before_extend_walls_to_floor.ply", mesh)
                for ix in [wall_class_ix, door_class_ix, surface_class_ix]:
                    max_intersects_per_area_per_polygon = {p.item() : config.max_ray_intersects_per_m2 if ix != surface_class_ix else config.max_ray_intersects_per_m2 / 2 for p in torch.unique(polygon_set_3d.triangle_polygons)}
                    extend_walls_to_floor(polygon_set_3d, ix, floor_class_ix, up_vector, target_ray_origins, target_ray_dests, vertex_plane_assignments=polygon_set_3d.vertices.vertex_plane_assignments, max_intersects_per_area_per_polygon=max_intersects_per_area_per_polygon, plot=False, margin=extension_margin, must_be_below=config.multi_floor, max_angle_to_vertical=config.max_wall_angle_for_extension)
                    if do_extend_walls_to_ceiling:
                        
                        extend_walls_to_floor(polygon_set_3d, ix, ceiling_class_ix, -up_vector, target_ray_origins, target_ray_dests, vertex_plane_assignments=polygon_set_3d.vertices.vertex_plane_assignments, max_intersects_per_area_per_polygon=max_intersects_per_area_per_polygon, plot=False, margin=extension_margin, must_be_below=config.multi_floor, max_angle_to_vertical=config.max_wall_angle_for_extension)
                
                polygon_set_3d.simplify_vertices()
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_after_extend_walls_to_floor.ply", mesh)
                vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr(step))
                plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr(step) / config.plane_lr_downscale)
        if simplify and step in simplification_steps:
            with torch.no_grad():
                
                # holes_planar, holes_lines = polygon_set_3d.find_holes()
                # holes = holes_planar + holes_lines
                # print(f"Number of holes: {len(holes)}")
                # if len(holes) > 0:
                #     hole_vertices = torch.cat([polygon_vertices[hole] for hole in holes])
                #     hole_vertex_point_cloud = o3d.geometry.PointCloud()
                #     hole_vertex_point_cloud.points = o3d.utility.Vector3dVector(hole_vertices.detach().cpu().numpy())
                #     o3d.io.write_point_cloud(f"{out_dir}/holes_{step}.ply", hole_vertex_point_cloud)
                #     mesh = polygon_set_3d.export_triangle_mesh()
                #     o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_holes.ply", mesh)
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_before_simplify.ply", mesh)

                polygon_set_3d.simplify()
                polygon_set_3d.clean()
                # optimizer = torch.optim.Adam(polygon_set_3d.get_params(), lr=lr(step))
                vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr(step))
                plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr(step) / config.plane_lr_downscale)
                
                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_after_simplify.ply", mesh)

        
        if merge_planes and step in plane_merge_steps and not step in extend_walls_to_floor_steps:
            with torch.no_grad():
                # if filter_classes:
                #     mesh_with_aggregated_openseg = o3d.io.read_point_cloud("/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/meshes/scannet++_debug/opengs/run23_openseg/full-DepthAndNormalMapsPoisson_mesh.ply")
                #     classes = np.load("/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/meshes/scannet++_debug/opengs/run23_openseg/aggregated_wall_floor_ceiling.npy")
                #     classes = torch.from_numpy(classes).to(device)
                #     points_with_classes = np.array(mesh_with_aggregated_openseg.points)
                #     points_with_classes = torch.from_numpy(points_with_classes).float().to(device)
                #     point_faces = get_closest_face_per_point(polygon_set_3d.triangles, polygon_set_3d.get_vertices(), points_with_classes, max_dist=max_dist)
                #     for polygon in range(polygon_set_3d.vertices.n_planes):
                #         face_points = points_with_classes[point_faces == polygon]
                #         face_point_classes = classes[point_faces == polygon]
                #         polygon_is_wall = face_point_classes.mean() > 0.5
                #         if not polygon_is_wall:
                #             polygon_set_3d.triangle_polygons[polygon] = -1
                #         else:
                #             print(f"Plane {polygon} is a wall")
                    
                mesh_before_merge = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_before_merge.ply", mesh_before_merge)
                # we assign every point a polygon: but only if it is very close (ransac-like)
                target_point_faces = get_closest_face_per_point(polygon_set_3d.triangles, polygon_set_3d.get_vertices(), target_coords, max_dist=vertex_merge_thresh)
                polygon_set_3d.merge_planes(target_coords, target_point_faces, target_normals, torch.ones_like(target_colors), recompute_plane_eqs=lr(step) == 0)

                mesh = polygon_set_3d.export_triangle_mesh()
                o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}_after_merge.ply", mesh)
                # polygon_set_3d.clear_cache()
                # mesh = polygon_set_3d.export_triangle_mesh()
                # o3d.io.write_triangle_mesh(os.path.join(out_dir, f"fitted_mesh_{step}_after_merge.ply"), mesh)
                # optimizer = torch.optim.Adam(polygon_set_3d.get_params(), lr=lr(step))
                vertex_optimizer = torch.optim.Adam([polygon_set_3d.vertices.vertices], lr=lr(step))
                plane_optimizer = torch.optim.Adam([polygon_set_3d.vertices.planes], lr=lr(step) / config.plane_lr_downscale)
        

        polygon_set_3d.clear_cache()

        polygon_vertices, projection_distances = polygon_set_3d.get_vertices(return_distances=True)
        
        reg_strength = config.regularization_strength
        regularization_strength = lambda step : reg_strength
        # if chamfer_distance:
        #     regularization_strength = 5 * 1e-6
        #     # regularization_strength = lambda step : 0.01 *(1 if step < 300 else 0.2 if step < max(simplification_steps) // 2 else 0.1) #(0.01 if step < max(simplification_steps) else 0.001))
        # elif point_triangle_distance:
        #     regularization_strength = lambda step : 0 # 0.00001 # * (1 if step < 300 else 1 if step < max(simplification_steps) // 2 else 0.1) #(0.01 if step < max(simplification_steps) else 0.001))
        # elif ray_tracing_distance:
        #     regularization_strength = lambda step : 5 * 1e-6 # if step < 500 else 5 * 1e-6
        if lr(step) > 0:
            if len(target_coords) > 2 * num_samples:
                random_target_ixes = np.random.choice(len(target_coords), num_samples, replace=False)
                # random_target_ixes = torch.randint(0, len(target_coords), (num_samples,), device=device)
                random_target_ixes = torch.from_numpy(random_target_ixes).to(device)
                target_coord_samples, target_normal_samples, target_color_samples = target_coords[random_target_ixes], target_normals[random_target_ixes], target_colors[random_target_ixes]   
                if target_ray_dests is not None:
                    random_ray_ixes = np.random.choice(len(target_ray_origins), num_samples, replace=False)
                    random_ray_ixes = torch.from_numpy(random_ray_ixes).to(device)

                    
            else:
                target_coord_samples, target_normal_samples, target_color_samples = target_coords, target_normals, target_colors

            fitting_loss = 0.01 * projection_distances.mean() # add this to stop the points from wandering off and becoming numerically unstable
            if ray_tracing_distance:
                triangle_planes = polygon_set_3d.get_planes()[polygon_set_3d.triangle_polygons]
                inner_edge_mask = polygon_set_3d.get_inner_edge_mask_with_cache()
                t = time.time()
                polygon_areas_per_triangle = area_per_polygon[polygon_set_3d.triangle_polygons]
                ray_tracing_loss, intersected_triangles = ray_tracing_distance_loss(vertices=polygon_vertices, triangles=polygon_set_3d.triangles, triangle_plane_eqs=triangle_planes, ray_origins=target_ray_origins[random_ray_ixes], cover_points=target_coords[random_target_ixes], ray_dests=target_ray_dests[random_ray_ixes], max_dist=max_dist, is_inner_edge_mask=inner_edge_mask, margin=ray_intersection_margin, polygon_areas_per_triangle=polygon_areas_per_triangle)
                fitting_loss += 100 * (len(random_ray_ixes) / len(target_ray_origins)) * ray_tracing_loss
                # print(f"Ray tracing distance took {time.time() - t} seconds")
            if chamfer_distance:
                points_with_classes, normals, (colors, ) = sample_points_from_mesh(polygon_set_3d.triangles, polygon_vertices, triangle_features=[polygon_set_3d.get_triangle_colors()], num_samples=num_samples, return_normals=True)

                points_with_classes = torch.cat((points_with_classes, polygon_vertices), dim=0)
                normals = torch.cat((normals, torch.zeros_like(polygon_vertices)), dim=0)
                # sample_src = torch.cat((sample_src, normals_src, 100 * colors_src), dim=1)
                # sample_trg = torch.cat((sample_trg, normals_trg, 100 * colors_trg), dim=1)


                if window:
                    t = time.time()
                    t2_total = 0
                    windowed_src, windowed_src_ixes = split_into_windows(points_with_classes, window_split_x, window_split_y, window_split_z)
                    windowed_targets, windowed_target_ixes = split_into_windows(target_coord_samples, window_split_x, window_split_y, window_split_z)
                    windowed_src_normals = [normals[ix] for ix in windowed_src_ixes]
                    windowed_targets_normals = [target_normal_samples[ix] for ix in windowed_target_ixes]

                    position_loss = 0
                    normal_loss = 0
                    total_src_len = sum(len(src) for src in windowed_src)
                    total_trg_len = sum(len(trg) for trg in windowed_targets)
                    for src, trg, src_norm, target_norm in zip(windowed_src, windowed_targets, windowed_src_normals, windowed_targets_normals):
                        if len(src) == 0 or len(trg) == 0:
                            continue
                        t2 = time.time()
                        loss_to_closest_src, loss_to_closest_trg = truncated_chamfer_distance(src, trg, src_norm, target_norm, max_dist=max_dist, max_angle=max_angle, include_normals=include_normals)
                        position_loss += loss_to_closest_src.mean() + loss_to_closest_trg.mean()
                        # loss_to_closest_src = loss_to_closest_src.sum() / total_src_len
                        # loss_to_closest_trg = loss_to_closest_trg.sum() / total_trg_len
                        # position_loss += loss_to_closest_src + loss_to_closest_trg

                        t2_total += time.time() - t2
                    # print(f"Chamfer took {time.time() - t} seconds, whereof {t2_total} seconds were in the loop")
                        # position_loss += (trg[closest_src][:, :3] - src[:, :3]).norm(dim=-1).mean()
                        # normal_loss += (target_norm[closest_src] - src_norm)[mask].norm(dim=-1).mean()
                    # closest_trg, closest_src = windowed_mutual_nearest_neighbors(points, target_coords, window_split_x, window_split_y, window_split_z)
                else:
                    t = time.time()
                    loss_to_closest_src, loss_to_closest_trg = truncated_chamfer_distance(points_with_classes, target_coord_samples, normals, target_normal_samples, max_dist=max_dist, max_angle=max_angle, include_normals=include_normals)
                    position_loss += loss_to_closest_src.mean() + loss_to_closest_trg.mean()
                    print(f"Chamfer took {time.time() - t} seconds")
                # color_loss = (colors[closest_trg] - target_colors).norm(dim=-1).mean() + (colors - target_colors[closest_src]).norm(dim=-1).mean()
                # normal_loss = (normals[closest_trg] - target_normals).norm(dim=-1).mean() + (normals - target_normals[closest_src]).norm(dim=-1).mean()
                if not point_triangle_distance:
                    fitting_loss += 5 * position_loss # + color_loss # + normal_loss
                else:
                    fitting_loss += 0.1 * position_loss
            if point_triangle_distance:
                # use pytorch3d point face distance
                triangles = polygon_set_3d.triangles.unsqueeze(0)
                vertices = polygon_vertices.unsqueeze(0)
                import pytorch3d
                pytorch3d_meshes = pytorch3d.structures.Meshes(verts=vertices, faces=triangles)
                pytorch3d_pcds = pytorch3d.structures.Pointclouds(points=target_coord_samples.unsqueeze(0))
                
                fitting_loss += 1000 * point_mesh_face_distance(pytorch3d_meshes, pytorch3d_pcds, max_dist=max_dist)
                # fitting_loss = point_to_face_distance(pytorch3d_meshes, pytorch3d_pcds)


            loss = fitting_loss

            if regularize and regularization_strength(step) > 0:


                # if regularize_wall_reach_ceiling:
                #     planes = polygon_set_3d.get_planes()
                #     contour_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=True)
                #     # triangle_planes = planes[polygon_set_3d.triangle_polygons]
                #     vertex_constraints = polygon_set_3d.vertices.vertex_plane_assignments
                #     loss += 100 * walls_must_reach_the_floor_loss(polygon_vertices, contour_edges=contour_edges, vertex_constraints=vertex_constraints, floor_eq=floor_eq, ceiling_eq=ceiling_eq)

                # vertex-edge magnetism -------------
                # contour_edges = polygon_set_3d.get_contour_edges(return_non_watertight=True)
                
                contour_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=True)
                # assert (contour_edges == contour_edges_cached).all()
                if len(contour_edges) > 0:
                    k = 1
                    
                    vertex_edge_indices, vertex_edge_distances = get_closest_edges_to_point_sq_dist(polygon_vertices, polygon_vertices[contour_edges], k=k)
                    vertex_edge_distances = (vertex_edge_distances + 1e-4).sqrt()
                    # vertex_edge_distances = vertex_edge_distances.sort(dim=-1).values
                    # contour_edge_polygons = polygon_set_3d.vertices.vertex_plane_assignments[contour_edges[vertex_edge_indices]]
                    # vertex_polygons = self.vertices.vertex_plane_assignments[:, 0]
                    # vertex_edge_distances[(contour_edge_polygons == vertex_polygons[:, None, None, None]).all(dim=-1).any(dim=-1)] = float("inf") # if one of the edge vertices has the exact same polygon assignment (usually on the same plane): ignore
                    # different_polygon_mask = (polygon_set_3d.vertices.vertex_plane_assignments[:, :, None, None, None] != contour_edge_polygons[:, None]) | (polygon_set_3d.vertices.vertex_plane_assignments[:, :, None, None, None] == -1)
                    # different_polygon_mask = different_polygon_mask.all(dim=1).all(dim=-1).any(dim=-1)
                    # different_polygon_mask = torch.ones_like(different_polygon_mask, dtype=torch.bool)
                    
                    # what we consider merged must not unmerge
                    # loss +=  (vertex_edge_distances[(vertex_edge_distances > 0) & (vertex_edge_distances < vertex_merge_thresh)] ** 2).sum()
                    
                    if regularize_edge_magnetism and step > 100:
                        loss +=  (vertex_edge_distances[(vertex_edge_distances > 0) & (vertex_edge_distances < vertex_merge_thresh)] ** 2).sum()
                        vertex_edge_distances = vertex_edge_distances[(vertex_edge_distances > 0) & (vertex_edge_distances <  magnetism_threshold)]
                        # loss += regularization_strength(step) * 50 * (vertex_edge_distances + 1e-4).sqrt().mean()
                        # loss += regularization_strength(step) * 20 * (vertex_edge_distances).sum()
                        if chamfer_distance:
                            if step < 100:
                                loss += regularization_strength(step) * torch.log(vertex_edge_distances[vertex_edge_distances < magnetism_threshold] + 1e-4).sum()
                            else:
                                loss += regularization_strength(step) * 5 * torch.log(vertex_edge_distances[vertex_edge_distances < magnetism_threshold] + 1e-4).sum()
                                loss += regularization_strength(step) * 0.1 * torch.log(vertex_edge_distances[vertex_edge_distances >= magnetism_threshold] + 1e-4).sum()
                        # loss += regularization_strength(step) * 50 * vertex_edge_distances.abs().mean()
                        else:

                            loss += regularization_strength(step) * 0.05 * torch.log(magnetism_threshold + vertex_edge_distances[vertex_edge_distances < magnetism_threshold] + 1e-4).sum()
                            # if step < 500:
                            #     loss += regularization_strength(step) * 10 * torch.log(vertex_edge_distances[vertex_edge_distances < magnetism_threshold] + 1e-4).sum()
                            # else:
                            #     loss += regularization_strength(step) * 0.1 * torch.log(vertex_edge_distances[vertex_edge_distances < magnetism_threshold] + 1e-4).sum()

                if regularize_face_magnetism and step > min(simplification_steps) and step > 100:
                    t = time.time()
                    triangle_vertices = polygon_vertices[polygon_set_3d.triangles]
                    triangle_normals = plane_eqs[polygon_set_3d.triangle_polygons][:, :3]
                    face_indices, vertex_face_distances, closest_points = find_k_closest_triangles(polygon_vertices, triangle_vertices.detach(), triangle_normals.detach(), k=20)
                    vertex_polygons = polygon_set_3d.vertices.vertex_plane_assignments
                    face_polygons = polygon_set_3d.triangle_polygons[face_indices]
                    not_shared = (face_polygons[:, :, None] != vertex_polygons[:, None]).all(dim=-1)
                    vertex_face_distances = vertex_face_distances[not_shared]
                    
                    # what is merged must not unmerge
                    # loss +=  (vertex_face_distances[(vertex_face_distances > 0) & (vertex_face_distances < vertex_merge_thresh)] ** 2).sum()
                    assert not torch.isinf(vertex_face_distances).any()
                    assert not torch.isnan(vertex_face_distances).any()
                    # loss += regularization_strength(step) * vertex_face_distances[vertex_face_distances < magnetism_threshold].sum()
                    loss += regularization_strength(step) * 0.1 * torch.log(magnetism_threshold + vertex_face_distances[vertex_face_distances < magnetism_threshold] + 1e-4).sum()
                    # print(f"Face magnetism took {time.time() - t} seconds")
                # last_merge_step = -100 if step <= plane_merge_steps[0] else max([s for s in plane_merge_steps if s <= step])
                # steps_sice_last_merge = step - last_merge_step
                # next_merge_step = min([s for s in plane_merge_steps if s > step], default=iterations)
                # if step < max(simplification_stets) // 2 or steps_sice_last_merge > (next_merge_step - last_merge_step) // 3:
                
                if regularize_non_watertight:
                    # non_watertight_edges = polygon_set_3d.get_contour_edges()
                    non_watertight_edges = polygon_set_3d.get_non_watertight_edges(return_if_attached_to_edge=False)

                    # # let's consider it watertight if both vertices are close to the edge
                    # contour_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=True)
                    # if len(contour_edges) > 0:
                    #     k = 5
                    #     vertex_edge_indices, vertex_edge_distances = get_closest_edges_to_point(polygon_vertices, polygon_vertices[contour_edges], k=5)
                        

                    #     vertex_part_of_multiple_polygons = (polygon_set_3d.vertices.vertex_plane_assignments[:, 1] != -1) 
                    #     MAX_N_POLYGONS_PER_VERTEX = 3
                    #     vertex_close_edges = contour_edges[vertex_edge_indices]
                    #     # print(f"first close edge: {vertex_close_edges[0]}")
                    #     vertex_close_polygons = self.vertices.vertex_plane_assignments[vertex_close_edges].permute(0, 2, 1, 3).reshape(-1, 2, MAX_N_POLYGONS_PER_VERTEX * k)
                    #     # print(f"first close polygon: {vertex_close_polygons[0]}")
                    #     vertex_close_enough = (vertex_edge_distances < merge_thresh).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, MAX_N_POLYGONS_PER_VERTEX).view(-1, 2, MAX_N_POLYGONS_PER_VERTEX * k)
                    #     close_polygon_valid = (vertex_close_polygons != -1) & vertex_close_enough
                    #     # print(f"Ratio of close polygons: {close_polygon_valid.float().mean()}")
                    #     vertex_plane = polygon_set_3d.vertices.vertex_plane_assignments[:, 0]
                    #     vertex_close_to_different_polygon = ((vertex_close_polygons != vertex_plane[:, None, None]) & close_polygon_valid).any(dim=-1).all(dim=-1) # both edges any different polygon
                    #     watertight_vertex = vertex_close_to_different_polygon | vertex_part_of_multiple_polygons

                    #     # print(f"Ratio of watertight vertices: {watertight_vertex.float().mean()}")

                    #     non_watertight_edges = non_watertight_edges[~watertight_vertex[non_watertight_edges].any(dim=-1)]


                    non_watertight_distance = (polygon_vertices[non_watertight_edges[:, 0]] - polygon_vertices[non_watertight_edges[:, 1]]).norm(dim=-1)
                    # non_watertight_loss = torch.clamp(non_watertight_loss, max=5 * max_dist)
                    non_watertight_loss = (non_watertight_distance + 1e-4).mean()
                    # loss += regularization_strength(step) * 0.01 * non_watertight_loss
                    if chamfer_distance:
                        if step < warmup_steps:
                            loss += regularization_strength(step) * 1000 * non_watertight_loss
                        elif step < 200:
                            loss += regularization_strength(step) * 100 * non_watertight_loss
                        else:
                            # non_watertight_loss = torch.clamp(non_watertight_loss, max=5 * max_dist)
                            # non_watertight_loss[non_watertight_loss > max_dist] = 0
                            # loss += regularization_strength(step) * 0.01 * non_watertight_loss
                            loss += regularization_strength(step) * 50 * non_watertight_loss
                    else:
                        # if step < 500:
                        # loss += regularization_strength(step) * torch.log(1e-4 + non_watertight_distance).sum()
                        loss += regularization_strength(step) * 0.5 * torch.log(magnetism_threshold + non_watertight_distance).sum()
                        loss += regularization_strength(step) * 0.5 * non_watertight_distance.sum()
                        # else:
                        #     loss += regularization_strength(step) * 10 * torch.log(1e-4 + non_watertight_distance).sum()
                    if step in simplification_steps:
                        non_watergight_line_set = o3d.geometry.LineSet()
                        non_watergight_line_set.points = o3d.utility.Vector3dVector(polygon_vertices.detach().cpu().numpy())
                        non_watergight_line_set.lines = o3d.utility.Vector2iVector(non_watertight_edges.detach().cpu().numpy())
                        o3d.io.write_line_set(f"{out_dir}/fitted_mesh_non_watertight_edges_{step}.ply", non_watergight_line_set)

                    all_edges = torch.stack([polygon_set_3d.triangles[:, [0, 1]], polygon_set_3d.triangles[:, [1, 2]], polygon_set_3d.triangles[:, [2, 0]]], dim=0).view(-1, 2)
                    all_edges = torch.unique(all_edges.sort(dim=-1).values, dim=0)
                    all_edge_lengths = (polygon_vertices[all_edges[:, 0]] - polygon_vertices[all_edges[:, 1]]).norm(dim=-1)
                    # if step < 300:
                    #     loss += regularization_strength(step) * 10 * all_edge_lengths.mean()
                    loss += regularization_strength(step) * 0.1 * all_edge_lengths.sum()


                # mesh growth
                if regularize_area:
                    triangle_areas = polygon_set_3d.compute_triangle_areas()
                    if chamfer_distance:
                        # if step < 300:
                        #     loss -= 0.05 * triangle_areas.sum() # 0.05 is too much growth!
                        # else:
                        #     loss -= 0.05 * triangle_areas.sum()
                        if step < 300:
                            loss -= 10 * (triangle_areas + 1e-4).sqrt().mean()
                        else:
                            loss -= 5 * (triangle_areas + 1e-4).sqrt().mean()
                    else:
                        # per_polygon_area = 0
                        
                        # for polygon in range(polygon_set_3d.vertices.n_planes):
                        #     triangle_mask = polygon_set_3d.triangle_polygons == polygon
                        #     polygon_triangles = polygon_set_3d.triangles[triangle_mask]
                        #     polygon_edges = torch.cat([polygon_triangles[:, [0, 1]], polygon_triangles[:, [1, 2]], polygon_triangles[:, [2, 0]]], dim=0)
                        #     unique_edges, edge_counts = torch.unique(polygon_edges, return_counts=True, dim=0)
                        #     polygon_contour_edges = unique_edges[edge_counts == 1]

                        #     polygon_contour_length = (polygon_vertices[polygon_contour_edges[:, 0]] - polygon_vertices[polygon_contour_edges[:, 1]]).norm(dim=-1)
                        #     polygon_contour_length = polygon_contour_length.sum()
                        #     polygon_area = triangle_areas[triangle_mask].sum()

                        #     # loss += regularization_strength(step) * 0.1 * polygon_contour_length / (polygon_area + 1e-4).sqrt()

                            
                            
                            # loss += regularization_strength(step) * 100000 * (polygon_area + 1e-4).sqrt().sum()
                            
                        # loss -= regularization_strength(step) * 0.001 * ((triangle_areas + 1e-4).sqrt()).sum()
                        loss -= regularization_strength(step) * 0.00002 * (triangle_areas ** 2).sum()
                    # if step < 300:
                    #     loss -= regularization_strength(step) * 0.01 * triangle_areas.sum()
                    # else:
                    #     loss -= regularization_strength(step) * 0.1 * triangle_areas.sum()

                # shared_edges = polygon_set_3d.get_contour_edges(return_non_watertight=False)
                # shared_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=False)
                # assert (shared_edges == shared_edges_cached).all()
                # if len(shared_edges) > 0:
                #     shared_edge_lengths = (polygon_vertices[shared_edges[:, 0]] - polygon_vertices[shared_edges[:, 1]]).norm(dim=-1)
                #     # shared_edge_lengths = torch.clamp(shared_edge_lengths, max=5 * max_dist)
                #     shared_edge_lengths = (shared_edge_lengths + 1e-5).sqrt().mean()
                #     loss -= regularization_strength(step) * 150 * shared_edge_lengths.mean()
                
            
                # vertex magnetism ---------------
                if regularize_vertex_magnetism:
                    with torch.no_grad():
                        vertex_nn, _ = mutual_nearest_neighbors(polygon_vertices.unsqueeze(0), polygon_vertices.unsqueeze(0), torch.tensor([len(polygon_vertices)]).to(device), torch.tensor([len(polygon_vertices)]).to(device), k=3)
                    vertex_magnetism = (polygon_vertices[vertex_nn.squeeze(0)] - polygon_vertices[:, None]).norm(dim=-1)
                    # vertex_magnetism = (vertex_magnetism[(vertex_magnetism > 0) & (vertex_magnetism < 5 * max_dist)] + 1e-6).sqrt().mean()
                    # vertex_magnetism = torch.clamp(vertex_magnetism, max=5 * max_dist)
                    vertex_magnetism = vertex_magnetism[(vertex_magnetism > 0) & (vertex_magnetism < magnetism_threshold)]
                    loss += regularization_strength(step) * 0.05 * torch.log(0.5 * magnetism_threshold + vertex_magnetism + 1e-4).sum()
                    
                # loss += projection_distances.abs().mean()

                if False: # step > warmup_steps: # and step < max(simplification_stets + plane_merge_steps):

                    # contour_edges = polygon_set_3d.get_contour_edges()
                    # contour_loss = 0 * shared_edge_growth_strength * - (polygon_vertices[contour_edges[:, 0]] - polygon_vertices[contour_edges[:, 1]]).norm(dim=-1).mean()

                    # contour_loss = 0 if step > 250 else contour_loss
                    # all_edges = polygon_set_3d.get_unique_edges()
                    # edge_length_loss = (polygon_vertices[all_edges[:, 0]] - polygon_vertices[all_edges[:, 1]]).norm(dim=-1).mean()
                    outer_edges = polygon_set_3d.get_inner_edges()
                    inner_edge_length_loss = (((polygon_vertices[outer_edges[:, 0]] - polygon_vertices[outer_edges[:, 1]])).norm(dim=-1)**2).mean()

                    # + 0.025 * contour_loss
                    projection_distance_loss = projection_distances.abs().mean()
                    regularization_loss = 10 * vertex_edge_distances  +  0.0001 * inner_edge_length_loss + 0.1 * vertex_magnetism # + projection_distance_loss
                    loss += regularization_strength(step) * regularization_loss

                edge_triplets = polygon_set_3d.find_edge_pairs_with_cache()
                assert polygon_vertices.shape[0] >= edge_triplets.max(), f"Max vertex index: {polygon_vertices.shape[0]}, max edge index: {edge_triplets.max()}"

                if step > 500 and regularize_straightness:
                    # penalize distance to line connecting first and last
                    edge_triplets = polygon_set_3d.find_edge_pairs_with_cache()
                    assert polygon_vertices.shape[0] >= edge_triplets.max(), f"Max vertex index: {polygon_vertices.shape[0]}, max edge index: {edge_triplets.max()}"
                    triplet_vertices = polygon_vertices[edge_triplets]
                    points_to_project = triplet_vertices[:, 1]
                    line_points = torch.cat([triplet_vertices[:, 0], triplet_vertices[:, 2]], dim=1)
                    stable_distance = (line_points[:, :3] - line_points[:, 3:]).norm(dim=-1) > 1e-5
                    _, dist = project_points_to_lines_torch(points_to_project[stable_distance], line_points[stable_distance])
                    # too_far_mask = dist > magnetism_threshold
                    loss += regularization_strength(step) * (dist + 1e-4).sqrt().sum()



                if regularize_rectangularity and step > 500:
                    edge_triplets = polygon_set_3d.find_edge_pairs_with_cache()
                    triplet_vertices = polygon_vertices[edge_triplets]
                    e1 = triplet_vertices[:, 1] - triplet_vertices[:, 0]
                    e2 = triplet_vertices[:, 2] - triplet_vertices[:, 1]
                    # dot_product = (e1 * e2).sum(dim=-1).abs() / (e1.norm(dim=-1) * e2.norm(dim=-1) + 1e-10)
                    # angle_loss = torch.minimum(1 - dot_product, dot_product)
                    # angle_loss = (angle_loss + 1e-5).log().mean()
                    # loss += regularization_strength(step) * 0.1 * 50 * angle_loss
                    # # angle_loss = (angle_loss + 1e-5).sqrt()$
                    # # angle_loss = (1 + 10 * angle_loss).log().mean()
                    # loss += 10 * angle_loss.mean()

                    angle = torch.acos(torch.clamp((e1 * e2).sum(dim=-1) / (e1.norm(dim=-1) * e2.norm(dim=-1) + 1e-10), -1 + 1e-4, 1 - 1e-4))
                    angle_deg = torch.rad2deg(angle)
                    angle_to_90 = torch.minimum(torch.abs(angle_deg - 90), torch.minimum(torch.abs(angle_deg - 180), torch.abs(angle_deg)))
                    # angle_to_90 = torch.minimum(torch.abs(angle_deg - 90), torch.abs(angle_deg))
                    # angle_to_90[angle_to_90 > 30] = 0.5 * angle_to_90[angle_to_90 > 30]
                    # angle_to_90[angle_to_90 > 15] = 0 #0.2 * angle_to_90[angle_to_90 > 30]
                    # angle_to_90 = angle_to_90 * (e1.norm(dim=-1) * e2.norm(dim=-1))
                    angle_loss = angle_to_90
                    loss += regularization_strength(step) * 1e-2 * angle_loss.sum()
                    # if step < 300:
                    #     loss += regularization_strength(step) * 0.5 * 50 * angle_loss
                    # else:
                    #     loss += regularization_strength(step) * 5 * 50 * angle_loss

                        # contour_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=True)
                        # triangle_vertices = polygon_vertices[polygon_set_3d.triangles]
                        # e1, e2 = triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 1]
                        # smallest_angle_in_triangle
                if regularize_manhattan:
                    # contour_edges = polygon_set_3d.get_contour_edges_with_cache(return_non_watertight=True)
                    contour_edges = polygon_set_3d.get_non_watertight_edges(return_if_attached_to_edge=False)
                    contour_edge_polygons = polygon_set_3d.vertices.vertex_plane_assignments[contour_edges]
                    edge_classes = polygon_set_3d.polygon_classes[contour_edge_polygons]
                    edge_is_wall = ((edge_classes == wall_class_ix) | (edge_classes == surface_class_ix) | (edge_classes == door_class_ix) | (edge_classes == stair_class_ix)).any(dim=-1).all(dim=-1)
                    contour_edges = contour_edges[edge_is_wall]
                    e1 = polygon_vertices[contour_edges[:, 1]] - polygon_vertices[contour_edges[:, 0]]
                    # up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
                    angle_to_up = torch.acos(torch.clamp((e1 * up_vector).sum(dim=-1) / (e1.norm(dim=-1) * up_vector.norm(dim=-1) + 1e-10), -1 + 1e-4, 1 - 1e-4))
                    angle_to_up_deg = torch.rad2deg(angle_to_up)
                    angle_to_manhattan = torch.minimum(torch.abs(angle_to_up_deg - 90), torch.minimum(torch.abs(angle_to_up_deg - 180), torch.abs(angle_to_up_deg)))
                    # angle_to_manhattan[angle_to_manhattan > 15] = 0 #0.2 * angle_to_manhattan[angle_to_manhattan > 30]
                    angle_to_manhattan = angle_to_manhattan
                    if ray_tracing_distance:
                        loss += regularization_strength(step) * 0.03 * angle_to_manhattan.sum()
                    else:
                        regularization_strength(step) * angle_to_manhattan.sum()
                        # if step < min(extend_walls_to_floor_steps):
                        #     loss += regularization_strength(step) * angle_to_manhattan.sum()
                        # else:
                        #     loss += regularization_strength(step) * 5 * angle_to_manhattan.sum()
                            

                    # floor plane must be horizontal
                    planes = polygon_set_3d.get_planes()
                    floor_planes = planes[polygon_set_3d.polygon_classes == floor_class_ix] 
                    floor_normal = floor_planes[:, :3]
                    angle_to_up = torch.acos(torch.clamp((floor_normal * up_vector).sum(dim=-1) / (floor_normal.norm(dim=-1) * up_vector.norm(dim=-1) + 1e-10), -1 + 1e-4, 1 - 1e-4))
                    angle_to_up_deg = torch.rad2deg(angle_to_up)
                    loss += regularization_strength(step) * 5 * 100 * angle_to_up_deg.sum()

                    # walls must be vertical
                    wall_planes = planes[polygon_set_3d.polygon_classes == wall_class_ix]
                    wall_normal = wall_planes[:, :3]
                    angle_to_up = torch.acos(torch.clamp((wall_normal * up_vector).sum(dim=-1) / (wall_normal.norm(dim=-1) * up_vector.norm(dim=-1) + 1e-10), -1 + 1e-4, 1 - 1e-4))
                    angle_to_up_deg = torch.rad2deg(angle_to_up)
                    angle_to_up_deg = torch.minimum(torch.abs(angle_to_up_deg - 90), torch.minimum(torch.abs(angle_to_up_deg - 180), torch.abs(angle_to_up_deg)))
                    angle_to_up_deg  = angle_to_up_deg[angle_to_up_deg < 30]
                    loss += regularization_strength(step) * 50 * angle_to_up_deg.sum()


                    # if step < 300:
                    #     loss += regularization_strength(step) * 1 * 50 * angle_to_manhattan.mean()
                    # else:
                    #     loss += regularization_strength(step) * 10 * 50 * angle_to_manhattan.mean()
                    
                # triangle_vertices = polygon_vertices[polygon_set_3d.triangles]
                # e1, e2, e3 = triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 1], triangle_vertices[:, 0] - triangle_vertices[:, 2]
                # angle_1 = torch.acos(torch.clamp((e1 * e2).sum(dim=-1) / (e1.norm(dim=-1) * e2.norm(dim=-1) + 1e-10), -1 + 1e-6, 1 - 1e-6))
                # angle_2 = torch.acos(torch.clamp((e2 * e3).sum(dim=-1) / (e2.norm(dim=-1) * e3.norm(dim=-1) + 1e-10), -1 + 1e-6, 1 - 1e-6))
                # angle_3 = torch.acos(torch.clamp((e3 * e1).sum(dim=-1) / (e3.norm(dim=-1) * e1.norm(dim=-1) + 1e-10), -1 + 1e-6, 1 - 1e-6))
                # max_angle_rad = torch.maximum(torch.maximum(angle_1, angle_2), angle_3)
                # max_angle_deg = torch.rad2deg(max_angle_rad)
                # angle_to_90 = torch.abs(max_angle_deg - 90)
                # angle_loss = (angle_to_90).mean()
                # loss += 0.3 * angle_loss
            # loss = fitting_loss + 0.1 * non_watertight_loss + 0.1 * contour_loss + 0.01 * projection_distance_loss


            # loss = loss_chamfer  # + color_loss + normal_loss
            # old_vertices = polygon_set_3d.vertices.vertices.clone()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(polygon_set_3d.vertices.vertices, max_norm=10, error_if_nonfinite=True)
            # torch.nn.utils.clip_grad_norm_(polygon_set_3d.vertices.planes, max_norm=50, error_if_nonfinite=True)
            # print(f"grad norm: verts {polygon_set_3d.vertices.vertices.grad.norm(dim=-1).mean()} planes {polygon_set_3d.vertices.planes.grad.norm(dim=-1).mean()}")
            # optimizer.step()
            # optimizer.zero_grad()
            vertex_optimizer.step()
            vertex_optimizer.zero_grad()
            plane_optimizer.step()
            plane_optimizer.zero_grad()
            assert not torch.isnan(polygon_set_3d.vertices.vertices).any()
            assert not torch.isnan(polygon_set_3d.vertices.planes).any()
        # print(f"Loss: {loss.item()}")
        # print(f"Has changed: ", (old_vertices != polygon_set_3d.vertices.vertices).any())

        if step % save_interval == 0:
            # print(f"Step {step}, loss: {loss.item()}, num_intersects: {len(intersected_triangles)}")
            mesh = polygon_set_3d.export_triangle_mesh()
            o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_{step}.ply", mesh)
            print(f"Saved checkpoint at {out_dir}/fitted_mesh_{step}.ply")
            
            pcd = polygon_set_3d.export_color_coded_vertex_cloud()
            o3d.io.write_point_cloud(f"{out_dir}/vertices_{step}.ply", pcd)

            # save the model's state
            torch.save(polygon_set_3d.state_dict(), f"{out_dir}/polygon_set_3d_{step}.pt")
            # make sure the model can be loaded
            # polygon_set_3d_new = PolygonSet3D(torch.empty((0,3)), torch.tensor([]), torch.tensor([]), device=device)
            # polygon_set_3d_new.load_state_dict(torch.load(f"{out_dir}/polygon_set_3d_{step}.pt"), strict=False)
            # polygon_set_3d = polygon_set_3d_new
            # o3d.visualization.draw_geometries([mesh, pcd])

            # also save sampled points and target points
            # give each pair of mutually nearest neighbors a different color
            
            # x_lengths = torch.tensor([len(points)]).to(device)
            # y_lengths = torch.tensor([len(target_coords)]).to(device)
            # closest_src, closest_trg = mutual_nearest_neighbors(points.unsqueeze(0), target_coords.unsqueeze(0), x_lengths, y_lengths)
            # closest_src = closest_src.view(-1)
            # closest_trg = closest_trg.view(-1)

            # mesh = o3d.geometry.TriangleMesh()
            # vertices = []
            # triangles = []
            # for s, v in zip(points[closest_trg].cpu().detach().numpy(), target_coords.cpu().detach().numpy()):
            #     vertices.append(s)
            #     vertices.append(v)
            #     triangles.append([len(vertices) - 2, len(vertices) - 1, len(vertices) - 1])

            # mesh.vertices = o3d.utility.Vector3dVector(vertices)
            # mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # o3d.io.write_triangle_mesh(f"{out_dir}/sampled_points_{step}.ply", mesh)
        # points = points.detach().cpu().numpy()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # # normal-colored
        # # pcd.colors = o3d.utility.Vector3dVector(normals.detach().cpu().numpy() * 0.5 + 0.5) 
        # # color-colored
        # pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        # # o3d.visualization.draw_geometries([pcd])
        # # save to tmp
        # o3d.io.write_point_cloud(f"tmp.ply", pcd)
        # o3d.io.write_triangle_mesh(f"mesh.ply", mesh)
        

    polygon_set_3d.clean()


    
    torch.save(polygon_set_3d.state_dict(), f"{out_dir}/polygon_set_3d.pt")
    mesh = polygon_set_3d.export_triangle_mesh()
    o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh.ply", mesh)
    print(f"Wrote fitted mesh to {out_dir}/fitted_mesh.ply")

    # new_info, new_vertices = polygon_set_3d.to_polygon_info()
    # np.save(f"{out_dir}/vertices.npy", new_vertices)
    # poly_info_new = {info["id"]: info for info in new_info}
    # with open(f"{out_dir}/polygon_info.json", 'w') as f:
    #     json.dump(poly_info_new, f)

    # save torch model 
    # o3d.visualization.draw_geometries([mesh])
    print(f"Number of polygon contours after simplification: {sum(map(len, [polygon['contours'] for polygon in polygon_info]))} whereof {len(polygon_info)} are outer contours.")
    print(f"Number of vertices after simplification: {len(mesh.vertices)}")

    # # create a filtered version after polygon simplification
    # polygon_mask = [info["class"] == "wall" for info in polygon_info]
    # filtered_mesh = polygon_set_3d.export_triangle_mesh(polygon_mask=polygon_mask)
    # o3d.io.write_triangle_mesh(f"{out_dir}/fitted_mesh_walls_only.ply", filtered_mesh)
    # print(f"Wrote filtered mesh to {out_dir}/fitted_mesh_walls_only.ply")

    if nerfstudio_transforms_file is not None:
        copied_mesh = transform_mesh(mesh, nerfstudio_transforms_file)
        if copied_mesh is not None:
            o3d.io.write_triangle_mesh(f"{out_dir}/transformed_mesh.ply", copied_mesh)
            print(f"Wrote transformed mesh to {out_dir}/transformed_mesh.ply")



def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for room splitting script.")
    parser.add_argument("--scene-type", type=str, default="scannetpp", help="Type of scene to process.")
    parser.add_argument('--rectified-ply-path', type=str, required=True, help='Path to the rectified PLY file.')
    parser.add_argument('--target-pcd-path', type=str, required=True, help='Path to the target PCD file.')
    parser.add_argument('--target-vertex-classes', type=str, required=True, help='Path to the target PCD file.')
    parser.add_argument('--target-vertex-class-names', type=str, required=True, help='Path to the target PCD file.')
    parser.add_argument('--target-pcd-ray-origins-path', type=str, default=None, help='Path to the target PCD ray origins file (the rays that produced the target pcd).')
    parser.add_argument('--target-pcd-ray-dests-path', type=str, default=None, help='Path to the ray dests file.')
    parser.add_argument('--object-mesh', type=str, default=None, help='Path to a mesh of objects (not part of target points)')
    parser.add_argument('--polygon-info-path', type=str, required=True, help='Path to the JSON polygon info file.')
    parser.add_argument('--output-dir', type=str, default="polygon_fitting_output", help='Output directory for the fitted polygons.')
    parser.add_argument('--device', type=str, default="auto", help='Device to use for fitting.')
    parser.add_argument('--ray-classes', type=str, default=None, help='If given, the rays will be filtered and windows will be removed', required=False)
    parser.add_argument('--nerfstudio-transforms-file', type=str, default=None, help='Path to the nerf studio transforms file. (JSON)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.scene_type == "scannetpp":
        config = ScannetppConfig()
    elif args.scene_type == "matterport":
        config = MatterportConfig()
    else:
        raise ValueError(f"Scene type {args.scene_type} not supported.")

    # load the stuff and assert that they match the # vertices
    # Load the rectified PLY file
    ply_data = o3d.io.read_triangle_mesh(args.rectified_ply_path)

    np.random.seed(0)
    # def set_deterministic(seed=42):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed) 

    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.enabled = False
    # seed = 42 # any number 
    # set_deterministic(seed=seed)

    # make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # remove all checkpoints
    for f in os.listdir(args.output_dir):
        if f.startswith("fitted_mesh") or f.startswith("sampled_points") or f.startswith("vertices") or f.startswith("transformed_mesh"):
            os.remove(os.path.join(args.output_dir, f))

    # Load the polygon info
    with open(args.polygon_info_path, 'r') as f:
        polygon_info = json.load(f)

    polygon_info = {int(k): v for k, v in polygon_info.items()}
    for k, polygon in polygon_info.items():
        polygon["shared_edges"] = {int(k): v for k, v in polygon["shared_edges"].items()}

    # device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # save transformed version of input mesh
    if args.nerfstudio_transforms_file is not None and os.path.exists(args.nerfstudio_transforms_file):
        copied_mesh = transform_mesh(ply_data, args.nerfstudio_transforms_file)
        if copied_mesh is not None:
            o3d.io.write_triangle_mesh(f"{args.output_dir}/transformed_input_mesh.ply", copied_mesh)
            print(f"Wrote transformed mesh to {args.output_dir}/transformed_input_mesh.ply")

    # load the target PCD vertex classes npy
    target_vertex_classes = np.load(args.target_vertex_classes)
    target_vertex_class_names = np.load(args.target_vertex_class_names)
    # Load the target PCD file
    # try:
    if True:
        _mesh = o3d.io.read_triangle_mesh(args.target_pcd_path)
        area = _mesh.get_surface_area()
        
        
        # filter out connected components with area less than threshold
        min_area = 0.01
        triangle_clusters, cluster_n_triangles, area_per_cluster = _mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        area_per_cluster = np.asarray(area_per_cluster)
        triangle_cluster_areas = area_per_cluster[triangle_clusters]
        triangles_to_remove = np.where(triangle_cluster_areas < min_area)[0]
        _mesh.remove_triangles_by_index(triangles_to_remove)

        if len(_mesh.triangles) > 0 and area > len(_mesh.vertices) / 12_000:
            # sample points from the mesh at fixed density
            n_points = area * 12_000
            assert len(target_vertex_classes) == len(_mesh.vertices), "Target vertex classes and target pcd must have the same length"
            # if len(_mesh.vertex_normals) == 0:
            _mesh.compute_vertex_normals()
            per_vertex_normals = np.array(_mesh.vertex_normals)
            per_face_normals = per_vertex_normals[np.array(_mesh.triangles)].mean(axis=1)
            per_face_classes = target_vertex_classes[np.array(_mesh.triangles)[:, 0]]
            per_face_colors = np.array(_mesh.vertex_colors)[np.array(_mesh.triangles)[:, 0]]
            samples, _ , (colors, classes, normals) =  sample_points_from_mesh(torch.from_numpy(np.array(_mesh.triangles)).to(device).long(), torch.from_numpy(np.array(_mesh.vertices)).to(device).float(), triangle_features=[torch.from_numpy(per_face_colors).to(device).float(), torch.from_numpy(per_face_classes).to(device).float(), torch.from_numpy(per_face_normals).to(device)], num_samples=int(n_points), return_normals=True)
            normals /= normals.norm(dim=-1, keepdim=True)
            classes /= classes.norm(dim=-1, keepdim=True)
            target_vertex_classes = classes.cpu().numpy()
            samples = samples.cpu().numpy()
            normals = normals.cpu().numpy()
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(samples)
            target_pcd.normals = o3d.utility.Vector3dVector(normals)
            target_pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

        else:
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = _mesh.vertices
            target_pcd.colors = _mesh.vertex_colors
            from pathlib import Path
            # if (Path(args.target_pcd_path).parent / "aggregated_normals.npy").exists():
            #     normals = np.load(Path(args.target_pcd_path).parent / "aggregated_normals.npy")
            #     target_pcd.normals = o3d.utility.Vector3dVector(normals)
            # else:
            _mesh.compute_vertex_normals()
            target_pcd.normals = _mesh.vertex_normals
        assert len(np.array(target_pcd.normals)) == len(np.array(target_pcd.points)), "Normals and points must have the same length"
    # except:
    #     target_pcd = o3d.io.read_point_cloud(args.target_pcd_path)
    #     if target_pcd.normals is None:
    #         target_pcd.estimate_normals()

    if args.target_pcd_ray_origins_path is not None and args.target_pcd_ray_dests_path is not None:
        target_pcd_ray_origins = np.load(args.target_pcd_ray_origins_path)
        if args.target_pcd_ray_dests_path.endswith(".npy"):
            target_ray_dests = np.load(args.target_pcd_ray_dests_path)
        else:
            target_ray_dests = np.array(o3d.io.read_point_cloud(args.target_pcd_ray_dests_path).points)
        assert len(target_pcd_ray_origins) == len(target_ray_dests), "Ray origins and points must have the same length"

    
    assert np.all([np.isin(np.concatenate(polygon["contours"]), np.arange(len(ply_data.vertices))).all() for polygon in polygon_info.values()]), "Not all polygons are valid"

    assert len(target_vertex_classes) == len(target_pcd.points), "Target vertex classes and target pcd must have the same length"
    
    # if target classes are one-dimensional: one-hot
    if len(target_vertex_classes.shape) == 1:
        target_vertex_classes = np.arange(len(target_vertex_class_names))[None, :] == target_vertex_classes[:, None]

    assert target_vertex_classes.shape[1] == len(target_vertex_class_names), f"All classes must be valid, {target_vertex_classes.shape[1]} != {len(target_vertex_class_names)}"


    # load the object mesh
    object_mesh = None
    if args.object_mesh is not None:
        object_mesh = o3d.io.read_triangle_mesh(args.object_mesh)
        assert len(np.array(object_mesh.vertices)) > 0, "Object mesh must have vertices if it is given"

    ray_classes = np.load(args.ray_classes)
    if args.ray_classes and "window" in target_vertex_class_names:
        assert len(ray_classes) == len(target_pcd_ray_origins), "Ray classes and ray origins must have the same length"
        dangerous_classes = ['window', 'outdoor', 'dangerous_noise', 'unknown']
        
        window_ix = np.where(np.isin(target_vertex_class_names, dangerous_classes))[0]
        is_not_window = ~np.isin(ray_classes, window_ix)
        print(f'window index: {window_ix}, ratio of non-window rays: {is_not_window.mean()} N window rays: {len(is_not_window) - is_not_window.sum()}')
        target_pcd_ray_origins = target_pcd_ray_origins[is_not_window]
        target_ray_dests = target_ray_dests[is_not_window]        
    else:
        print("No ray classes given or no window class in target vertex class names")

    print(np.unique(ray_classes.astype(int)))
    from collections import Counter
    print(Counter(np.array(target_vertex_class_names)[ray_classes[ray_classes != 255].astype(int)]))

    
    # Remove the rays that intersect already at this point
    # vertices = torch.from_numpy(np.array(ply_data.vertices))
    # triangles = torch.from_numpy(np.array(ply_data.triangles))

    # does_intersect, primitive_ids, intersection_points, len_of_overlap = compute_ray_mesh_intersections_ray_tracing(vertices, triangles, torch.from_numpy(target_pcd_ray_origins), torch.from_numpy(target_ray_dests), margin=0.01)
    # print(f"Number of intersections: {does_intersect.sum()} (ratio: {does_intersect.float().mean()})")
    # target_pcd_ray_origins = target_pcd_ray_origins[~does_intersect]
    # target_ray_dests = target_ray_dests[~does_intersect]

    # save sampled points with normals as color
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(target_pcd.points)
    # normals = np.array(target_pcd.normals)
    # colors = (normals + 1) / 2
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud(f"{args.output_dir}/target_pcd_with_normals.ply", pcd)

    # Fit the polygons
    fit_polygon_collection(config, ply_data, target_pcd, polygon_info, out_dir=args.output_dir, device=device, nerfstudio_transforms_file=args.nerfstudio_transforms_file, target_pcd_ray_origins=target_pcd_ray_origins, target_ray_dests=target_ray_dests, target_vertex_classes=target_vertex_classes, target_vertex_class_names=target_vertex_class_names, object_mesh=object_mesh)