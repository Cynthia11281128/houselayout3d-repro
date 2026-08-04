"""authored by ChatGPT, ported from https://github.com/CGAL/cgal-swig-bindings/blob/main/examples/python/polygonal_triangulation.py"""

from typing import List
from CGAL.CGAL_Kernel import Point_2, Polygon_2
from CGAL.CGAL_Triangulation_2 import Constrained_triangulation_2, Triangulation_2
import numpy as np
from shapely import GeometryCollection, LineString, MultiLineString
import shapely
from shapely.validation import make_valid
from shapely import Polygon, union_all, coverage_union_all, MultiPolygon, Point


def filter_min_area(cgal_polygons, numpy_polygons, min_area = 1e-10):
    filtered_cgal_polygons = []
    filtered_numpy_polygons = []
    for cgal_polygon, numpy_polygon in zip(cgal_polygons, numpy_polygons):
        if abs(cgal_polygon.area()) > min_area:
            filtered_cgal_polygons.append(cgal_polygon)
            filtered_numpy_polygons.append(numpy_polygon)
    return filtered_cgal_polygons, filtered_numpy_polygons
    

def split_polygon_at_self_intersections(points : list):
    
    if tuple(points[0]) == tuple(points[-1]):
        points = points[:-1]   

    polygons = []   
    while len(points) > 0:
        unique_points, point_indices, counts = np.unique(points, axis=0, return_counts=True, return_index=True)
        if (counts > 1).any():
            cutpoint = unique_points[np.where(counts > 1)[0][0]]
            cutpoint_mask = np.all(points == cutpoint, axis=1)
            first_occurrence = np.where(cutpoint_mask)[0][0]
            last_occurrence = np.where(cutpoint_mask)[0][-1]
            head = points[:first_occurrence]
            tail = points[last_occurrence:]
            polygons.append(np.concatenate([tail, head]))
            points = points[first_occurrence:last_occurrence]
        else:
            polygons.append(points)
            break
    
    outer_polygons = []
    _polygons = []
    for outer_polygon in polygons:
        if len(outer_polygon) < 3:
            continue
        if tuple(outer_polygon[0]) == tuple(outer_polygon[-1]):
            outer_polygon = outer_polygon[:-1]
        # Create a polygon from the list of points
        outer_polygon_cgal = Polygon_2()
        for (x, y) in outer_polygon:
            outer_polygon_cgal.push_back(Point_2(x, y))
        
        # Ensure the polygon is simple and oriented counter-clockwise
        if outer_polygon_cgal.is_clockwise_oriented():
            outer_polygon_cgal.reverse_orientation()
        outer_polygons.append(outer_polygon_cgal)    
        _polygons.append(outer_polygon)
    return outer_polygons, _polygons



def triangulate_polygon(points : list, holes : list = [], correct_holes=True, delete_small=True):
    """
    Triangulates a polygon given its vertices using CGAL.

    Parameters:
    points (list of tuples): A list of (x, y) tuples representing the polygon vertices in order.

    Returns:
    list of tuples: A list of triangles, each triangle is represented by three points (x, y).
    """
    if correct_holes:
        outer_polygons, outer_polygon_points = split_polygon_at_self_intersections(points)
    else:
        outer_polygons, outer_polygon_points = [], []
        for outer_polygon in points:
            new_outer_polygons, new_outer_polygon_points = split_polygon_at_self_intersections(outer_polygon)
            outer_polygons.extend(new_outer_polygons)
            outer_polygon_points.extend(new_outer_polygon_points)
            
    if delete_small:
        outer_polygons, outer_polygon_points = filter_min_area(outer_polygons, outer_polygon_points)
    # # Create a polygon from the list of points
    # outer_polygon = Polygon_2()
    # for (x, y) in points:
    #     outer_polygon.push_back(Point_2(x, y))
    #     
    # # Ensure the polygon is simple and oriented counter-clockwise
    # if outer_polygon.is_clockwise_oriented():
    #     outer_polygon.reverse_orientation()

    hole_polygons = []
    hole_polygons_points = []
    for hole in holes:
        # hole_polygon = Polygon_2()
        # for (x, y) in hole:
        #     hole_polygon.push_back(Point_2(x, y))
        # if hole_polygon.is_clockwise_oriented():
        #     hole_polygon.reverse_orientation()
        # hole_polygons.append(hole_polygon)
        hole_poly, hole_poly_points = split_polygon_at_self_intersections(hole)
        # hole_poly, hole_poly_points = filter_min_area(hole_poly, hole_poly_points)
        for hole_p, hole_p_points in zip(hole_poly, hole_poly_points):
            if correct_holes and all([outer_polygon.oriented_side(Point_2(*hole_p_points[0])) == -1 for outer_polygon in outer_polygons]):
                outer_polygons.append(hole_p)
                outer_polygon_points.append(hole_p_points)
            else:
                hole_polygons.append(hole_p)
                hole_polygons_points.append(hole_p_points)

    # for hole_polygon, hole_polygon_points in zip(hole_polygons, hole_polygons_points):
    #     if all(outer_polygon.oriented_side(Point_2(*hole_polygon_points[0])) == 1 for outer_polygon in outer_polygons):
    #         outer_polygons.append(hole_polygon)
    #         outer_polygon_points.append(hole_polygon_points)
    #         break
    
        # hole_polygons.extend(hole_poly)
        # hole_polygons_points.extend(hole_poly_points)

        # check if the hole is inside an outer polygon
        # point_inside = hole_poly.bool_inside(outer_polygon)

    if len(outer_polygons) == 0:
        return np.array([]), [], []

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for outer_polygon in outer_polygons:
    #     outer_polygon_points = np.array([[p.x(), p.y()] for p in outer_polygon.vertices()])
    #     ax.plot(outer_polygon_points[:, 0], outer_polygon_points[:, 1], 'o-', color='blue')
    # for hole in holes:
    #     print()
    #     hole_points = np.array(hole)
    #     ax.plot(hole_points[:, 0], hole_points[:, 1], 'o-', color='red')
    # plt.savefig("triangulation3.png")
    # plt.close()

    # Set up the triangulation object
    triangulation = Constrained_triangulation_2()

    # all_points = list(points) + [p for hole in holes for p in hole]
    all_points = np.concatenate(outer_polygon_points + hole_polygons_points, axis=0)

    # create a vertex for every point
    unique_points, unique_inverse = np.unique(all_points, axis=0, return_inverse=True)
    vertices = [triangulation.insert(Point_2(*p)) for p in unique_points]
    vertex2index = {v: i for i, v in enumerate(vertices)}

    if len(holes) == 0:
        # Insert polygon edges as constraints in the triangulation
        for i in range(len(all_points)):
            # p1 = Point_2(*points[i])
            # p2 = Point_2(*points[(i + 1) % len(points)])
            # points2index[(p1.x(), p1.y())] = i
            # triangulation.insert_constraint(p1, p2)
            v1 = vertices[unique_inverse[i]]
            v2 = vertices[unique_inverse[(i + 1) % len(all_points)]]
            triangulation.insert_constraint(v1, v2)
    else:
        offset = 0
        for polygon_ix, poly in enumerate(outer_polygons + hole_polygons):
            n_vertices = len(outer_polygon_points[polygon_ix]) if polygon_ix < len(outer_polygon_points) else len(hole_polygons_points[polygon_ix - len(outer_polygon_points)])
            for i in range(n_vertices):
                v1 = vertices[unique_inverse[offset + i]]
                v2 = vertices[unique_inverse[offset + ((i + 1) % n_vertices)]]
                triangulation.insert_constraint(v1, v2)
            offset += n_vertices

    

    additional_vertices = [v for v in triangulation.finite_vertices() if v not in vertex2index]
    for v in additional_vertices:
        vertex2index[v] = len(vertex2index)

    new_points = [(v.point().x(), v.point().y()) for v in vertices] + [(v.point().x(), v.point().y()) for v in additional_vertices]


    # Collect the triangles from the triangulation
    triangles = []
    triangle_ixes = []
    for face in triangulation.finite_faces():
        # Get the vertices of the face (triangle)
        v1 = face.vertex(0)
        v2 = face.vertex(1)
        v3 = face.vertex(2)

        p1 = (v1.point().x(), v1.point().y())
        p2 = (v2.point().x(), v2.point().y())
        p3 = (v3.point().x(), v3.point().y())

        # p1 = (face.vertex(0).point().x(), face.vertex(0).point().y())
        # p2 = (face.vertex(1).point().x(), face.vertex(1).point().y())
        # p3 = (face.vertex(2).point().x(), face.vertex(2).point().y())
        
        # check whether the face is inside the polygon

        # Check if the face is inside the polygon using the centroid method @TODO this is not correct!?
        centroid = Point_2((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)

        
        # import matplotlib.pyplot as plt
        # plt.plot(*zip(*points), 'o-')
        # plt.gca().set_aspect('equal', adjustable='box')
        # triangle = np.array([p1, p2, p3, p1])
        # plt.plot(triangle[:, 0], triangle[:, 1], 'r', alpha=0.5)
        # # for t in triangles:
        # #     plt.plot([t[0][0], t[1][0], t[2][0], t[0][0]], [t[0][1], t[1][1], t[2][1], t[0][1]], 'b', alpha=0.5)
        # plt.scatter(centroid.x(), centroid.y(), c='yellow')
        # plt.title(f"Is inside: {[polygon.oriented_side(centroid) == -1 for polygon in outer_polygons]}")
        # plt.savefig("triangulation.png")
        # plt.close()

        # Only add the triangle if the centroid is inside the polygon
        # if its outside of all polygons continue
        if correct_holes and all([polygon.oriented_side(centroid) == -1 for polygon in outer_polygons]):
            continue
        # if its inside of a hole continue
        if correct_holes and any([hole_polygon.oriented_side(centroid) == 1 for hole_polygon in hole_polygons]):
            continue

        
        # plt.plot(triangle[:, 0], triangle[:, 1], 'r', alpha=0.5)
        # triangle_2 = [new_points[vertex2index[v1]], new_points[vertex2index[v2]], new_points[vertex2index[v3]]]
        # plt.plot([triangle_2[0][0], triangle_2[1][0], triangle_2[2][0], triangle_2[0][0]], [triangle_2[0][1], triangle_2[1][1], triangle_2[2][1], triangle_2[0][1]], 'b', alpha=0.5)
        # make sure the triangle is oriented counter-clockwise
        # if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) < 0:
        #     p2, p3 = p3, p2
        #     v2, v3 = v3, v2

        triangles.append((p1, p2, p3))
        triangle_ixes.append((vertex2index[v1], vertex2index[v2], vertex2index[v3]))

        assert np.isclose(p1, new_points[vertex2index[v1]]).all()
        assert np.isclose(p2, new_points[vertex2index[v2]]).all()
        assert np.isclose(p3, new_points[vertex2index[v3]]).all()

        # knn_nearest_neighbors = triangulation.nearest_neighbors(p1, 1)
        # i1 = points2index[p1]
        # i2 = points2index[p2]
        # i3 = points2index[p3]
        # triangle_ixes.append((i1, i2, i3))

    
    # import matplotlib.pyplot as plt
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.plot(*zip(*points), 'o-')
    # for t, triangle in enumerate(triangle_ixes):
    #     if t !=  2:
    #         continue
    #     triangle = np.stack(new_points, axis=0)[list(triangle)]
    #     # assert np.isclose(triangle, triangles[t]).all()
    #     triangle = np.concatenate([triangle, [triangle[0]]])
    #     # triangle = np.array([new_points_2d[t] for t in triangle])
    #     plt.plot(triangle[:, 0], triangle[:, 1], 'b')

    # triangle_2 = triangles[2]
    # plt.plot([triangle_2[0][0], triangle_2[1][0], triangle_2[2][0], triangle_2[0][0]], [triangle_2[0][1], triangle_2[1][1], triangle_2[2][1], triangle_2[0][1]], 'r')
    # plt.show()

    return np.stack(new_points, axis=0), triangles, triangle_ixes



def get_faces_of_constrained_delaunay(edges : np.ndarray):
    """
    Triangulates a polygon given its vertices using CGAL.

    Parameters:
    points (list of tuples): A list of (x, y) tuples representing the polygon vertices in order.

    Returns:
    list of tuples: A list of triangles, each triangle is represented by three points (x, y).
    """
    # Set up the triangulation object
    triangulation = Constrained_triangulation_2()

    for (v1, v2) in edges:
        v1 = triangulation.insert(Point_2(*v1))
        v2 = triangulation.insert(Point_2(*v2))
        triangulation.insert_constraint(v1, v2)
        

    # Collect the triangles from the triangulation
    triangles = []
    for face in triangulation.finite_faces():
        # Get the vertices of the face (triangle)
        v1 = face.vertex(0)
        v2 = face.vertex(1)
        v3 = face.vertex(2)

        p1 = (v1.point().x(), v1.point().y())
        p2 = (v2.point().x(), v2.point().y())
        p3 = (v3.point().x(), v3.point().y())

        triangles.append((p1, p2, p3))


    return np.array(triangles)



def adjacency_list_to_connected_contour(adjacency_list: dict) -> list:
    vertices_that_still_have_neighbors = [k for k in adjacency_list.keys() if len(adjacency_list[k]) > 0]
    if len(vertices_that_still_have_neighbors) == 0:
        return []
    vertex_list = [vertices_that_still_have_neighbors[0]]
    # done[vertex_list[0]] = True
    while True:
        # assert len(adjacency_list[vertex_list[-1]]) % 2 == 0, f"Vertex {vertex_list[-1]} has {len(adjacency_list[vertex_list[-1]])} adjacent vertices"
        current_vertex = vertex_list[-1]
        if len(adjacency_list[current_vertex]) == 0:
            # assert vertex_list[0] == vertex_list[-1] or len(vertex_list) == 2, f"Vertex {vertex_list[-1]} has no neighbors but is not the start vertex"
            vertices_that_still_have_neighbors = [k for k in adjacency_list.keys() if len(adjacency_list[k]) > 0 and k in vertex_list]
            if len(vertices_that_still_have_neighbors) > 0:
                # roll the vertex list so that it starts with a vertex that still has neighbors
                # print(vertex_list, current_vertex)
                current_vertex = vertices_that_still_have_neighbors[0]
                vertex_list = vertex_list[vertex_list.index(current_vertex):] + vertex_list[1:vertex_list.index(current_vertex)]
                # print(vertex_list) 
            else:
                # print(f"Breaking because of vertex {vertex_list[-1]} that has no neighbors")
                break 
        neighbor = adjacency_list[current_vertex].pop()
        while neighbor == current_vertex and len(adjacency_list[current_vertex]) > 0:
            neighbor = adjacency_list[current_vertex].pop()
        if neighbor == current_vertex:
            break
        # # also remove the edge from the neighbor to the current vertex
        adjacency_list[neighbor].remove(current_vertex)
        vertex_list.append(neighbor)

    return vertex_list

def triangulate_polygon_from_edges2(edges : np.array):
    
    adjacency_list = {i: set() for i in np.unique(edges.flatten())}
    for (v1, v2) in edges:
        adjacency_list[v1].add(v2)
        adjacency_list[v2].add(v1)

    # print(np.unique(list(map(len, adjacency_list.values())), return_counts=True))

    vertex_lists = [adjacency_list_to_connected_contour(adjacency_list)]

    while len([k for k in adjacency_list.keys() if len(adjacency_list[k]) > 0]) > 0:
        vertex_lists.append(adjacency_list_to_connected_contour(adjacency_list))
    
    return sorted(vertex_lists, key=lambda x: len(x), reverse=True)
    



def triangulate_polygon_from_edges(edges : list):
    """
    Triangulates a polygon given its vertices using CGAL.

    Parameters:
        edges (np.ndarray): An array of shape (n_edges, 2, 2) representing the polygon edge coordinates in 2d

    Returns:
        list of tuples: A list of triangles, each triangle is represented by three points (x, y).
    """
    # Set up the triangulation object
    triangulation = Constrained_triangulation_2()

    # create a vertex for every point
    unique_points, unique_inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    unique_inverse = unique_inverse.reshape(-1, 2, 2)
    vertices = [triangulation.insert(Point_2(*p)) for p in unique_points]
    vertex2index = {v: i for i, v in enumerate(vertices)}

    # Insert polygon edges as constraints in the triangulation
    for (v1, v2) in vertices[unique_inverse]:
        triangulation.insert_constraint(v1, v2)

    
    additional_vertices = [v for v in triangulation.finite_vertices() if v not in vertex2index]
    for v in additional_vertices:
        vertex2index[v] = len(vertex2index)

    new_points = [(v.point().x(), v.point().y()) for v in vertices] + [(v.point().x(), v.point().y()) for v in additional_vertices]


    # Collect the triangles from the triangulation
    triangles = []
    triangle_ixes = []
    for face in triangulation.finite_faces():
        # Get the vertices of the face (triangle)
        v1 = face.vertex(0)
        v2 = face.vertex(1)
        v3 = face.vertex(2)

        p1 = (v1.point().x(), v1.point().y())
        p2 = (v2.point().x(), v2.point().y())
        p3 = (v3.point().x(), v3.point().y())

        # check whether the face is inside the polygon

        # Check if the face is inside the polygon using the centroid method
        centroid = Point_2((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)


        # Only add the triangle if the centroid is inside the polygon
        # if outer_polygon.oriented_side(centroid) == -1:
        #     continue
        # if any([hole_polygon.oriented_side(centroid) == 1 for hole_polygon in hole_polygons]):
            # continue

        
        # plt.plot(triangle[:, 0], triangle[:, 1], 'r', alpha=0.5)
        # triangle_2 = [new_points[vertex2index[v1]], new_points[vertex2index[v2]], new_points[vertex2index[v3]]]
        # plt.plot([triangle_2[0][0], triangle_2[1][0], triangle_2[2][0], triangle_2[0][0]], [triangle_2[0][1], triangle_2[1][1], triangle_2[2][1], triangle_2[0][1]], 'b', alpha=0.5)
        # make sure the triangle is oriented counter-clockwise
        # if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) < 0:
        #     p2, p3 = p3, p2
        #     v2, v3 = v3, v2

        triangles.append((p1, p2, p3))
        triangle_ixes.append((vertex2index[v1], vertex2index[v2], vertex2index[v3]))

        assert np.isclose(p1, new_points[vertex2index[v1]]).all()
        assert np.isclose(p2, new_points[vertex2index[v2]]).all()
        assert np.isclose(p3, new_points[vertex2index[v3]]).all()

        # knn_nearest_neighbors = triangulation.nearest_neighbors(p1, 1)
        # i1 = points2index[p1]
        # i2 = points2index[p2]
        # i3 = points2index[p3]
        # triangle_ixes.append((i1, i2, i3))

    
    # import matplotlib.pyplot as plt
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.plot(*zip(*points), 'o-')
    # for t, triangle in enumerate(triangle_ixes):
    #     if t !=  2:
    #         continue
    #     triangle = np.stack(new_points, axis=0)[list(triangle)]
    #     # assert np.isclose(triangle, triangles[t]).all()
    #     triangle = np.concatenate([triangle, [triangle[0]]])
    #     # triangle = np.array([new_points_2d[t] for t in triangle])
    #     plt.plot(triangle[:, 0], triangle[:, 1], 'b')

    # triangle_2 = triangles[2]
    # plt.plot([triangle_2[0][0], triangle_2[1][0], triangle_2[2][0], triangle_2[0][0]], [triangle_2[0][1], triangle_2[1][1], triangle_2[2][1], triangle_2[0][1]], 'r')
    # plt.show()

    return np.stack(new_points, axis=0), triangles, triangle_ixes


def geometry_to_edges(geometry):
    if isinstance(geometry, LineString):
        boundary_vertices = np.array(geometry.coords)
        return np.stack([boundary_vertices[:-1], boundary_vertices[1:]], axis=1)
    elif isinstance(geometry, MultiLineString):
        if geometry.is_empty:
            return np.empty((0, 2, 2))
        boundary_edges = []
        for line in geometry.geoms:
            line_vertices = np.array(line.coords)
            boundary_edges.append(np.stack([line_vertices[:-1], line_vertices[1:]], axis=1))
        return np.concatenate(boundary_edges, axis=0)
    elif isinstance(geometry, Polygon):
        return geometry_to_edges(geometry.boundary)
    elif isinstance(geometry, MultiPolygon):
        return np.concatenate([geometry_to_edges(polygon) for polygon in geometry.geoms], axis=0)
    elif isinstance(geometry, GeometryCollection):
        if len(geometry.geoms) == 0:
            return np.empty((0, 2, 2))
        return np.concatenate([geometry_to_edges(geom) for geom in geometry.geoms], axis=0)
    elif isinstance(geometry, Point):
        return np.empty((0, 2, 2))
    else:
        print("Geometry is of type", type(geometry))
        return np.empty((0, 2, 2))


def triangles_to_polygon_union(triangles: List[np.ndarray], simplify=True, min_area=0.01, delete_small=True):
    input_triangles = []
    for triangle in triangles:
        if len(triangle) < 3 or np.isnan(triangle).any():
            continue
        try:
            input_triangles.append(Polygon(list(triangle)))
        except shapely.errors.GEOSException:
            try: 
                input_triangles.append(Polygon(list(triangle)).buffer(0))
            except shapely.errors.GEOSException:
                pass
    input_triangles = [make_valid(p) if not p.is_valid else p for p in input_triangles]
    input_triangles = [p for p in input_triangles if p.area > 1e-8 and p.is_valid and isinstance(p, Polygon)]

    try:    
        polygon_union = union_all(input_triangles, grid_size=1e-5)
    except shapely.errors.GEOSException:
        polygon_union = union_all(input_triangles)
    
    area = polygon_union.area
    if area < min_area and delete_small:
        return [], []

    if simplify:
        polygon_union = polygon_union.simplify(tolerance=1e-3)

    if not polygon_union.is_valid:
        polygon_union = make_valid(polygon_union)
    return polygon_union, input_triangles


def triangulate_polygon_from_triangles(triangles: List[np.ndarray], simplify=True, min_area=0.01, delete_small=True, plot=False):
    """Take a list of triangles and return a new list of triangles based on the intersection of the input triangles"""
    

    polygon_union, input_triangles = triangles_to_polygon_union(triangles, simplify=simplify, min_area=min_area, delete_small=delete_small)
    if len(input_triangles) == 0:
        return [], [], []
        # print("Simplified polygon is invalid")
        # polygon_union = polygon_union_rdp

    boundary_edges = geometry_to_edges(polygon_union)
    assert boundary_edges.shape[1] == 2 and boundary_edges.shape[2] == 2

    if plot:
        import matplotlib.pyplot as plt
        # plt.plot(boundary_edges[:, 0], boundary_edges[:, 1], color='black', linewidth=0.5)
        for edge in boundary_edges:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='black', linewidth=0.5)
        plt.scatter(boundary_edges[:, 0, 0], boundary_edges[:, 0,  1], color='black', s=1)
        plt.scatter(boundary_edges[:, 1, 0], boundary_edges[:, 1,  1], color='black', s=1)
        # plt.plot(np.array(triangle_pts_2d)[:, 0], np.array(triangle_pts_2d)[:, 1], color='green', linewidth=1)
        plt.savefig("union.png")
        plt.close()

    new_triangles_2d = get_faces_of_constrained_delaunay(boundary_edges)
    
    if len(new_triangles_2d) == 0:
        return [], [], []
    
    triangle_centers = np.mean(np.array(new_triangles_2d), axis=1)
    polygons_tree = shapely.STRtree(input_triangles)
    inner_triangles_2d = []
    for triangle, triangle_center in zip(new_triangles_2d, triangle_centers):
        pt = Point(*triangle_center)
        possible_polygons = polygons_tree.query(pt)
        if any(input_triangles[poly].contains(pt) for poly in possible_polygons) > 0:
            inner_triangles_2d.append(triangle)
    # center_points = [Point(*p) for p in np.mean(new_triangles_2d, axis=1)]
    # shapely.prepare(polygon_union)
    # inner_triangles_2d = new_triangles_2d[shapely.contains(polygon_union, center_points)]


    inner_triangles_2d = np.array(inner_triangles_2d)
    points_2d, ixes = np.unique(inner_triangles_2d.reshape(-1, 2), axis=0, return_inverse=True)
    inner_triangle_ixes = ixes.reshape(-1, 3)

    return points_2d, inner_triangles_2d, inner_triangle_ixes


def triangulate_polygon_from_triangles_shapely(triangles: List[np.ndarray], simplify=False, min_area=0.01, delete_small=True, plot=False):
    """Take a list of triangles and return a new list of triangles based on the intersection of the input triangles"""
    # from CGAL.CGAL_Kernel import Point_2, Polygon_2, Polyg
    # from CGAL.CGAL_Polygon_2 import Polygon_2
    # from CGAL.CGAL_Boolean_set_operations_2 import Polygon_set_2, polygons_with_holes

    # import matplotlib.pyplot as plt

    # triangles_closed = np.concatenate([triangles, triangles[:, :1]], axis=1)
    

    # for triangle in triangles_closed:
    #     plt.plot(triangle[:, 0], triangle[:, 1], linewidth=0.5)
    # plt.savefig("triangulation.png")
    # plt.close()
    
    polygons = [Polygon(list(triangle)) for triangle in triangles]
    polygons = [make_valid(p) if not p.is_valid else p for p in polygons]

    polygon_union = union_all(polygons)

    if simplify:
        polygon_union = polygon_union.simplify(tolerance=1e-3)
            # print("Simplified polygon is invalid")
        # polygon_union = polygon_union_rdp
    
    if not polygon_union.is_valid:
        polygon_union = make_valid(polygon_union)

    area = polygon_union.area
    if area < min_area and delete_small:
        return [], [], []

    # plot the union
    numpy_polygons = []
    numpy_holes = []
    if isinstance(polygon_union, Polygon):
        polygon_shapely = [polygon_union]
    elif isinstance(polygon_union, MultiPolygon):
        polygon_shapely = polygon_union.geoms
    else:
        polygon_shapely = []
        for geom in polygon_union.geoms:
            if isinstance(geom, Polygon):
                polygon_shapely.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygon_shapely.extend(geom.geoms)
            elif isinstance(geom, LineString):
                pass
            else:
                print("Ignoring geometry of type", type(geom))

    for polygon in polygon_shapely:
        if isinstance(polygon, Polygon):
            numpy_polygon = np.array(polygon.exterior.xy).T
            if len(numpy_polygon) < 4:
                continue
            numpy_polygons.append(numpy_polygon)
            for hole in polygon.interiors:
                numpy_hole = np.array(hole.xy).T
                if len(numpy_hole) < 4:
                    continue
                numpy_holes.append(numpy_hole)
        else:
            print("Ignoring polygon of type", type(polygon))
    
    if plot:
        import matplotlib.pyplot as plt
        for polygon in numpy_polygons:
            plt.plot(polygon[:, 0], polygon[:, 1], color='blue', linewidth=0.1)
        for hole in numpy_holes:
            plt.plot(hole[:, 0], hole[:, 1], color='red', linewidth=0.1)
        # plt.plot(np.array(triangle_pts_2d)[:, 0], np.array(triangle_pts_2d)[:, 1], color='green', linewidth=1)
        plt.savefig("union.png")
        plt.close()

    new_points_2d, triangle_points_2d, triangle_ixes = triangulate_polygon(numpy_polygons, numpy_holes, correct_holes=False, delete_small=False)

    if len(triangle_points_2d) == 0 or (len(triangle_ixes) == 1 and area < min_area * 2 and delete_small):
        return [], [], []
    triangle_centers = np.mean(np.array(triangle_points_2d), axis=1)
    
    inside_triangles_ixes, inside_triangle_points_2d = [], []
    for triangle_center, triangle, triangle_pts_2d in zip(triangle_centers, triangle_ixes, triangle_points_2d):
        try:
            if polygon_union.contains(Point(*triangle_center)) and Polygon(triangle_pts_2d).area > 1e-6:
                inside_triangles_ixes.append(triangle)
                inside_triangle_points_2d.append(triangle_pts_2d)
        except shapely.errors.GEOSException:
            from matplotlib import path as mplPath
            print("GEOSException")
            if any([mplPath.Path(list(_triangle) + [_triangle[0]]).contains_point(triangle_center) for _triangle in triangles]):
                inside_triangles_ixes.append(triangle)
                inside_triangle_points_2d.append(triangle_pts_2d)
        # inside_triangles_ixes, inside_triangle_points_2d = zip(*[(triangle, triangle_pts_2d) for triangle_center, triangle, triangle_pts_2d in zip(triangle_centers, triangle_ixes, triangle_points_2d) if polygon_union.contains(Point(*triangle_center))])
    # except shapely.errors.GEOSException:
    #     return [], [], []

    # for triangle in inside_triangles_ixes:
    #     for i in range(3):
    #         plt.plot([new_points_2d[triangle[i % 3]][0], new_points_2d[triangle[(i + 1) % 3]][0]], [new_points_2d[triangle[i % 3]][1], new_points_2d[triangle[(i + 1) % 3]][1]], color='black', linewidth=0.1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("union.png")
    # plt.close()


    return new_points_2d, inside_triangle_points_2d, inside_triangles_ixes