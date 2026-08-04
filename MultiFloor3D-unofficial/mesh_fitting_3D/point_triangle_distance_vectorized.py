from matplotlib import pyplot as plt
import torch

class Edge3:
    def __init__(self, a, b):
        # a and b are tensors of shape (k, 3)
        self.A = a  # Shape: (k, 3)
        self.B = b  # Shape: (k, 3)
        self.Delta = b - a  # Shape: (k, 3)

    def point_at(self, t):
        # t is a tensor of shape (k,)
        return self.A + t.unsqueeze(-1) * self.Delta  # Shape: (k, 3)

    def project(self, p):
        # p is a tensor of shape (k, 3)
        return torch.sum((p - self.A) * self.Delta, dim=-1) / (torch.sum(self.Delta * self.Delta, dim=-1) + 1e-10)  # Shape: (k,)


class Plane:
    def __init__(self, point, direction):
        # point and direction are tensors of shape (k, 3)
        self.Point = point  # Shape: (k, 3)
        self.Direction = direction  # Shape: (k, 3)

    def is_above(self, q):
        # q is a tensor of shape (k, 3)
        return torch.sum(self.Direction * (q - self.Point), dim=-1) > 0  # Shape: (k,)


class Triangle:
    def __init__(self, a, b, c, triangle_normals):
        # a, b, c are tensors of shape (k, 3)
        self.EdgeAb = Edge3(a, b)  # Shape: (k, 3)
        self.EdgeBc = Edge3(b, c)  # Shape: (k, 3)
        self.EdgeCa = Edge3(c, a)  # Shape: (k, 3)
        # self.TriNorm = torch.cross(a - b, a - c, dim=-1)  # Shape: (k, 3)
        self.TriNorm = triangle_normals  # Shape: (k, 3)

        # Precompute planes
        self.PlaneAb = Plane(self.EdgeAb.A, torch.cross(self.TriNorm, self.EdgeAb.Delta, dim=-1))  # Shape: (k, 3)
        self.PlaneBc = Plane(self.EdgeBc.A, torch.cross(self.TriNorm, self.EdgeBc.Delta, dim=-1))  # Shape: (k, 3)
        self.PlaneCa = Plane(self.EdgeCa.A, torch.cross(self.TriNorm, self.EdgeCa.Delta, dim=-1))  # Shape: (k, 3)

    def closest_point_to(self, p):
        # p is a tensor of shape (k, 3)
        # Projections onto edges
        uab = self.EdgeAb.project(p)  # Shape: (k,)
        uca = self.EdgeCa.project(p)  # Shape: (k,)
        ubc = self.EdgeBc.project(p)  # Shape: (k,)

        # Check if the point is closest to a vertex
        mask_a = (uca > 1) & (uab < 0)  # Shape: (k,)
        mask_b = (uab > 1) & (ubc < 0)  # Shape: (k,)
        mask_c = (ubc > 1) & (uca < 0)  # Shape: (k,)

        # Closest point is a vertex
        closest_point = torch.where(
            mask_a.unsqueeze(-1), self.EdgeAb.A,
            torch.where(
                mask_b.unsqueeze(-1), self.EdgeBc.A,
                torch.where(
                    mask_c.unsqueeze(-1), self.EdgeCa.A, p
                )
            )
        )  # Shape: (k, 3)

        # Check if the point is closest to an edge
        mask_ab = (uab >= 0) & (uab <= 1) & (~self.PlaneAb.is_above(p))  # Shape: (k,)
        mask_bc = (ubc >= 0) & (ubc <= 1) & (~self.PlaneBc.is_above(p))  # Shape: (k,)
        mask_ca = (uca >= 0) & (uca <= 1) & (~self.PlaneCa.is_above(p))  # Shape: (k,)

        # Closest point is on an edge
        closest_point = torch.where(
            mask_ab.unsqueeze(-1), self.EdgeAb.point_at(uab),
            torch.where(
                mask_bc.unsqueeze(-1), self.EdgeBc.point_at(ubc),
                torch.where(
                    mask_ca.unsqueeze(-1), self.EdgeCa.point_at(uca), closest_point
                )
            )
        )  # Shape: (k, 3)

        # If the point is inside the triangle, project onto the plane
        mask_inside = ~(mask_a | mask_b | mask_c | mask_ab | mask_bc | mask_ca)  # Shape: (k,)
        if mask_inside.any():
            # Project onto the triangle's plane
            plane_normal = self.TriNorm  # Shape: (k, 3)
            plane_point = self.EdgeAb.A  # Shape: (k, 3)
            t = torch.sum((plane_point - p) * plane_normal, dim=-1) / (torch.sum(plane_normal * plane_normal, dim=-1) + 1e-10)  # Shape: (k,)
            closest_point = torch.where(
                mask_inside.unsqueeze(-1), p + t.unsqueeze(-1) * plane_normal, closest_point
            )  # Shape: (k, 3)

        return closest_point  # Shape: (k, 3)


def compute_distances(points, triangles, triangle_normals):
    # points: Tensor of shape (k, 3)
    # triangles: Tensor of shape (k, 3, 3), where each triangle is defined by 3 vertices
    a = triangles[:, 0, :]  # Shape: (k, 3)
    b = triangles[:, 1, :]  # Shape: (k, 3)
    c = triangles[:, 2, :]  # Shape: (k, 3)

    # Create a Triangle object for each batch
    triangle_objects = Triangle(a, b, c, triangle_normals)  # Shape: (k,)

    # Compute the closest points
    closest_points = triangle_objects.closest_point_to(points)  # Shape: (k, 3)

    # Compute the distances
    distances = torch.norm(points - closest_points, dim=-1)  # Shape: (k,)

    return distances, closest_points


def compute_distances_batch(points_batch, triangles, triangle_normals):
    # points_batch: Tensor of shape (batch_size, 3)
    # triangles: Tensor of shape (N, 3, 3), where each triangle is defined by 3 vertices
    batch_size = points_batch.shape[0]
    N = triangles.shape[0]
    
    # Expand points and triangles to compute distances in a batched manner
    points_expanded = points_batch.unsqueeze(1).expand(batch_size, N, 3)  # Shape: (batch_size, N, 3)
    triangles_expanded = triangles.unsqueeze(0).expand(batch_size, N, 3, 3)  # Shape: (batch_size, N, 3, 3)
    triangle_normals_expanded = triangle_normals.unsqueeze(0).expand(batch_size, N, 3)  # Shape: (batch_size, N, 3)

    # Flatten the batch dimensions to use the existing compute_distances function
    points_flat = points_expanded.reshape(batch_size * N, 3)  # Shape: (batch_size * N, 3)
    triangles_flat = triangles_expanded.reshape(batch_size * N, 3, 3)  # Shape: (batch_size * N, 3, 3)
    triangle_normals_flat = triangle_normals_expanded.reshape(batch_size * N, 3)  # Shape: (batch_size * N, 3)

    # Compute distances for all point-triangle pairs
    distances_flat, closest_points_flat = compute_distances(points_flat, triangles_flat, triangle_normals_flat)  # Shape: (batch_size * N,)
    
    # Reshape distances back to (batch_size, N)
    distances = distances_flat.reshape(batch_size, N)  # Shape: (batch_size, N)
    closest_points = closest_points_flat.reshape(batch_size, N, 3)  # Shape: (batch_size, N, 3)
    
    return distances, closest_points

@torch.jit.script
def find_k_closest_triangles(points : torch.Tensor, triangles : torch.Tensor, triangle_normals : torch.Tensor, k : int):
    # points: Tensor of shape (M, 3)
    # triangles: Tensor of shape (N, 3, 3), where each triangle is defined by 3 vertices
    # k: Number of closest triangles to store for each point
    # batch_size: Number of points to process at a time
    
    M = points.shape[0]
    N = triangles.shape[0]

    batch_size = 10_000_000 // N  # Compute a reasonable batch size based on memory constraints
    
    # Initialize a tensor to store the k closest triangles for each point
    # closest_distances = torch.full((M, k), float('inf'), device=points.device)  # Shape: (M, k)
    closest_indices = torch.zeros((M, k), dtype=torch.long, device=points.device)  # Shape: (M, k)
    closest_points = torch.zeros((M, k, 3), device=points.device)  # Shape: (M, k, 3)

    # Process points in batches
    with torch.no_grad():
        for i in range(0, M, batch_size):
            batch_end = min(i + batch_size, M)
            points_batch = points[i:batch_end]  # Shape: (batch_size, 3)
            
            # Compute distances for the current batch of points to all triangles
            distances_batch, closest_points_batch = compute_distances_batch(points_batch, triangles, triangle_normals)  # Shape: (batch_size, N)
            
            # Find the k smallest distances and their indices for the current batch
            batch_closest_distances, batch_closest_indices = torch.topk(distances_batch, k, largest=False, sorted=True)  # Shape: (batch_size, k)
            batch_closest_points = torch.gather(closest_points_batch, 1, batch_closest_indices.unsqueeze(-1).expand(batch_closest_distances.shape[0], k, 3))  # Shape: (batch_size, k, 3)

            # Store the results for the current batch
            # closest_distances[i:batch_end] = batch_closest_distances
            closest_indices[i:batch_end] = batch_closest_indices
            closest_points[i:batch_end] = batch_closest_points

    # recompute the distances with grad
    points_flat = points.unsqueeze(1).expand(M, k, 3).reshape(M * k, 3)
    triangles_flat = torch.index_select(triangles, 0, closest_indices.flatten()).reshape(M * k, 3, 3)
    triangle_normals_flat = torch.index_select(triangle_normals, 0, closest_indices.flatten()).reshape(M * k, 3)
    distances_flat, _ = compute_distances(points_flat, triangles_flat, triangle_normals_flat)
    distances = distances_flat.reshape(M, k)

    # assert closest_distances.allclose(distances), f"{closest_distances} != {distances}"
    # assert closest_distances.allclose((closest_points - points.unsqueeze(1)).norm(dim=-1)), f"{closest_distances} != {(closest_points - points.unsqueeze(1)).norm(dim=-1)}"
    
    return closest_indices, distances, closest_points




def test_vectorized_closest_point():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define k points and k triangles
    k = 800
    points = torch.rand(k, 3) * 5 - 1  # Random points in range [-1, 4)
    triangle_1 = torch.rand(1, 3, 3).repeat(k // 2, 1, 1)  # Shape: (k, 3, 3)
    triangle_2 = torch.rand(1, 3, 3).repeat(k // 2, 1, 1)  # Shape: (k, 3, 3)
    triangles = torch.cat((triangle_1, triangle_2), dim=0)  # Shape: (k, 3, 3)

    # Compute distances
    distances = compute_distances(points, triangles)

    # Compute the hash (sum of closest points)
    closest_points = Triangle(triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]).closest_point_to(points)
    hash = closest_points.sum(dim=0)  # Shape: (3,)

    # Expected hash value (from the C# test case)
    expected_hash = torch.tensor([1496.28118561104, 618.196568578824, 0.0])

    # Compare the computed hash to the expected hash
    tolerance = 1e-5
    if torch.allclose(hash, expected_hash, atol=tolerance):
        print("Vectorized test passed! The hash matches the expected value.")
    else:
        print("Vectorized test failed! The hash does not match the expected value.")
        print(f"Computed hash: {hash}")
        print(f"Expected hash: {expected_hash}")


if __name__ == "__main__":

    # Run the test
    test_vectorized_closest_point()
    plt.savefig('closest_point_to_vectorized.png')
