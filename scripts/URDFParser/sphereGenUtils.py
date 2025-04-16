import trimesh
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
import os

def compute_bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute a simple bounding sphere that encloses all points.
    
    This version uses the average of all points as the center,
    and the maximum distance from that center as the radius.
    
    For many applications, this is acceptable.
    For a minimal enclosing sphere, more complex algorithms exist.
    
    :param points: (N, 3) array of 3D points
    :return: (center, radius)
    """
    center = points.mean(axis=0)
    # Euclidean distance from the center
    distances = np.linalg.norm(points - center, axis=1)
    radius = distances.max()
    return center, radius

class KDNode:
    """
    A node in a simple 3D k-d tree.
    
    - `points`: the points that fall within this node (only stored if it's a leaf).
    - `center`, `radius`: bounding sphere for the leaf's points (None if not a leaf).
    - `left`, `right`: child KDNode(s).
    - `depth`: the current depth in the k-d tree.
    """
    def __init__(
        self,
        points: np.ndarray,
        depth: int = 0,
        max_depth: int = 5,
        min_leaf_size: int = 50
    ):
        self.depth = depth
        self.points = None  # We'll store points only in leaves
        self.center = None
        self.radius = None
        self.left = None
        self.right = None

        # If we've reached max depth or have few enough points, make this a leaf.
        if depth >= max_depth or len(points) <= min_leaf_size:
            self.points = points
            self.center, self.radius = compute_bounding_sphere(points)
        else:
            # Determine the dimension along which to split (largest variance is common in KD-trees).
            variances = points.var(axis=0)  # variance along x, y, z
            split_dim = np.argmax(variances)

            # Sort points by the chosen dimension and find the median index
            sorted_idx = np.argsort(points[:, split_dim])
            mid = len(sorted_idx) // 2

            # Partition into left/right sets
            left_points = points[sorted_idx[:mid]]
            right_points = points[sorted_idx[mid:]]

            # Recursively build child nodes
            self.left = KDNode(
                left_points,
                depth + 1,
                max_depth,
                min_leaf_size
            )
            self.right = KDNode(
                right_points,
                depth + 1,
                max_depth,
                min_leaf_size
            )

            
def build_kdtree(
    points: np.ndarray,
    max_depth: int = 5,
    min_leaf_size: int = 50
) -> KDNode:
    """
    Helper to build a k-d tree from an array of 3D points.
    
    :param points: (N, 3) array of 3D points
    :param max_depth: maximum depth of the tree
    :param min_leaf_size: leaf if the subset has <= this many points
    :return: KDNode (the root of the tree)
    """
    return KDNode(points, 0, max_depth, min_leaf_size)

def generateSphereEnvelop(mesh, maxEdgeLength = 0.05, KDTreeDepth = 3, minLeafSize = 200):
    # 1. Subdivide the mesh to ensure uniform distribution of points
    v, f = trimesh.remesh.subdivide_to_size(vertices = mesh.vertices, faces = mesh.faces, max_edge=maxEdgeLength)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    trimesh.repair.fix_winding(mesh)
    
    # 2. Choose what points to group.
    #    Option 1: use all unique vertices of the mesh
    points = mesh.vertices  
    #    Option 2 (example): use triangle centroids for a coarser grouping
    # points = mesh.triangles_center

    # 3. Build the KD-tree up to depth = 5 (example)
    root = build_kdtree(points, max_depth=KDTreeDepth, min_leaf_size=minLeafSize)

    # Now each leaf node (KDNode with self.points != None)
    # will have a bounding sphere: (node.center, node.radius).
    # You can traverse the tree to access them, e.g.:
    def collect_leaf_spheres(node: KDNode) -> list[tuple[np.ndarray, float]]:
        if node.left is None and node.right is None:
            # Leaf node
            return [(node.center, node.radius)]
        # Otherwise, gather leaf spheres from children
        spheres = []
        if node.left:
            spheres.extend(collect_leaf_spheres(node.left))
        if node.right:
            spheres.extend(collect_leaf_spheres(node.right))
        return spheres

    leaf_spheres = collect_leaf_spheres(root)

    return leaf_spheres, root