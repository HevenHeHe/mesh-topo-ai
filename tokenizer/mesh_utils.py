"""
Mesh utilities for edge adjacency, connectivity, and geometry helpers.
Pure Python + NumPy, no heavy dependencies.
"""
from typing import List, Tuple, Dict, Set
import numpy as np


def build_edge_to_faces(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Map each mesh edge (sorted vertex pair) to the list of face indices using it.

    Args:
        faces: (F, 3) array of vertex indices.

    Returns:
        Dict mapping (min_v, max_v) -> [face_idx, ...]
    """
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for f_idx, face in enumerate(faces):
        for i in range(3):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % 3])
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            edge_map.setdefault(edge, []).append(f_idx)
    return edge_map


def build_face_adjacency(faces: np.ndarray) -> Dict[int, Set[int]]:
    """
    Build face adjacency via shared edges.

    Returns:
        Dict mapping face_idx -> set of neighboring face indices.
    """
    edge_map = build_edge_to_faces(faces)
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(faces))}
    for edge, f_list in edge_map.items():
        if len(f_list) == 2:
            f0, f1 = f_list
            adj[f0].add(f1)
            adj[f1].add(f0)
        # Boundary edges (len==1) or non-manifold (len>2) handled naturally
    return adj


def extract_connected_components(
    adj: Dict[int, Set[int]],
    blocked_edges: Set[Tuple[int, int]] = None
) -> List[Set[int]]:
    """
    Extract connected components from face adjacency graph,
    optionally treating blocked_edges as removed connections.

    Args:
        adj: Face adjacency dict.
        blocked_edges: Set of (face_a, face_b) tuples to treat as disconnected.

    Returns:
        List of face index sets, each being one connected component.
    """
    if blocked_edges is None:
        blocked_edges = set()
    else:
        # Normalize ordering for lookup
        blocked_edges = {(a, b) if a < b else (b, a) for a, b in blocked_edges}

    visited = set()
    components = []

    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            f = stack.pop()
            if f in visited:
                continue
            visited.add(f)
            comp.add(f)
            for neighbor in adj[f]:
                if neighbor in visited:
                    continue
                pair = (f, neighbor) if f < neighbor else (neighbor, f)
                if pair not in blocked_edges:
                    stack.append(neighbor)
        components.append(comp)

    return components


def normalize_mesh(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize vertices to unit bounding box centered at origin.

    Returns:
        (normalized_vertices, center, scale)
    """
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float((bbox_max - bbox_min).max())
    if scale < 1e-8:
        scale = 1.0
    norm_verts = (vertices - center) / scale
    return norm_verts, center, scale


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face normals.

    Returns:
        (F, 3) array of normalized face normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return normals / norms


def quantize_positions(vertices: np.ndarray, bits: int = 10) -> np.ndarray:
    """
    Quantize normalized vertices to integer grid [0, 2^bits).

    Args:
        vertices: Already normalized to roughly [-0.5, 0.5] or [0, 1].
        bits: Number of bits per axis.

    Returns:
        (V, 3) array of integers.
    """
    grid = 2 ** bits
    # Shift to [0, 1]
    shifted = vertices - vertices.min(axis=0)
    scaled = shifted / (shifted.max(axis=0) + 1e-8)
    quantized = np.clip((scaled * (grid - 1)).astype(np.int64), 0, grid - 1)
    return quantized
