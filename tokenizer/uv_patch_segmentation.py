"""
UV-Driven Patch Segmentation
============================
Core module for Scheme A: UV-Guided Face Cluster as Token.

Given a mesh with UVs, this module:
1. Detects UV seams (mesh edges where adjacent faces disagree on UV mapping).
2. Cuts the mesh along seams, duplicating vertices as needed.
3. Extracts connected face components = patches (UV islands).

Each patch is a self-contained sub-mesh with consistent UV parameterization.
"""
from typing import List, Dict, Tuple, Set
import numpy as np

from .mesh_utils import build_edge_to_faces, build_face_adjacency, extract_connected_components


class MeshPatch:
    """
    A single patch (UV island) extracted from a mesh.
    """
    def __init__(
        self,
        patch_id: int,
        global_faces: np.ndarray,
        local_faces: np.ndarray,
        local_vertices: np.ndarray,
        local_uvs: np.ndarray,
        global_vertex_remap: np.ndarray,
        boundary_edges: Set[Tuple[int, int]],
    ):
        self.patch_id = patch_id
        self.global_faces = global_faces       # (F_p, 3) indices into original mesh
        self.local_faces = local_faces         # (F_p, 3) indices into local_vertices
        self.local_vertices = local_vertices   # (V_p, 3)
        self.local_uvs = local_uvs             # (V_p, 2)
        self.global_vertex_remap = global_vertex_remap  # (V_p,) maps local -> original vertex
        self.boundary_edges = boundary_edges   # Set of (local_a, local_b) on seam boundary

    def __repr__(self):
        return (
            f"MeshPatch(id={self.patch_id}, faces={len(self.local_faces)}, "
            f"verts={len(self.local_vertices)}, boundary_edges={len(self.boundary_edges)})"
        )


def detect_uv_seams(
    faces: np.ndarray,
    uv_faces: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    Detect mesh edges that are UV seams.

    A mesh edge is a UV seam if:
    - It is shared by two faces, BUT those two faces use different UV coordinates
      along that edge (different UV vertex indices).
    - OR it is on a boundary (only one face uses it) and the UV edge is not
      on the UV boundary (rare in well-formed data).

    For robustness, we use the conservative rule:
    - Group faces by mesh edge.
    - Collect the (uv_idx_a, uv_idx_b) pair for that edge in each face.
    - If there are multiple distinct UV pairs for the same mesh edge, it's a seam.

    Args:
        faces: (F, 3) vertex indices.
        uv_faces: (F, 3) UV indices, aligned with faces.

    Returns:
        Set of mesh edges (sorted vertex index tuples) that are UV seams.
    """
    edge_uv_pairs: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    for f_idx in range(len(faces)):
        face_v = faces[f_idx]
        face_uv = uv_faces[f_idx]
        for i in range(3):
            v0 = int(face_v[i])
            v1 = int(face_v[(i + 1) % 3])
            mesh_edge = (v0, v1) if v0 < v1 else (v1, v0)

            uv0 = int(face_uv[i])
            uv1 = int(face_uv[(i + 1) % 3])
            uv_edge = (uv0, uv1) if uv0 < uv1 else (uv1, uv0)

            edge_uv_pairs.setdefault(mesh_edge, set()).add(uv_edge)

    seams = set()
    for mesh_edge, uv_set in edge_uv_pairs.items():
        # If more than one distinct UV mapping exists for this mesh edge,
        # it means two adjacent faces parameterize the edge differently -> seam.
        if len(uv_set) > 1:
            seams.add(mesh_edge)
        # Note: We deliberately do NOT flag boundary mesh edges as seams
        # unless they have UV discontinuities. Boundary edges are natural
        # patch boundaries but not necessarily "seams" in the UV sense.

    return seams


def build_face_pairs_for_seams(
    faces: np.ndarray,
    seam_edges: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """
    Convert seam edges (vertex pairs) to blocked face adjacency pairs.

    If two faces share a seam edge, they belong to different patches,
    so we block their adjacency during component extraction.

    Returns:
        Set of (face_a, face_b) tuples to block.
    """
    edge_to_faces = build_edge_to_faces(faces)
    blocked = set()
    for edge in seam_edges:
        f_list = edge_to_faces.get(edge, [])
        # If exactly two faces share this seam edge, block them.
        # Non-manifold edges (>2 faces) are inherently problematic;
        # we block all pairwise adjacencies across them to be safe.
        for i in range(len(f_list)):
            for j in range(i + 1, len(f_list)):
                fa, fb = f_list[i], f_list[j]
                pair = (fa, fb) if fa < fb else (fb, fa)
                blocked.add(pair)
    return blocked


def segment_mesh_by_uv_islands(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
) -> List[MeshPatch]:
    """
    Main entry point: segment a mesh into UV-island patches.

    Algorithm:
    1. Detect UV seam edges.
    2. Block face adjacency across seams.
    3. Flood-fill to extract connected face components.
    4. For each component, build a local sub-mesh with duplicated boundary vertices.

    Args:
        vertices: (V, 3) float array.
        faces: (F, 3) int array of vertex indices.
        uvs: (U, 2) float array of UV coordinates.
        uv_faces: (F, 3) int array of UV indices, aligned with faces.

    Returns:
        List of MeshPatch objects.
    """
    # 1. Detect seams
    seam_edges = detect_uv_seams(faces, uv_faces)

    # 2. Build adjacency and block seam crossings
    adj = build_face_adjacency(faces)
    blocked_pairs = build_face_pairs_for_seams(faces, seam_edges)
    components = extract_connected_components(adj, blocked_pairs)

    # 3. Build patches
    patches: List[MeshPatch] = []
    for patch_id, comp_faces in enumerate(components):
        patch = _build_patch_from_component(
            patch_id, comp_faces, vertices, faces, uvs, uv_faces, seam_edges
        )
        patches.append(patch)

    return patches


def _build_patch_from_component(
    patch_id: int,
    comp_face_indices: Set[int],
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    seam_edges: Set[Tuple[int, int]],
) -> MeshPatch:
    """
    Given a set of face indices forming one component, build a local sub-mesh.

    Strategy:
    - Collect all (global_vertex, uv_coord) pairs used by faces in this component.
    - Because seam vertices are duplicated in UV space (different UVs at the same
      3D position), we treat each (global_vertex, uv_index) pair as a unique
      local vertex. This naturally handles the cut along seams.
    """
    comp_face_indices = sorted(comp_face_indices)
    global_faces = faces[comp_face_indices]       # (F_p, 3)
    global_uv_faces = uv_faces[comp_face_indices]  # (F_p, 3)

    # Mapping from (global_v_idx, global_uv_idx) -> local_v_idx
    vertex_key_to_local: Dict[Tuple[int, int], int] = {}
    local_vertices_list = []
    local_uvs_list = []
    global_vertex_remap_list = []

    for f_off in range(len(comp_face_indices)):
        for i in range(3):
            g_v = int(global_faces[f_off, i])
            g_uv = int(global_uv_faces[f_off, i])
            key = (g_v, g_uv)
            if key not in vertex_key_to_local:
                local_idx = len(local_vertices_list)
                vertex_key_to_local[key] = local_idx
                local_vertices_list.append(vertices[g_v])
                local_uvs_list.append(uvs[g_uv])
                global_vertex_remap_list.append(g_v)

    local_vertices = np.stack(local_vertices_list, axis=0)   # (V_p, 3)
    local_uvs = np.stack(local_uvs_list, axis=0)             # (V_p, 2)
    global_vertex_remap = np.array(global_vertex_remap_list, dtype=np.int64)

    # Build local face indices
    local_faces = np.zeros_like(global_faces)
    for f_off in range(len(comp_face_indices)):
        for i in range(3):
            g_v = int(global_faces[f_off, i])
            g_uv = int(global_uv_faces[f_off, i])
            local_faces[f_off, i] = vertex_key_to_local[(g_v, g_uv)]

    # Identify boundary edges within this patch
    # A local edge is on boundary if its corresponding mesh edge is a seam
    boundary_edges: Set[Tuple[int, int]] = set()
    for f_off in range(len(comp_face_indices)):
        for i in range(3):
            lv0 = int(local_faces[f_off, i])
            lv1 = int(local_faces[f_off, (i + 1) % 3])
            # Map back to global vertices to check if this is a seam edge
            gv0 = global_vertex_remap[lv0]
            gv1 = global_vertex_remap[lv1]
            mesh_edge = (gv0, gv1) if gv0 < gv1 else (gv1, gv0)
            if mesh_edge in seam_edges:
                local_edge = (lv0, lv1) if lv0 < lv1 else (lv1, lv0)
                boundary_edges.add(local_edge)

    return MeshPatch(
        patch_id=patch_id,
        global_faces=np.array(comp_face_indices, dtype=np.int64),
        local_faces=local_faces,
        local_vertices=local_vertices,
        local_uvs=local_uvs,
        global_vertex_remap=global_vertex_remap,
        boundary_edges=boundary_edges,
    )
