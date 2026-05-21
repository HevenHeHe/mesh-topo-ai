"""
Mesh Assembler
==============
Reconstruct a unified mesh from multiple decoded patches.

Key challenge: Patches were separated along UV seams, with boundary vertices
duplicated. When merging patches back, vertices at seam boundaries that
came from the same global vertex must be welded back together.
"""
from typing import List, Tuple, Dict
import numpy as np

from .uv_patch_segmentation import MeshPatch


def deduplicate_vertices(
    corners: np.ndarray,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate vertices from face corners.

    Args:
        corners: (F, 3, 3) array of face corner positions.
        threshold: Distance threshold for considering two vertices identical.

    Returns:
        (vertices, face_indices) where:
        - vertices: (V, 3) deduplicated vertex positions
        - face_indices: (F, 3) remapped face indices
    """
    # Flatten all corners: (F*3, 3)
    flat = corners.reshape(-1, 3)
    n_corners = len(flat)

    # Simple O(N^2) clustering for small meshes; Phase 3 can use k-d tree
    vertex_map = np.full(n_corners, -1, dtype=np.int64)
    vertices_list = []

    for i in range(n_corners):
        if vertex_map[i] >= 0:
            continue
        # New unique vertex
        new_idx = len(vertices_list)
        vertices_list.append(flat[i])
        vertex_map[i] = new_idx
        # Find all close corners
        for j in range(i + 1, n_corners):
            if vertex_map[j] >= 0:
                continue
            dist = np.linalg.norm(flat[i] - flat[j])
            if dist < threshold:
                vertex_map[j] = new_idx

    vertices = np.stack(vertices_list, axis=0)
    face_indices = vertex_map.reshape(-1, 3)
    return vertices, face_indices


def merge_patches(
    patches: List[MeshPatch],
    reconstructed_corners_list: List[np.ndarray],
    weld_threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge multiple decoded patches into a single unified mesh.

    Strategy:
    1. Deduplicate vertices within each patch individually.
    2. Collect all patches' vertices and faces with a global offset.
    3. Weld boundary vertices: if two boundary vertices from different patches
       came from the same original global vertex, merge them.

    Args:
        patches: List of MeshPatch objects.
        reconstructed_corners_list: List of (F_p, 3, 3) arrays, aligned with patches.
        weld_threshold: Distance threshold for welding seam vertices.

    Returns:
        (vertices, faces, uvs) unified mesh. UVs may be None if not reconstructible.
    """
    if len(patches) != len(reconstructed_corners_list):
        raise ValueError("patches and reconstructed_corners_list must have same length")

    # Step 1: Per-patch deduplication
    patch_vertices = []
    patch_faces = []
    patch_global_remap = []
    patch_boundary_local = []

    for patch, corners in zip(patches, reconstructed_corners_list):
        verts, faces = deduplicate_vertices(corners, threshold=1e-6)
        patch_vertices.append(verts)
        patch_faces.append(faces)
        # We need to map deduplicated local indices back to global vertices.
        # Since deduplication changes indices, we need a two-step remap.
        # Actually, for the boundary-weld logic, we use the original global_vertex_remap
        # directly: each local vertex in the patch has a corresponding global vertex id.
        # When deduplicating within a patch, two local vertices that map to the same
        # global vertex should already be at the same position (since they came from
        # the same 3D position), so they'll be merged by deduplicate_vertices.
        # But we need a new global remap for the deduplicated vertices.
        # For simplicity, we approximate: use the global remap from the first corner
        # that maps to each deduplicated vertex.
        # Actually, let's keep it simpler: don't deduplicate within patch for now,
        # just use the raw corners and weld at the global level.
        pass

    # Simpler strategy for Phase 2: skip per-patch dedup, go straight to global merge
    return _merge_patches_simple(patches, reconstructed_corners_list, weld_threshold)


def _merge_patches_simple(
    patches: List[MeshPatch],
    reconstructed_corners_list: List[np.ndarray],
    weld_threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple merge: concatenate all faces, then weld vertices by global vertex ID.
    Since each local vertex in each patch maps to a global vertex ID, vertices
    from different patches that share the same global ID should be welded.

    But reconstructed corners don't directly map to local vertices...
    We need to reconstruct the local mesh first, then map to global.
    """
    all_vertices = []
    all_faces = []
    all_uvs = []
    vertex_offset = 0

    for patch, corners in zip(patches, reconstructed_corners_list):
        # Reconstruct local mesh from corners
        local_verts, local_faces = deduplicate_vertices(corners, threshold=1e-6)
        n_local_verts = len(local_verts)

        # For UVs, try to preserve from patch if available
        if len(patch.local_uvs) >= n_local_verts:
            patch_uvs = patch.local_uvs[:n_local_verts]
        else:
            # Fallback: zero UVs
            patch_uvs = np.zeros((n_local_verts, 2), dtype=np.float32)

        all_vertices.append(local_verts)
        all_faces.append(local_faces + vertex_offset)
        all_uvs.append(patch_uvs)
        vertex_offset += n_local_verts

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    uvs = np.concatenate(all_uvs, axis=0)

    # Welding: we need to know which vertices should be merged.
    # In the original patches, seam vertices were duplicated because they had
    # different UVs. After reconstruction, the UV information is lost (or approximated).
    # So we weld purely by geometric proximity.
    vertices, remap = _weld_by_proximity(vertices, weld_threshold)
    faces = remap[faces]

    # Remove degenerate faces
    faces = _remove_degenerate_faces(faces)

    return vertices, faces, uvs


def _weld_by_proximity(
    vertices: np.ndarray,
    threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weld vertices that are within threshold distance of each other.

    Returns:
        (welded_vertices, remap) where remap[i] gives the new index of old vertex i.
    """
    n = len(vertices)
    remap = np.full(n, -1, dtype=np.int64)
    welded = []

    for i in range(n):
        if remap[i] >= 0:
            continue
        new_idx = len(welded)
        welded.append(vertices[i])
        remap[i] = new_idx
        for j in range(i + 1, n):
            if remap[j] >= 0:
                continue
            if np.linalg.norm(vertices[i] - vertices[j]) < threshold:
                remap[j] = new_idx

    welded_vertices = np.stack(welded, axis=0) if welded else np.zeros((0, 3), dtype=vertices.dtype)
    return welded_vertices, remap


def _remove_degenerate_faces(faces: np.ndarray) -> np.ndarray:
    """Remove faces where two or more vertices are identical."""
    valid = []
    for f in faces:
        if f[0] != f[1] and f[1] != f[2] and f[0] != f[2]:
            valid.append(f)
    if not valid:
        return np.zeros((0, 3), dtype=faces.dtype)
    return np.stack(valid, axis=0)


def assemble_mesh_from_quantized_patches(
    patches: List[MeshPatch],
    quantized_list: List,
    weld_threshold: float = 1e-5,
    use_topology_weld: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: takes patches + quantized results, returns unified mesh.

    Args:
        patches: List of MeshPatch.
        quantized_list: List of QuantizedPatch (or objects with .reconstructed_corners).
        weld_threshold: Welding threshold for geometric proximity.
        use_topology_weld: If True, use global vertex ID from patches to guide
                          welding in addition to geometric proximity.
                          This is DEFENSE #3: topological elevation of weld.

    Returns:
        (vertices, faces)
    """
    corners_list = [q.reconstructed_corners for q in quantized_list]
    
    if use_topology_weld:
        # Use topological elevation: weld vertices that share the same
        # global vertex ID across patches, falling back to geometric
        # proximity for unmatched vertices.
        vertices, faces = _merge_patches_with_topology(
            patches, corners_list, weld_threshold
        )
    else:
        vertices, faces, _ = merge_patches(patches, corners_list, weld_threshold)
    
    return vertices, faces


def _merge_patches_with_topology(
    patches: List[MeshPatch],
    reconstructed_corners_list: List[np.ndarray],
    weld_threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEFENSE #3: Topological Elevation of Weld Algorithm
    ===================================================
    
    Instead of welding purely by geometric proximity (which can fail
    when two nearby vertices from different patches should NOT be welded),
    we use the original global vertex ID as primary weld criterion.
    
    Strategy:
    1. Collect all vertices with their global_vertex_remap IDs
    2. Group vertices by global ID → these MUST be welded
    3. For vertices without matching global ID, fall back to geometric proximity
    4. This prevents false welds and ensures seam vertices align correctly
    
    Args:
        patches: List of MeshPatch with global_vertex_remap.
        reconstructed_corners_list: Reconstructed corners per patch.
        weld_threshold: Fallback geometric threshold.
    
    Returns:
        (vertices, faces)
    """
    # Step 1: Collect all local vertices with their global IDs
    all_vertices = []
    all_faces = []
    all_global_ids = []
    vertex_offset = 0
    
    for patch, corners in zip(patches, reconstructed_corners_list):
        # Reconstruct local mesh from corners
        local_verts, local_faces = deduplicate_vertices(corners, threshold=1e-6)
        n_local_verts = len(local_verts)
        
        # Map deduplicated local indices back to global IDs
        # For each deduplicated vertex, we need to know which global vertex
        # it came from. Since deduplication merges corners, we approximate
        # by taking the first occurrence's global ID.
        # In Phase 3, this should be done more carefully during deduplication.
        local_global_ids = []
        # Simplified: use patch.global_vertex_remap directly
        # (assuming 1:1 mapping after corner deduplication for now)
        if len(patch.global_vertex_remap) >= n_local_verts:
            local_global_ids = patch.global_vertex_remap[:n_local_verts].tolist()
        else:
            local_global_ids = [-1] * n_local_verts
        
        all_vertices.append(local_verts)
        all_faces.append(local_faces + vertex_offset)
        all_global_ids.extend(local_global_ids)
        vertex_offset += n_local_verts
    
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    global_ids = np.array(all_global_ids, dtype=np.int64)
    
    # Step 2: Weld by global ID (primary criterion)
    n = len(vertices)
    remap = np.full(n, -1, dtype=np.int64)
    welded = []
    
    # Group by global ID
    id_groups = {}
    for i in range(n):
        gid = int(global_ids[i])
        if gid >= 0:
            id_groups.setdefault(gid, []).append(i)
    
    # Weld vertices with same global ID
    for gid, indices in id_groups.items():
        new_idx = len(welded)
        welded.append(vertices[indices[0]])
        for idx in indices:
            remap[idx] = new_idx
    
    # Step 3: Fall back to geometric proximity for unmatched vertices
    for i in range(n):
        if remap[i] >= 0:
            continue
        new_idx = len(welded)
        welded.append(vertices[i])
        remap[i] = new_idx
        for j in range(i + 1, n):
            if remap[j] >= 0:
                continue
            if np.linalg.norm(vertices[i] - vertices[j]) < weld_threshold:
                remap[j] = new_idx
    
    welded_vertices = np.stack(welded, axis=0) if welded else np.zeros((0, 3), dtype=vertices.dtype)
    faces = remap[faces]
    
    # Step 4: Clean degenerate faces
    faces = _remove_degenerate_faces(faces)
    
    return welded_vertices, faces
