"""
Unit tests for UV-driven patch segmentation.
Uses synthetic meshes with known UV seams to verify correctness.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tokenizer.uv_patch_segmentation import (
    detect_uv_seams,
    segment_mesh_by_uv_islands,
)


def build_test_plane_with_seam():
    """
    Build a 2x2 quad grid (in 3D) with a UV seam down the middle.

    3D layout (3x3 vertices):
        y=1:  v0 -- v1 -- v2
              |  \ |  / |
        y=0:  v3 -- v4 -- v5
              |  / |  \ |
        y=-1: v6 -- v7 -- v8

    But the middle column (v1, v4, v7) is duplicated in UV space:
    - Left patch uses UVs:  u in [0, 0.5]
    - Right patch uses UVs: u in [0.5, 1]
    This creates a UV seam along edges (v0,v1), (v3,v4), (v6,v7)
    and their mirror counterparts.

    Expected result: 2 patches (left and right).
    """
    # 3D vertices (3x3 grid)
    vertices = np.array([
        [-1.0,  1.0, 0.0],   # v0
        [ 0.0,  1.0, 0.0],   # v1
        [ 1.0,  1.0, 0.0],   # v2
        [-1.0,  0.0, 0.0],   # v3
        [ 0.0,  0.0, 0.0],   # v4
        [ 1.0,  0.0, 0.0],   # v5
        [-1.0, -1.0, 0.0],   # v6
        [ 0.0, -1.0, 0.0],   # v7
        [ 1.0, -1.0, 0.0],   # v8
    ], dtype=np.float32)

    # Faces (8 triangles covering 4 quads)
    faces = np.array([
        # top-left quad
        [0, 3, 4], [0, 4, 1],
        # top-right quad
        [1, 4, 5], [1, 5, 2],
        # bottom-left quad
        [3, 6, 7], [3, 7, 4],
        # bottom-right quad
        [4, 7, 8], [4, 8, 5],
    ], dtype=np.int64)

    # UV vertices: left side and right side are duplicated at the seam column
    # Left side UVs (u in [0, 0.5])
    uvs = np.array([
        [0.0, 1.0],   # uv0 (left of v0)
        [0.5, 1.0],   # uv1 (left of v1)  <- seam edge
        [0.0, 0.5],   # uv2 (left of v3)
        [0.5, 0.5],   # uv3 (left of v4)  <- seam edge
        [0.0, 0.0],   # uv4 (left of v6)
        [0.5, 0.0],   # uv5 (left of v7)  <- seam edge
        # Right side UVs (u in [0.5, 1.0])
        [0.5, 1.0],   # uv6 (right of v1) <- seam edge
        [1.0, 1.0],   # uv7 (right of v2)
        [0.5, 0.5],   # uv8 (right of v4) <- seam edge
        [1.0, 0.5],   # uv9 (right of v5)
        [0.5, 0.0],   # uv10 (right of v7) <- seam edge
        [1.0, 0.0],   # uv11 (right of v8)
    ], dtype=np.float32)

    # UV faces aligned with faces
    uv_faces = np.array([
        # top-left
        [0, 2, 3], [0, 3, 1],
        # top-right
        [6, 8, 9], [6, 9, 7],
        # bottom-left
        [2, 4, 5], [2, 5, 3],
        # bottom-right
        [8, 10, 11], [8, 11, 9],
    ], dtype=np.int64)

    return vertices, faces, uvs, uv_faces


def build_test_cylinder_with_seam():
    """
    Build a simple 8-vertex cylinder (side only, no caps) with a UV seam
    along one vertical edge.

    Expected: 1 patch for the side (since UV is a single strip with one seam
    that is the boundary of the UV map, not an internal seam).

    Actually for a cylinder unwrapped as a rectangle, the seam edge where
    the two ends meet IS a UV seam (discontinuity). The rest of the mesh
    is continuous in UV space. So we expect 1 patch.
    """
    n = 8
    verts = []
    uvs = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = np.cos(angle)
        z = np.sin(angle)
        verts.append([x, 1.0, z])
        verts.append([x, -1.0, z])
        u = i / n
        uvs.append([u, 1.0])
        uvs.append([u, 0.0])

    vertices = np.array(verts, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    faces = []
    uv_faces = []
    for i in range(n):
        i_next = (i + 1) % n
        # Two triangles per quad
        faces.append([2*i, 2*i+1, 2*i_next+1])
        faces.append([2*i, 2*i_next+1, 2*i_next])
        uv_faces.append([2*i, 2*i+1, 2*i_next+1])
        uv_faces.append([2*i, 2*i_next+1, 2*i_next])

    faces = np.array(faces, dtype=np.int64)
    uv_faces = np.array(uv_faces, dtype=np.int64)
    return vertices, faces, uvs, uv_faces


def test_detect_uv_seams_plane():
    """Seam detection on the synthetic plane."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    seams = detect_uv_seams(faces, uv_faces)

    # The seam should be along the edges that separate left/right patches:
    # (1,4) — middle-top  and (4,7) — middle-bottom
    expected_seams = {
        (1, 4),
        (4, 7),
    }
    for e in expected_seams:
        assert e in seams, f"Expected seam {e} not detected. Got: {seams}"
    print(f"OK: Detected {len(seams)} seam edges on plane (expected >= 2)")


def test_segment_plane():
    """Patch segmentation on the synthetic plane."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)

    assert len(patches) == 2, f"Expected 2 patches, got {len(patches)}"
    total_faces = sum(len(p.local_faces) for p in patches)
    assert total_faces == len(faces), f"Face count mismatch: {total_faces} vs {len(faces)}"

    # Verify each patch has boundary edges
    for p in patches:
        assert len(p.boundary_edges) > 0, f"Patch {p.patch_id} has no boundary edges"
        # Verify local faces are valid
        max_idx = p.local_faces.max()
        assert max_idx < len(p.local_vertices), f"Face index out of bounds in patch {p.patch_id}"

    print(f"OK: Segmented plane into {len(patches)} patches")
    for p in patches:
        print(f"   {p}")


def test_segment_cylinder():
    """Patch segmentation on cylinder (should be 1 patch for side)."""
    vertices, faces, uvs, uv_faces = build_test_cylinder_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)

    # A cylinder unwrapped as a single UV strip has the seam on the boundary
    # of the UV map. Our algorithm flags seams only when UV pairs differ,
    # which happens at the wrap-around edge (v_last top/bottom connect to v_0).
    # However, because it's a closed loop in 3D but open in UV, the wrap-around
    # edge IS a seam. But since every face is still reachable without crossing
    # a seam (the seam edges connect face n-1 to face 0), the whole side is
    # one connected component... wait, the seam edges connect face n-1 to face 0.
    # If we block that adjacency, the cylinder becomes a single strip that is
    # NOT connected at the ends -> but wait, all other edges are still connected,
    # so actually it's still 1 component. The seam only affects the wrap-around
    # edge, all other edges are continuous. So yes, 1 patch.
    # edge, all other edges are continuous. So yes, 1 patch.
    assert len(patches) == 1, f"Expected 1 patch for cylinder side, got {len(patches)}"
    print(f"OK: Cylinder segmented into {len(patches)} patch")


def test_roundtrip_geometry():
    """
    After segmentation, verify that local vertices match global vertices
    at corresponding positions (allowing for duplication at seams).
    """
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)

    for p in patches:
        for local_v, global_v in zip(p.local_vertices, p.global_vertex_remap):
            expected = vertices[global_v]
            assert np.allclose(local_v, expected), \
                f"Vertex mismatch in patch {p.patch_id}: local {local_v} != global {expected}"
    print("OK: Roundtrip geometry verified for all patches")


if __name__ == "__main__":
    test_detect_uv_seams_plane()
    test_segment_plane()
    test_segment_cylinder()
    test_roundtrip_geometry()
    print("\nAll UV patch segmentation tests passed.")
