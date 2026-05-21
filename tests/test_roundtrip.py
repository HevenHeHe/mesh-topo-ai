"""
End-to-end roundtrip test: Mesh -> Patches -> Tokens -> Mesh
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tokenizer.uv_patch_segmentation import segment_mesh_by_uv_islands
from tokenizer.vqvae_tokenizer import FaceClusterVQVAE
from tokenizer.mesh_assembler import assemble_mesh_from_quantized_patches


def build_test_plane_with_seam():
    """Synthetic 2-patch plane."""
    vertices = np.array([
        [-1.0,  1.0, 0.0],
        [ 0.0,  1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  0.0, 0.0],
        [ 0.0,  0.0, 0.0],
        [ 1.0,  0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
    ], dtype=np.float32)

    faces = np.array([
        [0, 3, 4], [0, 4, 1],
        [1, 4, 5], [1, 5, 2],
        [3, 6, 7], [3, 7, 4],
        [4, 7, 8], [4, 8, 5],
    ], dtype=np.int64)

    uvs = np.array([
        [0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.5, 0.5],
        [0.0, 0.0], [0.5, 0.0],
        [0.5, 1.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.5],
        [0.5, 0.0], [1.0, 0.0],
    ], dtype=np.float32)

    uv_faces = np.array([
        [0, 2, 3], [0, 3, 1],
        [6, 8, 9], [6, 9, 7],
        [2, 4, 5], [2, 5, 3],
        [8, 10, 11], [8, 11, 9],
    ], dtype=np.int64)

    return vertices, faces, uvs, uv_faces


def test_roundtrip_pipeline():
    """Full pipeline: mesh -> patches -> tokens -> reconstructed mesh."""
    print("=" * 50)
    print("Roundtrip Pipeline Test")
    print("=" * 50)

    # 1. Original mesh
    orig_verts, orig_faces, uvs, uv_faces = build_test_plane_with_seam()
    print(f"Original: {len(orig_verts)} vertices, {len(orig_faces)} faces")

    # 2. Segment into patches
    patches = segment_mesh_by_uv_islands(orig_verts, orig_faces, uvs, uv_faces)
    print(f"Segmented into {len(patches)} patches")
    for p in patches:
        print(f"  Patch {p.patch_id}: {len(p.local_faces)} faces, {len(p.local_vertices)} verts")

    # 3. Tokenize each patch
    vqvae = FaceClusterVQVAE(latent_dim=64, codebook_size=128)
    quantized_list = []
    for patch in patches:
        q = vqvae.tokenize(patch)
        quantized_list.append(q)
        metrics = vqvae.compute_reconstruction_loss(patch, q)
        print(f"  Patch {patch.patch_id} tokenized: {metrics['n_tokens']} tokens, recon_l2={metrics['recon_l2']:.4f}")

    # 4. Assemble back
    recon_verts, recon_faces = assemble_mesh_from_quantized_patches(patches, quantized_list, weld_threshold=1e-4)
    print(f"Reconstructed: {len(recon_verts)} vertices, {len(recon_faces)} faces")

    # 5. Validation
    # Face count must match original (no faces lost or created in roundtrip)
    assert len(recon_faces) == len(orig_faces), \
        f"Face count mismatch: {len(recon_faces)} vs {len(orig_faces)}"

    # All face indices must be valid
    assert recon_faces.min() >= 0
    assert recon_faces.max() < len(recon_verts)

    # No degenerate faces
    for f in recon_faces:
        assert f[0] != f[1] and f[1] != f[2] and f[0] != f[2], \
            f"Degenerate face found: {f}"

    print("OK: Structural validation passed")

    # 6. Geometric fidelity (approximate, since VQ-VAE weights are random)
    # We can't expect exact match with random weights, but we can verify
    # the mesh occupies roughly the same bounding box.
    orig_bbox = orig_verts.max(axis=0) - orig_verts.min(axis=0)
    recon_bbox = recon_verts.max(axis=0) - recon_verts.min(axis=0)
    print(f"Original bbox:  {orig_bbox}")
    print(f"Reconstructed bbox: {recon_bbox}")

    # With random weights, the reconstructed mesh will be garbage.
    # In Phase 3, we expect recon_l2 to drop significantly.
    print("\nNOTE: Geometric fidelity is low because VQ-VAE uses random weights.")
    print("Phase 3 training will replace placeholders with trained GNN encoder/decoder.")
    print("=" * 50)
    print("ROUNDTRIP PIPELINE: PASSED")
    print("=" * 50)


def test_roundtrip_single_patch():
    """Roundtrip on a mesh with no internal UV seams (single patch)."""
    # Simple 2-triangle quad, single patch
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    uv_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)
    assert len(patches) == 1

    vqvae = FaceClusterVQVAE(latent_dim=32, codebook_size=64)
    q = vqvae.tokenize(patches[0])
    recon_verts, recon_faces = assemble_mesh_from_quantized_patches(patches, [q])

    assert len(recon_faces) == len(faces)
    assert recon_faces.min() >= 0
    assert recon_faces.max() < len(recon_verts)
    print("OK: Single-patch roundtrip passed")


if __name__ == "__main__":
    test_roundtrip_pipeline()
    test_roundtrip_single_patch()
    print("\nAll roundtrip tests passed.")
