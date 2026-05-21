"""
Unit tests for Face-Cluster VQ-VAE Tokenizer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tokenizer.uv_patch_segmentation import segment_mesh_by_uv_islands
from tokenizer.vqvae_tokenizer import FaceClusterVQVAE


def build_test_plane_with_seam():
    """Re-use the synthetic plane from UV patch tests."""
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


def test_vqvae_shapes():
    """Verify output tensor shapes of the VQ-VAE pipeline."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)

    vqvae = FaceClusterVQVAE(latent_dim=64, codebook_size=128)

    for patch in patches:
        q = vqvae.tokenize(patch)
        assert q.code_indices.shape == (len(patch.local_faces),)
        assert q.face_features.shape == (len(patch.local_faces), 64)
        assert q.reconstructed_corners.shape == (len(patch.local_faces), 3, 3)
        assert q.patch_id == patch.patch_id

    print("OK: VQ-VAE output shapes correct for all patches")


def test_quantization_discreteness():
    """Code indices must be integers in [0, codebook_size)."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)
    vqvae = FaceClusterVQVAE(latent_dim=64, codebook_size=128)

    for patch in patches:
        q = vqvae.tokenize(patch)
        assert q.code_indices.dtype in (np.int64, np.int32)
        assert q.code_indices.min() >= 0
        assert q.code_indices.max() < 128

    print("OK: Code indices are valid discrete tokens")


def test_reconstruction_loss_computable():
    """Loss computation must run without error and return finite values."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)
    vqvae = FaceClusterVQVAE(latent_dim=64, codebook_size=128)

    for patch in patches:
        q = vqvae.tokenize(patch)
        metrics = vqvae.compute_reconstruction_loss(patch, q)
        assert np.isfinite(metrics["recon_l2"])
        assert metrics["n_faces"] == len(patch.local_faces)
        assert metrics["n_tokens"] == len(q.code_indices)

    print("OK: Reconstruction loss computable and finite")


def test_codebook_consistency():
    """Same input should yield same tokens (deterministic with fixed weights)."""
    vertices, faces, uvs, uv_faces = build_test_plane_with_seam()
    patches = segment_mesh_by_uv_islands(vertices, faces, uvs, uv_faces)
    vqvae = FaceClusterVQVAE(latent_dim=64, codebook_size=128)

    patch = patches[0]
    q1 = vqvae.tokenize(patch)
    q2 = vqvae.tokenize(patch)
    assert np.array_equal(q1.code_indices, q2.code_indices)
    assert np.allclose(q1.reconstructed_corners, q2.reconstructed_corners)

    print("OK: Tokenizer is deterministic")


if __name__ == "__main__":
    test_vqvae_shapes()
    test_quantization_discreteness()
    test_reconstruction_loss_computable()
    test_codebook_consistency()
    print("\nAll VQ-VAE tokenizer tests passed.")
