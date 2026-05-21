"""
Face-Cluster VQ-VAE Tokenizer
=============================
Scheme A: Each UV-island patch is encoded into a discrete token via
a VQ-VAE operating on the patch's face-adjacency graph.

Architecture (Phase 2 skeleton):
--------------------------------
Encoder (GNN on face-adjacency graph):
  Input:  Per-face features (normal, area, centroid, UV stretch)
  Output: Per-face latent vectors z_e(x)

Vector Quantization:
  z_q(x) = argmin_{e_k} || z_e(x) - e_k ||
  (straight-through estimator for backprop)

Decoder (GNN / MLP):
  Input:  Quantized face tokens z_q(x)
  Output: Reconstructed per-face geometry (face corners)

The sequence of quantized tokens (one per face in the patch) becomes
the "token sequence" for the autoregressive Transformer.
"""
from typing import List, Tuple, Optional, NamedTuple
import numpy as np

from .uv_patch_segmentation import MeshPatch
from .mesh_utils import compute_face_normals


class QuantizedPatch(NamedTuple):
    """
    A patch after VQ-VAE encoding.
    """
    patch_id: int
    code_indices: np.ndarray       # (F_p,) int, codebook indices
    face_features: np.ndarray      # (F_p, D) float, per-face latent vectors
    reconstructed_corners: np.ndarray  # (F_p, 3, 3) float, decoded face corners


class FaceClusterVQVAE:
    """
    VQ-VAE tokenizer for a single MeshPatch.

    Phase 2: Skeleton with numpy placeholders.
    Phase 3: Replace with PyTorch Geometric implementation.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        codebook_size: int = 512,
        num_gnn_layers: int = 3,
    ):
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_gnn_layers = num_gnn_layers

        # -------------------------------------------------------------------
        #  Phase 2: Random initialization as placeholder
        #  Phase 3: Load trained PyTorch weights
        # -------------------------------------------------------------------
        self._codebook = np.random.randn(codebook_size, latent_dim).astype(np.float32)
        self._codebook /= np.linalg.norm(self._codebook, axis=1, keepdims=True) + 1e-8

        # Placeholder projection matrices for encoder / decoder
        self._enc_w = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01
        self._dec_w = np.random.randn(latent_dim, 9).astype(np.float32) * 0.01  # 9 = 3 corners * 3 coords

    # -----------------------------------------------------------------------
    #  Feature Extraction
    # -----------------------------------------------------------------------
    def extract_face_features(self, patch: MeshPatch) -> np.ndarray:
        """
        Compute per-face geometric features for the patch.

        Features (per face):
        - Normal (3)
        - Area (1)
        - Centroid (3)
        - UV area (1)
        - UV stretch (1)
        - Boundary flag (1)
        Total: 10-D baseline; padded/truncated to latent_dim for the skeleton.

        Args:
            patch: MeshPatch to encode.

        Returns:
            (F_p, latent_dim) feature matrix.
        """
        verts = patch.local_vertices
        faces = patch.local_faces
        uvs = patch.local_uvs
        n_faces = len(faces)

        # 3D geometry
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        normals = compute_face_normals(verts, faces)          # (F, 3)
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1, keepdims=True)  # (F, 1)
        centroids = (v0 + v1 + v2) / 3.0                       # (F, 3)

        # UV geometry
        uv0 = uvs[faces[:, 0]]
        uv1 = uvs[faces[:, 1]]
        uv2 = uvs[faces[:, 2]]
        uv_areas = 0.5 * np.abs(
            (uv1[:, 0] - uv0[:, 0]) * (uv2[:, 1] - uv0[:, 1]) -
            (uv2[:, 0] - uv0[:, 0]) * (uv1[:, 1] - uv0[:, 1])
        )[:, None]  # (F, 1)

        # UV stretch: ratio of 3D area to UV area (clamped)
        uv_stretch = np.clip(areas / (uv_areas + 1e-8), 0.0, 10.0)

        # Boundary flag: does face touch a seam boundary?
        boundary_flags = np.zeros((n_faces, 1), dtype=np.float32)
        for f_idx in range(n_faces):
            face = faces[f_idx]
            for i in range(3):
                e = (int(face[i]), int(face[(i + 1) % 3]))
                e_sorted = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                if e_sorted in patch.boundary_edges:
                    boundary_flags[f_idx, 0] = 1.0
                    break

        # Concatenate baseline features
        baseline = np.concatenate([
            normals,           # 3
            areas,             # 1
            centroids,         # 3
            uv_areas,          # 1
            uv_stretch,        # 1
            boundary_flags,    # 1
        ], axis=1)  # (F, 10)

        # Pad or project to latent_dim
        if baseline.shape[1] >= self.latent_dim:
            return baseline[:, :self.latent_dim]
        else:
            pad = np.zeros((n_faces, self.latent_dim - baseline.shape[1]), dtype=np.float32)
            return np.concatenate([baseline, pad], axis=1)

    # -----------------------------------------------------------------------
    #  Encoder (placeholder)
    # -----------------------------------------------------------------------
    def encode(self, patch: MeshPatch) -> np.ndarray:
        """
        Encode patch to continuous latent vectors.
        Phase 2: Linear projection placeholder.
        Phase 3: GNN message passing on face-adjacency graph.
        """
        features = self.extract_face_features(patch)  # (F, D)
        # Placeholder: simple linear projection + tanh
        latent = np.tanh(features @ self._enc_w.T)    # (F, D)
        return latent

    # -----------------------------------------------------------------------
    #  Vector Quantization
    # -----------------------------------------------------------------------
    def quantize(self, latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            latent: (F, D) continuous latent vectors.

        Returns:
            codes:    (F,) codebook indices
            z_q:      (F, D) quantized latent vectors
        """
        # L2 distance to all codebook entries
        # latent: (F, D), codebook: (K, D)
        dist = (
            np.sum(latent ** 2, axis=1, keepdims=True)      # (F, 1)
            + np.sum(self._codebook ** 2, axis=1)            # (K,)
            - 2 * latent @ self._codebook.T                  # (F, K)
        )
        codes = np.argmin(dist, axis=1)
        z_q = self._codebook[codes]
        return codes, z_q

    # -----------------------------------------------------------------------
    #  Decoder (placeholder)
    # -----------------------------------------------------------------------
    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latents back to face corners.
        Phase 2: Linear projection placeholder.
        Phase 3: GNN / MLP decoder with mesh reconstruction loss.

        Args:
            z_q: (F, D) quantized vectors.

        Returns:
            corners: (F, 3, 3) reconstructed vertex positions per face.
        """
        flat = z_q @ self._dec_w  # (F, 9)
        corners = flat.reshape(-1, 3, 3)
        return corners

    # -----------------------------------------------------------------------
    #  Full pipeline
    # -----------------------------------------------------------------------
    def tokenize(self, patch: MeshPatch) -> QuantizedPatch:
        """
        Encode a MeshPatch into discrete tokens.
        """
        z_e = self.encode(patch)
        codes, z_q = self.quantize(z_e)
        corners = self.decode(z_q)
        return QuantizedPatch(
            patch_id=patch.patch_id,
            code_indices=codes,
            face_features=z_e,
            reconstructed_corners=corners,
        )

    def detokenize(self, quantized: QuantizedPatch) -> MeshPatch:
        """
        Reconstruct a MeshPatch from quantized tokens.
        Phase 2: Returns a patch with decoded corners; topology (face indices)
        is preserved from the original patch structure.
        """
        raise NotImplementedError("Phase 3: Full mesh reconstruction from corners")

    def compute_reconstruction_loss(
        self,
        patch: MeshPatch,
        quantized: QuantizedPatch,
    ) -> dict:
        """
        Compute reconstruction quality metrics.
        """
        verts = patch.local_vertices
        faces = patch.local_faces
        original_corners = np.stack([
            verts[faces[:, 0]],
            verts[faces[:, 1]],
            verts[faces[:, 2]],
        ], axis=1)  # (F, 3, 3)

        recon = quantized.reconstructed_corners
        l2 = np.mean(np.linalg.norm(original_corners - recon, axis=(1, 2)))
        chamfer_approx = l2  # Placeholder; true Chamfer requires point matching

        return {
            "recon_l2": float(l2),
            "chamfer_approx": float(chamfer_approx),
            "n_faces": len(faces),
            "n_tokens": len(quantized.code_indices),
        }
