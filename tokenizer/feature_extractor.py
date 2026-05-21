"""
Feature Extractor
=================
Extracts topological features from mesh patches.

This is a minimal placeholder that will be expanded in Phase 3
with full GNN-based feature extraction.
"""

from typing import NamedTuple
import numpy as np

from .uv_patch_segmentation import MeshPatch


class PatchFeatures(NamedTuple):
    """
    Topological features extracted from a single MeshPatch.
    """
    ordered_face_sequence: np.ndarray    # (F, 3) int, ordered face indices
    topology_descriptor: np.ndarray      # (F, 10) float, per-face 10-D feature
    adjacency_graph_edges: np.ndarray    # (F, 3) int, face adjacency
    face_normals: np.ndarray             # (F, 3) float, per-face normals
