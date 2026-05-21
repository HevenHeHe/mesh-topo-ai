#!/usr/bin/env python3
"""
batch_preprocess.py
===================
DEFENSE #2: UV Patch Density Kill Switch
========================================
Batch preprocessing script with hard filtering criteria to prevent
UV fragmentation from destroying transformer context windows.

This script implements the UV density kill switch:
    Density = total_faces / num_patches
    - Density > 50   → ✅ Keep (industrial-grade UV partitioning)
    - 20 < Density < 50 → ⚠️ Keep (borderline)
    - Density < 20   → ❌ Discard (UV is too fragmented)

Usage:
    python batch_preprocess.py \
        --input-dir /path/to/raw_meshes/ \
        --output-dir /path/to/processed/ \
        --min-patch-density 20.0 \
        --max-patches-per-mesh 50

Dependencies:
    - numpy
    - trimesh (for mesh loading)
    - tqdm (for progress bars)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import trimesh
except ImportError:
    print("❌ Error: trimesh not installed. Run: pip install trimesh")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path for tokenizer imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.feature_extractor import MeshPatch, PatchFeatures


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MIN_PATCH_DENSITY = 20.0
DEFAULT_MAX_PATCHES_PER_MESH = 50
DEFAULT_MIN_PATCH_FACES = 3  # A valid patch must have at least 3 faces


# =============================================================================
# MESH LOADING
# =============================================================================

def load_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from file using trimesh.
    
    Args:
        mesh_path: Path to mesh file (.obj, .stl, .ply, etc.)
    
    Returns:
        (vertices, faces) where vertices is (V, 3) and faces is (F, 3)
    """
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Could not load mesh: {mesh_path}")
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)
    
    return vertices, faces


def extract_uvs_and_segmentation(
    mesh_path: Path
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract UV coordinates and segmentation masks from mesh.
    
    For CAD datasets without native UV, this function attempts to:
    1. Load existing UVs from the mesh file
    2. If no UVs exist, run automatic UV unwrapping (xatlas)
    
    Args:
        mesh_path: Path to mesh file
    
    Returns:
        (uv_islands, segmentation_masks)
    """
    # Placeholder: In production, this would use xatlas or Blender's
    # Smart UV Project to generate UV islands.
    # For now, return dummy data.
    
    mesh = trimesh.load(str(mesh_path), force="mesh")
    
    # Try to get UVs from visual attributes
    uvs = []
    segmentation_masks = []
    
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        uv = mesh.visual.uv
        if uv is not None and len(uv) > 0:
            # Group UVs into islands (simplified)
            uvs.append(np.array(uv))
            segmentation_masks.append(np.ones(len(uv), dtype=np.bool_))
    
    # If no UVs found, return single island covering all faces
    if len(uvs) == 0:
        faces = np.array(mesh.faces)
        uvs.append(np.zeros((len(faces), 2, 3), dtype=np.float32))
        segmentation_masks.append(np.ones(len(faces), dtype=np.bool_))
    
    return uvs, segmentation_masks


# =============================================================================
# MESH NORMALIZATION
# =============================================================================

def normalize_mesh(vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize mesh to unit bounding box centered at origin.
    
    Returns:
        (normalized_vertices, bbox_info)
    """
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min
    scale = bbox_size.max()
    
    if scale < 1e-6:
        scale = 1.0
    
    normalized = (vertices - bbox_center) / scale
    
    bbox_info = {
        "center": bbox_center.tolist(),
        "size": bbox_size.tolist(),
        "scale": float(scale),
    }
    
    return normalized, bbox_info


# =============================================================================
# PATCH SPLITTING
# =============================================================================

def split_mesh_into_patches(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: List[np.ndarray],
    segmentation_masks: List[np.ndarray],
) -> List[MeshPatch]:
    """
    Split mesh into patches based on UV islands and segmentation masks.
    
    DEFENSE #2: This function now also computes and validates UV density.
    """
    patches = []
    
    # Simplified: Create one patch per UV island
    for i, (uv, mask) in enumerate(zip(uvs, segmentation_masks)):
        # Get faces associated with this UV island
        face_indices = np.where(mask)[0]
        
        if len(face_indices) < DEFAULT_MIN_PATCH_FACES:
            continue
        
        patch_faces = faces[face_indices]
        
        # Remap vertices to local indices
        unique_verts = np.unique(patch_faces.flatten())
        vert_remap = {old: new for new, old in enumerate(unique_verts)}
        
        local_faces = np.array([
            [vert_remap[v] for v in face]
            for face in patch_faces
        ], dtype=np.int64)
        
        local_vertices = vertices[unique_verts]
        
        patch = MeshPatch(
            mesh_index=0,
            patch_index=i,
            local_vertices=local_vertices,
            local_faces=local_faces,
            face_uv_coords=uv if len(uv.shape) == 3 else np.zeros(
                (len(patch_faces), 2, 3), dtype=np.float32
            ),
            face_segmentation_mask=np.ones(len(patch_faces), dtype=np.bool_),
            global_vertex_remap=unique_verts.astype(np.int64),
        )
        patches.append(patch)
    
    return patches


# =============================================================================
# FEATURE EXTRACTION (PLACEHOLDER)
# =============================================================================

def extract_patch_features(patch: MeshPatch) -> PatchFeatures:
    """
    Extract topological features from a mesh patch.
    Placeholder: In production, this calls the full feature extractor.
    """
    n_faces = len(patch.local_faces)
    
    # Dummy ordered face sequence
    ordered_faces = patch.local_faces.copy()
    
    # Dummy topology descriptor (10-D)
    topo_desc = np.zeros((n_faces, 10), dtype=np.float32)
    
    return PatchFeatures(
        ordered_face_sequence=ordered_faces,
        topology_descriptor=topo_desc,
        adjacency_graph_edges=np.zeros((n_faces, 3), dtype=np.int64),
        face_normals=np.zeros((n_faces, 3), dtype=np.float32),
    )


def convert_to_training_format(patch_features: List[PatchFeatures]) -> Dict:
    """Convert patch features to numpy training format."""
    return {
        "n_patches": len(patch_features),
        "ordered_faces": [pf.ordered_face_sequence for pf in patch_features],
        "topology_descriptors": [pf.topology_descriptor for pf in patch_features],
    }


def save_training_data(data: Dict, output_path: Path):
    """Save training data as NPZ file."""
    np.savez(
        str(output_path),
        n_patches=data["n_patches"],
        ordered_faces=np.array(data["ordered_faces"], dtype=object),
        topology_descriptors=np.array(data["topology_descriptors"], dtype=object),
    )


# =============================================================================
# UV DENSITY VALIDATION (DEFENSE #2)
# =============================================================================

def validate_uv_density(
    num_faces: int,
    num_patches: int,
    min_density: float = DEFAULT_MIN_PATCH_DENSITY,
) -> Tuple[bool, float]:
    """
    Validate UV patch density.
    
    Returns:
        (is_valid, density)
    """
    density = num_faces / max(num_patches, 1)
    is_valid = density >= min_density
    return is_valid, density


def validate_patches_per_mesh(
    num_patches: int,
    max_patches: int = DEFAULT_MAX_PATCHES_PER_MESH,
) -> bool:
    """
    Validate that number of patches per mesh is within transformer context.
    
    Returns:
        bool: True if within limits
    """
    return num_patches <= max_patches


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    min_patch_density: float = DEFAULT_MIN_PATCH_DENSITY,
    max_patches_per_mesh: int = DEFAULT_MAX_PATCHES_PER_MESH,
) -> Dict:
    """
    Main batch preprocessing pipeline with UV density kill switch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mesh files
    mesh_files = []
    for ext in ["*.obj", "*.stl", "*.ply", "*.fbx"]:
        mesh_files.extend(input_dir.glob(ext))
    
    stats = {
        "total": len(mesh_files),
        "processed": 0,
        "failed": 0,
        "discarded_by_uv_density": 0,
        "discarded_by_too_many_patches": 0,
        "discarded_by_invalid_patch": 0,
        "total_patches": 0,
        "total_faces": 0,
        "errors": [],
        "discarded_records": [],
    }
    
    total_faces = 0
    total_patches = 0
    
    for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
        try:
            print(f"\n[■] Processing: {mesh_file.name}")
            
            # Load mesh
            verts, faces = load_mesh(mesh_file)
            print(f"    ├── Loaded: {len(verts)} vertices, {len(faces)} faces")
            
            # Extract UVs
            uvs, segmentation_mask = extract_uvs_and_segmentation(mesh_file)
            num_patches = len(uvs)
            print(f"    ├── UV islands: {num_patches}")
            
            # === DEFENSE #2: UV Density Kill Switch ===
            is_valid, density = validate_uv_density(
                len(faces), num_patches, min_patch_density
            )
            
            if not is_valid:
                print(f"    ├── ⚠️  UV DENSITY REJECTED: "
                      f"density={density:.1f} (< {min_patch_density})"
                      f", faces/patch too fragmented. SKIPPING.")
                stats["discarded_by_uv_density"] += 1
                stats["discarded_records"].append({
                    "file": str(mesh_file),
                    "reason": "low_uv_density",
                    "faces": len(faces),
                    "patches": num_patches,
                    "density": density,
                })
                continue
            else:
                print(f"    ├── UV density: {density:.1f} faces/patch [✅ PASS]")
            
            # === DEFENSE #2: Max Patches per Mesh ===
            if not validate_patches_per_mesh(num_patches, max_patches_per_mesh):
                print(f"    ├── ⚠️  TOO MANY PATCHES: "
                      f"{num_patches} (> {max_patches_per_mesh}). SKIPPING.")
                stats["discarded_by_too_many_patches"] += 1
                stats["discarded_records"].append({
                    "file": str(mesh_file),
                    "reason": "too_many_patches",
                    "faces": len(faces),
                    "patches": num_patches,
                })
                continue
            
            total_faces += len(faces)
            total_patches += num_patches
            
            # Normalize mesh
            normalized_verts, bbox_info = normalize_mesh(verts)
            print(f"    ├── Normalized: bbox scale = {bbox_info['scale']:.4f}")
            
            # Split into patches
            patches = split_mesh_into_patches(
                normalized_verts, faces, uvs, segmentation_mask
            )
            print(f"    ├── Split into {len(patches)} patches")
            
            # Extract features
            patch_features = []
            for i, patch in enumerate(patches):
                if len(patch.local_faces) < DEFAULT_MIN_PATCH_FACES:
                    print(f"    ├── Patch {i}: {len(patch.local_faces)} faces [❌ SKIP]")
                    stats["discarded_by_invalid_patch"] += 1
                    continue
                
                features = extract_patch_features(patch)
                patch_features.append(features)
                print(f"    ├── Patch {i}: {len(patch.local_faces)} faces [✅]")
            
            if len(patch_features) == 0:
                print(f"    └── ❌ No valid patches after filtering")
                stats["failed"] += 1
                continue
            
            # Convert and save
            training_data = convert_to_training_format(patch_features)
            output_file = output_dir / f"{mesh_file.stem}.npz"
            save_training_data(training_data, output_file)
            print(f"    └── Saved: {output_file}")
            
            stats["processed"] += 1
            stats["total_patches"] += len(patch_features)
            stats["total_faces"] += sum(len(p.local_faces) for p in patches)
            
        except Exception as e:
            print(f"    └── ❌ Error: {e}")
            stats["failed"] += 1
            stats["errors"].append({
                "file": str(mesh_file),
                "error": str(e),
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files:              {stats['total']}")
    print(f"Successfully processed:   {stats['processed']}")
    print(f"Failed:                   {stats['failed']}")
    print(f"Discarded (UV density):   {stats['discarded_by_uv_density']}")
    print(f"Discarded (too many):     {stats['discarded_by_too_many_patches']}")
    print(f"Discarded (invalid):      {stats['discarded_by_invalid_patch']}")
    print(f"Total patches generated:  {stats['total_patches']}")
    print(f"Total faces:              {stats['total_faces']}")
    
    if stats['processed'] > 0:
        print(f"Avg faces/mesh:           {stats['total_faces'] / stats['processed']:.1f}")
        print(f"Avg patches/mesh:         {stats['total_patches'] / stats['processed']:.1f}")
        avg_density = stats['total_faces'] / max(stats['total_patches'], 1)
        print(f"Overall UV density:       {avg_density:.1f} faces/patch")
    print("=" * 60)
    
    # Quality warnings
    if stats['processed'] > 0:
        avg_patches = stats['total_patches'] / stats['processed']
        if avg_patches > 50:
            print(f"\n⚠️  WARNING: Avg patches/mesh ({avg_patches:.1f}) > 50.")
            print("     Transformer context may be insufficient.")
        
        discard_rate = (stats['discarded_by_uv_density'] + 
                        stats['discarded_by_too_many_patches']) / max(stats['total'], 1)
        if discard_rate > 0.3:
            print(f"\n🔴  CRITICAL: {discard_rate*100:.1f}% models discarded.")
            print("     Dataset quality too low. Consider:")
            print("       - Switching to ABC Dataset")
            print("       - Using RizomUV for better auto-unwrap")
            print("       - Lowering min_patch_density threshold")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocessing with UV density kill switch"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing raw mesh files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for processed training data"
    )
    parser.add_argument(
        "--min-patch-density", type=float, default=DEFAULT_MIN_PATCH_DENSITY,
        help=f"Minimum faces per UV patch (default: {DEFAULT_MIN_PATCH_DENSITY})"
    )
    parser.add_argument(
        "--max-patches-per-mesh", type=int, default=DEFAULT_MAX_PATCHES_PER_MESH,
        help=f"Maximum patches per mesh (default: {DEFAULT_MAX_PATCHES_PER_MESH})"
    )
    parser.add_argument(
        "--save-stats", type=str, default="preprocessing_stats.json",
        help="Path to save processing statistics"
    )
    
    args = parser.parse_args()
    
    stats = preprocess_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        min_patch_density=args.min_patch_density,
        max_patches_per_mesh=args.max_patches_per_mesh,
    )
    
    # Save stats
    stats_path = Path(args.save_stats)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n✅ Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
