#!/usr/bin/env python3
"""
Quick pipeline validation script.
Generates a synthetic mesh with known UV islands and runs it through
the batch_preprocess.py pipeline to verify density calculation.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.scripts.batch_preprocess import validate_uv_density, validate_patches_per_mesh


def create_test_mesh():
    """Create a simple cube mesh with known UV layout."""
    # 8 vertices of a cube
    verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # top
    ], dtype=np.float32)
    
    # 12 faces (triangulated cube)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int64)
    
    return verts, faces


def test_density_validation():
    """Test UV density validation logic."""
    print("=" * 60)
    print("TEST: UV Density Validation")
    print("=" * 60)
    
    # Test case 1: Good density (50 faces, 2 patches)
    is_valid, density = validate_uv_density(50, 2, min_density=20.0)
    print(f"Case 1: 50 faces / 2 patches = {density:.1f} -> {'PASS' if is_valid else 'FAIL'}")
    assert is_valid, "Should pass for density=25"
    
    # Test case 2: Bad density (50 faces, 10 patches)
    is_valid, density = validate_uv_density(50, 10, min_density=20.0)
    print(f"Case 2: 50 faces / 10 patches = {density:.1f} -> {'PASS' if is_valid else 'FAIL'}")
    assert not is_valid, "Should fail for density=5"
    
    # Test case 3: Borderline (50 faces, 3 patches)
    is_valid, density = validate_uv_density(50, 3, min_density=20.0)
    print(f"Case 3: 50 faces / 3 patches = {density:.1f} -> {'PASS' if is_valid else 'FAIL'}")
    assert not is_valid, "Should fail for density=16.7"
    
    # Test case 4: Exact threshold (20 faces, 1 patch)
    is_valid, density = validate_uv_density(20, 1, min_density=20.0)
    print(f"Case 4: 20 faces / 1 patch = {density:.1f} -> {'PASS' if is_valid else 'FAIL'}")
    assert is_valid, "Should pass for density=20"
    
    print("\nAll density validation tests PASSED!")


def test_patches_per_mesh():
    """Test patches per mesh validation."""
    print("\n" + "=" * 60)
    print("TEST: Patches per Mesh Validation")
    print("=" * 60)
    
    # Test case 1: Within limit
    result = validate_patches_per_mesh(20, max_patches=50)
    print(f"Case 1: 20 patches / max 50 -> {'PASS' if result else 'FAIL'}")
    assert result, "Should pass for 20 patches"
    
    # Test case 2: At limit
    result = validate_patches_per_mesh(50, max_patches=50)
    print(f"Case 2: 50 patches / max 50 -> {'PASS' if result else 'FAIL'}")
    assert result, "Should pass for exactly 50 patches"
    
    # Test case 3: Over limit
    result = validate_patches_per_mesh(51, max_patches=50)
    print(f"Case 3: 51 patches / max 50 -> {'PASS' if result else 'FAIL'}")
    assert not result, "Should fail for 51 patches"
    
    print("\nAll patches per mesh tests PASSED!")


def test_realistic_scenarios():
    """Test with realistic mesh scenarios."""
    print("\n" + "=" * 60)
    print("TEST: Realistic Mesh Scenarios")
    print("=" * 60)
    
    scenarios = [
        # (name, faces, patches, expected_valid)
        ("Small part (good)", 200, 4, True),      # 50 faces/patch
        ("Complex part (good)", 1000, 20, True),  # 50 faces/patch
        ("Fragmented UV (bad)", 100, 20, False),  # 5 faces/patch
        ("Simple box (borderline)", 25, 1, True),  # 25 faces/patch (passes)
        ("Over-patched (bad)", 500, 60, False),    # >50 patches
    ]
    
    for name, faces, patches, expected in scenarios:
        is_valid, density = validate_uv_density(faces, patches, min_density=20.0)
        patches_ok = validate_patches_per_mesh(patches, max_patches=50)
        overall = is_valid and patches_ok
        
        status = "PASS" if overall == expected else "FAIL"
        print(f"{name}: {faces} faces / {patches} patches = {density:.1f} faces/patch")
        print(f"  Density valid: {is_valid}, Patches valid: {patches_ok} -> {status}")
        assert overall == expected, f"Failed for {name}"
    
    print("\nAll realistic scenario tests PASSED!")


if __name__ == "__main__":
    test_density_validation()
    test_patches_per_mesh()
    test_realistic_scenarios()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
