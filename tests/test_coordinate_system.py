"""
Validate Blender Z-up <-> Training Y-up coordinate conversion.
"""
import numpy as np

def zup_to_yup(v):
    """Blender Z-up -> Standard Y-up (deep learning convention)"""
    x, y, z = v
    return np.array([x, z, -y])

def yup_to_zup(v):
    """Standard Y-up -> Blender Z-up"""
    x, y, z = v
    return np.array([x, -z, y])

def test_bijective():
    v = np.array([1.0, 2.0, 3.0])
    v2 = zup_to_yup(v)
    v3 = yup_to_zup(v2)
    assert np.allclose(v, v3), f"Round-trip failed: {v} -> {v2} -> {v3}"
    print("OK: Z-up <-> Y-up conversion is bijective")

if __name__ == "__main__":
    test_bijective()
