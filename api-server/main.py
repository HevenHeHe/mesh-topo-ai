"""
Mesh Topo AI - FastAPI Backend Stub
Phase 1: Returns mock low-poly mesh for architecture validation.
Phase 2: Will integrate real ONNX/PyTorch inference.
"""
import os
import tempfile
import base64
import json
import math
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="Mesh Topo AI API",
    description="AI-driven retopology & native UV generation",
    version="0.1.0"
)

# ---------------------------------------------------------------------------
#  Mock Geometry Generator (Phase 1)
# ---------------------------------------------------------------------------
def generate_mock_cylinder() -> bytes:
    """Generate a simple ASCII PLY with a UV-unwrapped cylinder-like shape."""
    # 16 vertices around, 2 caps = 18 vertices
    # Simplified: just a 16-sided prism
    verts = []
    faces = []
    n = 16
    h = 1.0
    r = 1.0

    # Side vertices (bottom ring, top ring)
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        verts.append((x, y, -h / 2))
        verts.append((x, y, h / 2))

    # Side faces (quads, split into tris)
    for i in range(n):
        v0 = 2 * i
        v1 = 2 * ((i + 1) % n)
        v2 = 2 * ((i + 1) % n) + 1
        v3 = 2 * i + 1
        faces.append((v0, v1, v2))
        faces.append((v0, v2, v3))

    # Cap centers
    bottom_center = len(verts)
    verts.append((0.0, 0.0, -h / 2))
    top_center = len(verts)
    verts.append((0.0, 0.0, h / 2))

    # Cap faces
    for i in range(n):
        v0 = 2 * i
        v1 = 2 * ((i + 1) % n)
        faces.append((bottom_center, v1, v0))
        v2 = 2 * i + 1
        v3 = 2 * ((i + 1) % n) + 1
        faces.append((top_center, v2, v3))

    # Build PLY ASCII
    header = f"""ply
format ascii 1.0
element vertex {len(verts)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
    body_lines = []
    for v in verts:
        body_lines.append(f"{v[0]} {v[1]} {v[2]}")
    for f in faces:
        body_lines.append(f"3 {f[0]} {f[1]} {f[2]}")

    ply_bytes = (header + "\n".join(body_lines) + "\n").encode("ascii")
    return ply_bytes

# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Mesh Topo AI backend is running", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": False, "mode": "mock"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Phase 1 stub:
    - Accepts any .obj/.ply upload
    - Ignores content (no real inference)
    - Returns a mock cylinder as base64-encoded PLY
    """
    ext = os.path.splitext(file.filename or "input.obj")[1].lower()
    if ext not in {".obj", ".ply", ".stl", ".fbx"}:
        raise HTTPException(400, detail=f"Unsupported file format: {ext}")

    # In Phase 2, we will read the file, run tokenizer + model, and return result.
    # For now, just consume the file and return mock geometry.
    _ = await file.read()

    mesh_bytes = generate_mock_cylinder()
    mesh_b64 = base64.b64encode(mesh_bytes).decode("ascii")

    return JSONResponse(content={
        "status": "success",
        "mode": "mock",
        "format": "ply",
        "mesh_b64": mesh_b64,
        "metadata": {
            "note": "This is a Phase 1 mock response. Real inference not yet integrated.",
            "original_filename": file.filename,
        }
    })

@app.get("/info")
def info():
    return {
        "project": "mesh-topo-ai",
        "pipeline_stages": [
            "1. Import high-poly / point cloud",
            "2. Tokenizer: UV-driven patch segmentation + strip extraction",
            "3. Transformer inference (autoregressive strip generation)",
            "4. Decode strips to mesh + native UV islands",
            "5. Export to Blender"
        ],
        "current_stage": 1,
        "supported_formats_in": ["obj", "ply"],
        "supported_formats_out": ["ply"],
    }

# ---------------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
