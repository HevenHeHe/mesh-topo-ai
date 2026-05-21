# Mesh Topo AI

AI-driven 3D asset processing pipeline that converts high-poly meshes or point clouds into low-poly assets with industrial-grade edge flow and native UV segmentation, delivered as a Blender add-on.

## Architecture

```
Blender Add-on (Python)  <--->  FastAPI Backend (WSL)  <--->  ONNX/PyTorch Inference
       |                               |                           |
    Export OBJ/PLY               Local port 8000            Autoregressive Transformer
    Import result                 Task scheduling              "Strips as Tokens"
```

## Quick Start (Phase 1 MVP)

### 1. Start Backend
```bash
cd api-server
pip install -r requirements.txt
python main.py
```

### 2. Install Blender Add-on
1. Open Blender 3.6+
2. Edit > Preferences > Add-ons > Install...
3. Select `blender-addon/__init__.py`
4. Enable "Mesh: Mesh Topo AI"

### 3. Run Mock Test
- Select any mesh object in Blender
- Open Sidebar (N) > "Mesh Topo AI"
- Click "Mock: Import Test Cylinder" to verify UI pipeline
- Click "Generate Low-Poly + UV" to test full export-infer-import loop (returns mock geometry)

## Roadmap

| Phase | Goal | ETA |
|-------|------|-----|
| 1 | MVP skeleton: Blender plugin + FastAPI stub + mock response | Now |
| 2 | Data pipeline: Tokenizer (UV-driven segmentation + strip extraction) | TBD |
| 3 | Model training: Transformer on simple furniture geometries | TBD |
| 4 | Integration: ONNX export + one-click install package | TBD |

## Cautions

- **Coordinate system**: Blender uses Z-up. The tokenizer normalizes to Y-up for training; conversion is handled during export/import.
- **UV seams**: Generated patch boundaries may produce duplicate vertices. Post-process uses "Merge by Distance".
- **VRAM**: Autoregressive generation of long sequences is memory-intensive. Hierarchical generation strategy is planned for Phase 3.

## License
MIT
