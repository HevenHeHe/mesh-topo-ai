# 数据管道调研报告 — Phase 2

> 调研时间: 2026-05-21
> 调研目标: 为训练提供可用的高质量 mesh + UV 数据源

---

## 1. ABC Dataset

- **规模**: ~1M CAD 模型
- **GitHub**: https://github.com/deep-geometry/abc-dataset
- **官网**: https://deep-geometry.github.io/abc-dataset/
- **格式**: 主要是 STEP 文件（CAD B-rep），无预生三角形 mesh
- **下载**: 需要通过官网申请或从论文附录获取
- **处理难度**: 高。需要从 STEP 三角化，且 GitHub repo 上写着"Processing software will be available soon"，表明工具链不完善。
- **UV**: CAD 模型无原生 UV，需要自动展开。

## 2. Fusion 360 Gallery Dataset ⭐ 推荐

- **GitHub**: https://github.com/AutodeskAILab/Fusion360GalleryDataset
- **规模**: ~20,000 原始设计
- **提供工具**: Python 工具链（基于 Fusion 360 API），支持格式转换、B-Rep 遍历、三角化

### 可用子集

| 子集 | 规模 | 下载链接 | 说明 |
|------|------|----------|------|
| **Segmentation** | 35,680 parts / 3.1 GB | https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip | 已包含三角形 mesh！每个模型有操作分割标签（Extrude/Fillet/Chamfer） |
| Segmentation Extended STEP | 42,912 STEP / 483 MB | https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1_extended_step.zip | 扩展的 STEP 文件（包含上面三角化失败的样本） |
| Reconstruction | 8,625 sequences / 2.0 GB | https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip | 设计序列数据，不适合本项目 |
| Assembly | 8,251 assemblies / 154,468 parts | 需通过工具下载 | 装配体数据，复杂度过高 |

### 关键发现

**Segmentation 子集已经包含三角形 mesh**！文档中提到：
> "additional files for which triangle meshes with close to 2500 edges could not be created"

这意味着大多数样本**已经有预生成的三角形 mesh**。这是目前最可直接使用的数据源。

### 格式细节
- 原始格式: `.smt` (Fusion 360 原生)
- 派生格式: 三角形 mesh (具体格式待解压确认)
- 标注: 每个面的建模操作类型（Extrude, Fillet, Chamfer, Revolve 等）

### 与本项目的关系
- ✅ 已有三角形 mesh — 无需从 STEP 三角化
- ⚠️ mesh 是自动生成的，拓扑质量未知，可能不是工业级布线
- ❌ 无原生 UV 信息 — 需要自动展开
- ✅ 操作标签可能帮助理解几何结构

---

## 3. 推荐数据管道方案

### 阶段 1: 快速验证（本周）
使用**Fusion 360 Segmentation Dataset**的已有三角形 mesh：
1. 下载 s2.0.1.zip (3.1 GB)
2. 解压并探索 mesh 格式（可能是 .obj/.ply/.stl 或 JSON 描述）
3. 用 xatlas 或 Blender 自动展开 UV
4. 运行本项目的 UV Patch Segmentation + Tokenizer 验证管道

### 阶段 2: 质量提升（下周）
若 Fusion 360 mesh 质量不足，转向 **ABC Dataset + 精细三角化**：
1. 从 ABC 获取 STEP 文件
2. 使用 FreeCAD / OpenCASCADE 进行高质量三角化（控制四边形比例、曲率适应）
3. 自动展开 UV
4. 构建训练集

### 工具链
- **三角化**: OpenCASCADE (PythonOCC) 戓 FreeCAD
- **UV 展开**: xatlas-python 戓 Blender 内置算法
- **Mesh I/O**: trimesh / pymeshio / assimp

---

## 4. 结论

**立即行动项**: 下载 Fusion 360 Segmentation s2.0.1.zip，验证其 mesh 格式和质量。若 mesh 可用，则这是最快的训练数据来源。

**备选方案**: 若 Fusion 360 mesh 质量过差，则采用 ABC STEP → 高质量三角化 → 自动 UV 展开的管道。
