# Mesh Topo AI — 项目阶段说明文档

> 版本: Phase 2 完成版  
> 最后更新: 2026-05-21  
> 负责人: Hermes CTO  
> 仓库: https://github.com/HevenHeHe/mesh-topo-ai

---

## 一、项目定位与目标

**Mesh Topo AI** 是一套基于深度学习的 3D 资产处理方案，目标是将凌乱的高模（High-poly）或点云数据，直接转化为具有工业级布线流向（Edge Flow）和原生 UV 分区（Native UV Segmentation）的低模资产，并以 Blender 插件形式交付。

### 核心价值主张

传统 3D 管线的痛点：
- **手工重拓扑（Retopology）** 耗时极长，一个角色模型可能需要数小时到数天
- **UV 展开** 依赖艺术家经验，自动展开工具（如 Smart UV Project）质量不稳定
- **高模 → 低模** 的映射需要反复迭代

Mesh Topo AI 的解法：
- 用**自回归 Transformer** 直接预测低模的几何 + 拓扑 + UV 分区
- 通过**数据驱动的 tokenizer** 将 mesh 转化为离散 token 序列
- 在 Blender 中一键完成"高模导出 → AI 推理 → 低模导入 + UV"

---

## 二、技术原理详解

### 2.1 方案演进：从 "Strips as Tokens" 到 "UV-Guided Face Cluster"

#### 原始方案的问题

项目初始设想采用 **"Strips as Tokens"**：将 mesh 用贪婪三角带剥离（Greedy Triangle Strip Peeling）转化为一维 token 序列。

但技术调研发现该方案存在**结构性矛盾**：

| 矛盾点 | 说明 |
|--------|------|
| **UV 盲** | 贪婪三角带以几何邻接为驱动，完全不感知 UV seams |
| **非唯一性** | 同一 mesh 存在多种 strip 分解方式，增加模型困惑度 |
| **复杂拓扑脆弱** | 工业模型常有非流形结构，strip 化会支离破碎 |
| **UV 边界断裂** | UV seam 处顶点必须拆分，强行打断三角带连续性 |

> **结论**: 标准贪婪三角带剥离与"原生 UV 分区"目标存在根本冲突，需要重新设计 tokenizer。

#### 方案 A：UV-Guided Face Cluster as Token

核心洞察：
> **UV seams 天然定义了 mesh 的结构化边界。沿 seams 将 mesh 切分为 patches（UV islands），每个 patch 内部的拓扑是连续的，patch 之间的边界是精确已知的。**

这带来了三个关键优势：
1. **序列长度大幅降低**：自回归模型操作的是 patch 序列（几十到几百个），而非面片序列（几千到几万个）
2. **UV 一致性天然保证**：每个 patch 就是一个 UV island，边界顶点在重建时只需按 seams 焊接
3. **工业级拓扑可行**：每个 patch 内部可以用 VQ-VAE 学习艺术家级别的面片排列模式

### 2.2 完整技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入层                                    │
│  High-Poly Mesh (.obj/.ply)  +  UV Coordinates                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 2: UV-Driven Patch Segmentation              │
│                                                                 │
│  1. UV Seam Detection                                           │
│     - 遍历所有 mesh edge                                        │
│     - 若相邻两面对同一边缘使用不同的 UV 顶点对 → 判定为 seam   │
│                                                                 │
│  2. Mesh Cutting                                                │
│     - 在 seam edges 处阻断 face adjacency                       │
│     - Flood-fill 提取连通分量 → 得到 Patches (UV islands)      │
│                                                                 │
│  3. Patch Localization                                          │
│     - 每个 patch 构建局部子 mesh                                │
│     - seam 顶点在 UV 空间中被自然复制（不同 UV 坐标）           │
│     - 记录 boundary edges 用于后续焊接                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: Face-Cluster VQ-VAE Tokenizer             │
│                                                                 │
│  Encoder (GNN on Face-Adjacency Graph)                          │
│     Input:  每个 patch 的面级几何特征                            │
│              - 法向 (3D)                                        │
│              - 面积 (1D)                                        │
│              - 重心 (3D)                                        │
│              - UV 面积 (1D)                                     │
│              - UV 拉伸比 (1D)                                   │
│              - 边界标记 (1D)                                    │
│     Output: 每个面的连续 latent 向量 z_e(x)                     │
│                                                                 │
│  Vector Quantization (VQ)                                       │
│     z_q(x) = argmin || z_e(x) - e_k ||                         │
│     - 使用 straight-through estimator 实现梯度回传              │
│     - Codebook 大小: 512 个离散码                               │
│                                                                 │
│  Decoder (GNN / MLP)                                            │
│     Input:  z_q(x)                                              │
│     Output: 重建的 face corners (3 vertices × 3 coords)        │
│                                                                 │
│  训练目标: L = L_recon + β · L_commitment                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: Autoregressive Transformer                │
│                                                                 │
│  - GPT-style Decoder-only Transformer                           │
│  - 输入: PointNet++ 编码的高模点云特征（全局条件）              │
│  - 输出: Patch token 序列的自回归预测                           │
│  - 条件化生成: 高模几何约束低模拓扑流向                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 4: Mesh Assembler                            │
│                                                                 │
│  1. Per-patch decode: tokens → reconstructed corners           │
│  2. Deduplicate vertices within each patch                     │
│  3. Merge patches: weld seam vertices by geometric proximity   │
│  4. Remove degenerate faces                                    │
│  5. Output: unified mesh with native UV islands                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 核心算法详解

#### UV Seam 检测算法

```python
# 伪代码
def detect_uv_seams(faces, uv_faces):
    edge_uv_pairs = {}  # mesh_edge -> set of uv_edges
    for each face:
        for each of 3 edges:
            mesh_edge = sorted(v0, v1)
            uv_edge = sorted(uv0, uv1)
            edge_uv_pairs[mesh_edge].add(uv_edge)
    
    # 若同一条 mesh edge 被相邻面映射到不同的 UV edge → seam
    seams = {edge for edge, uv_set in edge_uv_pairs.items() if len(uv_set) > 1}
    return seams
```

**关键理解**：UV seam 不是几何边界，而是**参数化不连续点**。同一个 3D 顶点在 UV 空间中被"撕裂"为多个 UV 顶点，这是 UV island 的本质。

#### Patch 切分算法

1. 构建面邻接图（face-adjacency graph）
2. 将 seam edges 对应的 face adjacencies **阻断**
3. 在阻断后的图上执行连通分量提取（flood-fill）
4. 每个连通分量 = 一个 patch

**顶点复制策略**：
- 在 patch 内部，每个 (global_vertex, global_uv) 对被视为唯一的局部顶点
- 这自然处理了 seam 处的 UV 分裂，无需显式"切"几何体

#### VQ-VAE Tokenizer

**编码器（当前为占位符，Phase 3 替换为 GNN）**：
- 输入: 每个 patch 的 face feature matrix (F_p × D)
- 处理: 线性投影 + tanh 激活
- 输出: 连续 latent vectors (F_p × latent_dim)

**向量量化**：
```python
# L2 距离计算
D = ||z_e||² + ||codebook||² - 2 * z_e @ codebook.T
code_index = argmin(D, axis=1)
z_q = codebook[code_index]
```

**解码器（当前为占位符）**：
- 输入: z_q (F_p × latent_dim)
- 输出: face corners (F_p × 3 × 3)

#### Mesh Assembler

重建流程：
1. **Deduplicate**: 将每个 patch 的 corner 列表展平，L2 聚类去重（阈值 1e-6）
2. **Merge**: 跨 patches 拼接所有顶点和面
3. **Weld**: 对几何位置相近的顶点执行焊接（阈值 1e-5），还原 seam 处的分裂
4. **Clean**: 删除退化面（任意两顶点相同）

### 2.4 坐标系处理

Blender 使用 **Z-up** 右手坐标系，而深度学习框架通常使用 **Y-up**。我们在数据管道中处理转换：

```python
# Blender Z-up → 训练 Y-up
yup = (x, z, -y)

# 训练 Y-up → Blender Z-up  
zup = (x, -z, y)
```

已验证为**双射变换**（bijective），无信息损失。

---

## 三、操作详解

### 3.1 环境要求

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 后端服务 + tokenizer |
| NumPy | 2.x | 数值计算 |
| Blender | 3.6+ | 插件运行环境 |
| FastAPI | 0.104+ | 后端 API（Phase 1） |
| PyTorch Geometric | 2.4+ | Phase 3 GNN 训练 |
| trimesh | 4.0+ | Mesh I/O（Phase 3） |

### 3.2 本地快速开始

#### 步骤 1: 克隆仓库

```bash
git clone https://github.com/HevenHeHe/mesh-topo-ai.git
cd mesh-topo-ai
```

#### 步骤 2: 运行单元测试（无需 Blender）

```bash
# 安装依赖
pip install numpy

# 运行所有测试
cd /path/to/mesh-topo-ai
python3 -m tests.test_coordinate_system
python3 -m tests.test_uv_patch_segmentation
python3 -m tests.test_vqvae_tokenizer
python3 -m tests.test_roundtrip
```

预期输出：
```
OK: Z-up <-> Y-up conversion is bijective
OK: Detected 2 seam edges on plane (expected >= 2)
OK: Segmented plane into 2 patches
OK: Cylinder segmented into 1 patch
OK: Roundtrip geometry verified for all patches
OK: VQ-VAE output shapes correct for all patches
OK: Code indices are valid discrete tokens
OK: Reconstruction loss computable and finite
OK: Tokenizer is deterministic
ROUNDTRIP PIPELINE: PASSED
```

#### 步骤 3: 启动后端服务

```bash
cd api-server
pip install -r requirements.txt
python main.py
```

服务将在 `http://127.0.0.1:8000` 启动，提供以下端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/infer` | POST | 上传 mesh，返回重建结果（当前为 mock） |
| `/info` | GET | 项目信息 |

测试：
```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": false, "mode": "mock"}
```

#### 步骤 4: 安装 Blender 插件

1. 打开 Blender 3.6+
2. `Edit > Preferences > Add-ons > Install...`
3. 选择 `blender-addon/__init__.py`
4. 勾选启用 "Mesh: Mesh Topo AI"
5. 按 `N` 打开 Sidebar → 找到 "Mesh Topo AI" 标签

**Mock 测试**（无需后端）：
1. 在 3D 视图中选择任意 mesh 对象
2. 点击 "Mock: Import Test Cylinder" → 验证 UI 和后处理流程

**完整测试**（需要后端运行）：
1. 确保后端在 `127.0.0.1:8000` 运行
2. 选择 mesh 对象
3. 点击 "Generate Low-Poly + UV" → 插件将：
   - 导出选中对象为临时 .obj/.ply
   - 异步发送 HTTP 请求到后端
   - 接收 base64 编码的重建 mesh
   - 导入 Blender 并匹配变换
   - 执行 "Merge by Distance" 后处理
   - 自动隐藏原始高模

### 3.3 项目目录结构

```
mesh-topo-ai/
├── README.md                          # 快速开始指南
├── .gitignore
│
├── blender-addon/
│   └── __init__.py                    # Blender 插件完整代码
│
├── api-server/
│   ├── main.py                        # FastAPI 后端（含 mock 推理）
│   └── requirements.txt               # Python 依赖
│
├── tokenizer/                         # 核心算法模块
│   ├── __init__.py
│   ├── mesh_utils.py                  # 边/面邻接、连通分量、量化
│   ├── uv_patch_segmentation.py       # UV seam 检测 + patch 切分
│   ├── vqvae_tokenizer.py            # Face-Cluster VQ-VAE
│   └── mesh_assembler.py             # Patches → 统一 mesh
│
├── tests/                             # 单元测试
│   ├── test_coordinate_system.py      # 坐标系转换验证
│   ├── test_uv_patch_segmentation.py  # Patch 分割测试
│   ├── test_vqvae_tokenizer.py       # Tokenizer 测试
│   └── test_roundtrip.py             # 端到端闭环测试
│
└── docs/
    ├── research-report-phase1.md      # Phase 1 技术调研报告
    ├── data-pipeline-report-phase2.md # Phase 2 数据管道报告
    └── PROJECT_GUIDE.md              # 本文档
```

---

## 四、当前状态与验证

### 4.1 Phase 1 ✅ 已完成

- Blender 插件外壳（UI + 导出/导入 + 异步 HTTP）
- FastAPI 后端 stub（mock 圆柱体响应）
- Tokenizer 骨架（类型定义 + 文档）
- 坐标系验证脚本

### 4.2 Phase 2 ✅ 已完成

- **UV Patch Segmentation**： seam 检测 + mesh 切割 + patch 提取
  - 人造平面测试：2 seams → 2 patches（各 4 面/6 顶点）
  - 圆柱体测试：1 patch（侧面连续）
  - 几何往返验证：local vertices 与 global vertices 精确匹配

- **Face Cluster VQ-VAE Tokenizer**：
  - 面级特征 10-D 基线（法向/面积/重心/UV 拉伸/边界标记）
  - 向量量化框架（codebook lookup + straight-through）
  - 编码器/解码器占位符（Phase 3 替换为 GNN）
  - 确定性验证、形状验证、重建损失计算

- **Mesh Assembler**：
  - 顶点去重（L2 聚类）
  - 跨 patches 焊接（几何邻近）
  - 退化面清理
  - **Roundtrip 验证通过**：8 面 → 2 patches → 8 tokens → 8 面重建

- **数据管道调研**：
  - 确认 Fusion 360 Gallery Segmentation 为首选项（35,680 预三角化 mesh）
  - ABC Dataset 为备选（STEP → 三角化管道）

### 4.3 技术债务与已知限制

| 限制 | 说明 | 解决阶段 |
|------|------|----------|
| VQ-VAE 使用随机权重 | 重建几何失真 | Phase 3（训练） |
| Encoder 是线性投影 | 无 GNN 消息传递 | Phase 3（PyG 实现） |
| Welding 基于纯几何距离 | 丢失了原始 global vertex ID 信息 | Phase 3（保留 weld 映射） |
| 无真实训练数据 | 仅测试合成数据 | Phase 2 后半（数据集） |
| Blender 插件使用同步导出 | 大文件可能阻塞 UI | Phase 4（异步导出优化） |

---

## 四、补充章：工程风险与防御性设计

> ⚠️ **本章节是对 Phase 2 实现的深度工程审查结果，标识出了 4 个严重风险并提供了对应的防御性方案。**
> 
> 完整详细文档见: `docs/RISK_ASSESSMENT.md`

### 4.1 风险盘点概览

| 编号 | 风险 | 等级 | 相关模块 | 防御措施 |
|------|------|------|----------|----------|
| RISK-1 | GNN Patch 局部重建时的"拓扑失忆"（微裂缝） | 🔴 CRITICAL | VQ-VAE Decoder | Topology Consistency Loss |
| RISK-2 | Fusion 360 数据集的"无 UV 危机"（UV 碎屑） | 🔴 CRITICAL | 数据管道 | UV Patch Density Kill Switch |
| RISK-3 | 自回归 Transformer 的"维数灾难与死锁" | 🟠 HIGH | Transformer | 层次化生成 + 边界条件化 |
| RISK-4 | Blender 插件的"IO 死锁" | 🟡 MEDIUM | Blender 插件 | 点云特征替代高模导出 |

### 4.2 防御性设计接口

以下防御措施已在代码库中提供接口：

1. **Topology Consistency Loss** (`tokenizer/vqvae_tokenizer.py`)
   - 计算相邻面共享顶点的重建坐标偏离
   - 训练时结入总损失: `L_total = L_recon + β·L_commitment + λ_topo·L_topo`
   - 预期效果: 共享顶点偏离 < 1e-6，无物理裂缝

2. **UV Density Kill Switch** (`tokenizer/scripts/batch_preprocess.py`)
   - 指标: `Density = 总面数 / UV Patches 数量`
   - 策略: Density < 20 → 直接丢弃
   - 同时检查: patches/mesh < 50（避免超出 transformer 上下文）

3. **拓扑升维焊接** (`tokenizer/mesh_assembler.py`)
   - 用 `global_vertex_remap` 作为主要焊接准则
   - 几何距离仅作为 fallback
   - 预期效果: 焊接成功率提升一个数量级

4. **点云特征替代方案** (待 Phase 4 实现)
   - 不发送完整高模 OBJ，只发送 PointNet++ 编码后的特征向量
   - 传输体积从 > 100MB 降至 < 1MB
   - Blender UI 无阻塞感

### 4.3 防御路线图

```
Phase 2 后半 数据验收
    │
    ├── [熔断] UV Patch Density < 20 → 丢弃
    └── [通过] 进入训练
            │
            ▼
Phase 3 VQ-VAE 训练
    │
    ├── [损失函数] L = L_recon + L_commitment + L_topo
    └── [验证] 共享顶点偏离 < 1e-6
            │
            ▼
Phase 3 Transformer 训练
    │
    ├── [边界条件化] Boundary conditioning
    └── [层次生成] Hierarchical generation
            │
            ▼
Phase 4 工程集成
    │
    └── [点云特征] 替代高模导出
```

---

## 五、后续计划

### Phase 2 后半（本周）

**目标**: 验证真实数据可用性，跑通第一批训练样本

1. **下载 Fusion 360 Segmentation Dataset**
   - 链接: https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip
   - 大小: 3.1 GB
   - 验证 mesh 格式和质量

2. **构建批量预处理脚本**
   ```
   raw_mesh/ → uv_unwrap/ → patches/ → tokens/
   ```
   - 用 xatlas / Blender 自动展开 UV
   - 批量运行 UV Patch Segmentation
   - 生成 VQ-VAE 训练用的 token 序列

3. **数据质量评估**
   - mesh 流形性检查
   - UV 展开质量（拉伸、重叠）
   - patch 数量分布统计

### Phase 3（2-4 周）

**目标**: 训练 VQ-VAE + Transformer 模型

1. **GNN Encoder/Decoder 实现**
   - 使用 PyTorch Geometric
   - 在 face-adjacency graph 上执行消息传递
   - GraphConv / GAT / TransformerConv 对比实验

2. **VQ-VAE 训练**
   - 重建损失（L2 chamfer）
   - Commitment loss（β = 0.25）
   - Codebook 更新（EMA）

3. **Transformer 训练**
   - GPT-style decoder-only
   - 条件输入: PointNet++ 高模点云编码
   - 自回归预测 patch token 序列

4. **训练基础设施**
   - PyTorch Lightning 训练框架
   - W&B 实验追踪
   - WSL GPU 环境配置（CUDA + PyTorch）

### Phase 4（2-3 周）

**目标**: 工程集成与发布

1. **ONNX 导出**
   - 将训练好的模型导出为 ONNX
   - 使用 onnxruntime-gpu 替代 PyTorch 依赖
   - 显著减小后端部署体积

2. **Blender 插件完善**
   - 异步导出（大文件不阻塞 UI）
   - 进度条和取消按钮
   - 错误处理和重试机制
   - 插件打包为 .zip

3. **性能优化**
   - 推理批处理
   - 显存管理（分层生成策略）
   - 缓存机制

4. **文档与示例**
   - 视频教程
   - 示例工作流
   - API 文档

---

## 六、关键决策记录 (ADR)

### ADR-001: 前端插件 + 本地后端解耦架构
- **决策**: Blender 插件通过 HTTP 与本地 FastAPI 服务通信
- **理由**: 避免在 Blender Python 中加载 PyTorch/ONNX；利用 WSL GPU 环境
- **状态**: ✅ 已实施

### ADR-002: 方案 A — UV-Guided Face Cluster as Token
- **决策**: 放弃原始 "Strips as Tokens"，改用 UV-driven patch segmentation + VQ-VAE
- **理由**: Strips 不感知 UV seams，与项目目标冲突；Face Clusters 天然对应 UV islands
- **状态**: ✅ 已实施

### ADR-003: Z-up ↔ Y-up 双射转换
- **决策**: 训练使用 Y-up，Blender 使用 Z-up，通过确定性矩阵转换
- **理由**: 兼容深度学习惯例和 Blender 原生坐标系
- **状态**: ✅ 已验证

### ADR-004: Fusion 360 作为首选训练数据源
- **决策**: 优先使用 Fusion 360 Gallery Segmentation（已三角化），ABC 为备选
- **理由**: Fusion 360 已提供 mesh，立即可用；ABC 需要额外三角化工具链
- **状态**: 🔄 待验证数据质量后确认

---

## 七、参考资源

### 核心论文
| 论文 | 链接 | 与本项目的关系 |
|------|------|----------------|
| MeshGPT | https://arxiv.org/abs/2311.15475 | VQ-VAE tokenizer 设计参考 |
| PolyGen | https://arxiv.org/abs/2002.10880 | 自回归 mesh 生成先驱 |
| ABC Dataset | https://arxiv.org/abs/1905.07553 | 备选数据源 |

### 数据集
| 数据集 | 下载 | 说明 |
|--------|------|------|
| Fusion 360 Segmentation | https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip | 35,680 预三角化 parts |
| ABC Dataset | https://deep-geometry.github.io/abc-dataset/ | ~1M CAD models (STEP) |

### 工具链
- **Blender Python API**: https://docs.blender.org/api/current/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **xatlas**: https://github.com/jpcy/xatlas (UV 展开)
- **trimesh**: https://trimesh.org/ (Mesh I/O)

---

*文档由 Hermes CTO 维护，随项目进展持续更新。*
