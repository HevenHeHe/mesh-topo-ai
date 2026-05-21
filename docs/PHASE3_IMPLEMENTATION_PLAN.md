# Phase 3 实施计划: VQ-VAE + Transformer 训练路线图

> 版本: v1.0  
> 状态: 待审批  
> 创建: 2026-05-21  
> 依据: `docs/RISK_ASSESSMENT.md` 风险盘点 + CTO 牵头人识别  
> 预期周期: 2-4 周  
> 负责人: Hermes CTO / AI 工程团队

---

## 一、Phase 3 总览

### 1.1 核心目标

**Phase 3 的唯一目标**：训练出一个能够将高模点云条件转化为**低模带原生 UV 分区** 的自回归 Transformer 生成模型。

具体分为三个子目标：
1. **基座 VQ-VAE**: 学习高质量的 patch token 化表示，重建误差 < 1e-3
2. **自回归 Transformer**: 学习 patch token 序列的分布，生成结果流形率 > 95%
3. **条件化管线**: 点云特征 → patch 序列的端到端工程

### 1.2 范围边界

**在范围内**：
- VQ-VAE Encoder/Decoder 的 PyTorch Geometric 实现
- 自回归 Transformer 的实现与训练
- 数据管道的批量预处理与质量监控
- 模型验证与评估脚本

**不在范围内**（Phase 4）：
- Blender 插件前端优化
- ONNX 导出与部署
- 模型服务化（FastAPI 生产环境）
- 用户文档与视频教程

### 1.3 里程碑时间表

| 周次 | 里程碑 | 验收标准 | 风险等级 |
|------|--------|----------|----------|
| Week 1 | 数据管道落地 + 基线训练 | 预处理 500+ mesh，训练 1 epoch | 🔴 CRITICAL |
| Week 2 | VQ-VAE 正式训练 | 重建 L2 < 1e-3，共享顶点偏离 < 1e-6 | 🔴 CRITICAL |
| Week 3 | Transformer 实现 + 初始训练 | 能生成有效的 patch 序列 | 🟠 HIGH |
| Week 4 | 联合训练 + 评估优化 | 端到端流形率 > 95% | 🟠 HIGH |

---

## 二、Week 1: 数据管道落地 — 熔断机制与基准训练

### 2.1 任务清单

#### 任务 1.1: 下载并验证数据集

**目标**: 获取可用的训练数据

**步骤**:
1. 下载 Fusion 360 Gallery Segmentation Dataset (3.1 GB)
   ```bash
   wget https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip
   unzip s2.0.1.zip -d data/fusion360/
   ```
2. 随机抽取 100 个样本进行手动检查：
   - 流形率：所有面都是三角形
   - UV 展开状态：是否已有 UV
   - 边界质量：碰撞体/干涉检查

3. 记录样本质量报告（近乎所有都没有展开好的 UV）

**验收标准**:
- [ ] 下载完成，文件完整性验证通过
- [ ] 100 个样本检查报告存档在 `data/fusion360/sample_report.json`
- [ ] 确认数据集格式与预期一致

#### 任务 1.2: UV 自动展开与熔断过滤

**目标**: 批量生成工业级 UV 分区

**步骤**:
1. 安装 xatlas + Blender 自动展开流水线
   ```bash
   pip install xatlas
   # Blender headless: blender -b -P scripts/auto_unwrap.py
   ```
2. 对每个 mesh 运行自动 UV 展开
3. 运行 `batch_preprocess.py` 并记录统计

**关键配置**:
```python
# batch_preprocess.py 默认参数
min_patch_density = 20.0      # CTO 判定：不是 50，而是 20
max_patches_per_mesh = 50     # Transformer 上下文窗口限制
```

**验收标准**:
- [ ] 成功展开 UV 的比例 > 90%
- [ ] 丢弃率 < 30%（若 > 30%需要切换到 ABC Dataset）
- [ ] 平均 patches/mesh < 20
- [ ] 平均 faces/patch > 20
- [ ] 最大 patches/mesh < 50

**备选方案**（丢弃率 > 30%时触发）:
- 转向 ABC Dataset
- 使用 RizomUV 级别的自动展开工具
- 合成数据增强

#### 任务 1.3: 基准训练脚本

**目标**: 验证 VQ-VAE 能够在真实数据上收敛

**步骤**:
1. 实现最小可行的 VQ-VAE 训练脚本 (`train_vqvae.py`)
2. 使用小规模子集（50 mesh）进行 5 epoch 快速验证
3. 监控损失曲线

**验收标准**:
- [ ] 能够完成一个 epoch 而不报错
- [ ] 重建损失在 5 epoch 内下降 > 50%
- [ ] 无 NaN/Inf 出现

### 2.2 Week 1 风险与应对

**风险**: xatlas 在复杂 CAD 模型上崩溃  
**应对**: 使用 Blender headless 模式作为 fallback

**风险**: 数据集 UV 展开质量过差  
**应对**: 使用 ABC Dataset + RizomUV

---

## 三、Week 2: VQ-VAE 正式训练 — 防御性设计落地

### 3.1 任务清单

#### 任务 2.1: PyTorch Geometric Encoder/Decoder 实现

**目标**: 替换 Phase 2 的 NumPy 占位符，实现真正的 GNN

**架构设计**:
```
Encoder (GNN):
  Input: Face features (F, D=10) + Edge index (2, E)
  Layers:
    - GraphConv(in=10, out=64) + ReLU
    - GraphConv(in=64, out=128) + ReLU
    - GraphConv(in=128, out=latent_dim=32)  # z_e(x)
  Output: Per-face latent vectors (F, 32)

Vector Quantization:
  Codebook: (K=512, D=32)
  更新: EMA 或 straight-through

Decoder (GNN/MLP):
  Input: z_q(x) (F, 32) + Edge index (2, E)
  Layers:
    - GraphConv(in=32, out=128) + ReLU
    - GraphConv(in=128, out=64) + ReLU
    - Linear(in=64, out=9)  # 3 vertices * 3 coords
  Output: Reconstructed corners (F, 3, 3)
```

**步骤**:
1. 安装 PyTorch Geometric
   ```bash
   pip install torch-geometric torch-scatter torch-sparse
   ```
2. 实现 `models/vqvae_gnn.py`
3. 实现训练 loop (`train_vqvae.py`)
4. 测试单个 patch 的前向/反向

**验收标准**:
- [ ] Encoder/Decoder 可以处理变长长度的 patch (3 ~ 200 faces)
- [ ] 重建输出形状正确 (F, 3, 3)
- [ ] 可以计算梯度，无报错

#### 任务 2.2: Topology Consistency Loss 整合

**目标**: 防止微裂缝，确保共享顶点坐标一致

**核心代码**（已在 `vqvae_tokenizer.py` 提供接口）:
```python
def compute_topology_consistency_loss(patch, quantized):
    # 已实现在 tokenizer/vqvae_tokenizer.py
    # 返回共享顶点重建坐标的平均 L2 偏离
    pass
```

**训练时总损失**（CTO 修正版）:
```python
# 渐进式权重调度！不要一开始就拉满
lambda_topo = min(1.0, 0.1 + epoch * 0.02)  # epoch 0: 0.1, epoch 45: 1.0

L_total = L_recon + beta * L_commitment + lambda_topo * L_topo
```

**验收标准**:
- [ ] 训练过程中无死亡代码
- [ ] 共享顶点偏离 < 1e-6（在验证集上）
- [ ] 重建网格流形率 > 98%

**风险**: 过早加入 L_topo 可能导致模型不收敛  
**防御**: 使用渐进式权重，epoch < 10 时仅 0.1

#### 任务 2.3: Codebook 管理

**目标**: 防止 Codebook Collapse（大量 dead codes）

**策略**:
1. **EMA 更新**:
   ```python
   # 使用指数移动平均更新 codebook
   codebook = gamma * codebook + (1 - gamma) * new_means
   ```
2. **Code Restart** (每 50 epoch):
   - 检测未使用的 code（使用次数 < threshold）
   - 重置为随机样本中的 latent 向量
3. **Commitment Loss**:
   ```python
   L_commitment = beta * ||z_e(x) - sg[z_q(x)]||^2
   beta = 0.25  # 固定
   ```

**验收标准**:
- [ ] Codebook 利用率 > 70% (每个 code 在一个 epoch 中至少被使用一次)
- [ ] 重建损失 < 1e-3
- [ ] Perplexity 稳定（无突然跌落）

#### 任务 2.4: GNN Over-smoothing 防御

**目标**: 防止 GNN 层数过深导致 latent 趨于相同

**策略**:
1. 限制 GNN 深度 <= 4 层
2. 使用 GAT (Graph Attention) 替代标准 GraphConv
3. 添加 residual connection + layer normalization
4. 监控 latent 向量的方差

**验收标准**:
- [ ] 每一层的输出方差 > 0.1
- [ ] 最后一层输出的平均值与标准差保持稳定

### 3.2 Week 2 训练资源需求

- GPU: RTX 3060+ (12GB VRAM)
- 时间: 每个 epoch ~ 30 分钟，计划 100 epoch
- 存储: ~ 10 GB (缓冲数据 + 检查点)
- 工具: PyTorch Lightning + Weights & Biases

---

## 四、Week 3: Transformer 实现与训练

### 4.1 任务清单

#### 任务 3.1: Patch Token 序列化

**目标**: 将 mesh 转化为 Transformer 可处理的 token 序列

**步骤**:
1. 每个 mesh 转化为 patch token 序列:
   ```python
   # mesh: [patch_0, patch_1, ..., patch_N]
   # 每个 patch 转化为 codebook index
   tokens = [vq_vae.encode(patch) for patch in mesh.patches]
   # tokens: [int_0, int_1, ..., int_N]
   ```
2. 构建训练数据集 (`MeshTokenDataset`)
3. 实现 DataLoader，支持 padding 和 masking

**验收标准**:
- [ ] 能够批量处理变长序列
- [ ] 序列最长度 <= 512 tokens
- [ ] 数据集大小 > 1000 mesh

#### 任务 3.2: GPT-Style Transformer 实现

**目标**: 实现自回归生成模型

**架构设计**:
```
Transformer (Decoder-only):
  - Layers: 12
  - Hidden dim: 512
  - Attention heads: 8
  - Dropout: 0.1
  - Max sequence length: 512

Input:
  - [START] token
  - Patch token sequence
  - [END] token

Output:
  - Next token prediction

损失函数:
  - CrossEntropyLoss 在 token 预测上
```

**步骤**:
1. 实现 `models/transformer.py`
2. 实现训练脚本 (`train_transformer.py`)
3. 快速验证: 10 epoch 内收敛

**验收标准**:
- [ ] 能够预测出有效的 patch token 序列
- [ ] 每个生成的 token 都在 codebook 范围内
- [ ] 生成序列的长度合理

#### 任务 3.3: 边界条件化 (Boundary Conditioning)

**目标**: 让 Transformer 知道哪些边界必须对齐

**实现**:
在每个 patch token 中嵌入边界元数据：
```python
# 边界特征（作为额外的输入向量）
boundary_features = {
    "boundary_vertex_count": int,      # 该 patch 边界上的顶点数
    "seam_edge_count": int,            # 该 patch 的 seam edge 数量
    "is_closed_loop": bool,            # 边界是否形成闭合环
    "neighbor_patch_ids": List[int],   # 相邻 patch 的索引（padding 到固定长度）
    "shared_boundary_length": float,   # 与前置 patch 共享边界的长度
}

# 在 Transformer 输入层拼接
input_embedding = [
    boundary_emb(boundary_features),  # (D_boundary,)
    patch_emb(patch_token),            # (D_patch,)
]
```

**验收标准**:
- [ ] 生成的网格在 patch 边界处的洪差 < 1e-3
- [ ] 生成结果的流形率 > 95%

#### 任务 3.4: 级联 Pipeline 回退 (Cascade Fallback)

**目标**: 如果自回归效果不佳，启动级联管线

**触发条件**（自动监控）:
```python
trigger_conditions = {
    "patch_mismatch_rate": "> 15%",      # 边界不匹配的 patch 比例
    "average_seam_gap": "> 1e-3",        # patch 边界平均缝隙
    "manifold_rate": "< 90%",            # 生成 mesh 的流形率
}
```

**级联策略**:
1. Transformer 只生成粗骨架（大致面片排列 + UV island 布局）
2. 对每个 patch 内部，使用可微分优化器精化
3. 这样 Transformer 的误差被可微分优化器缓解

**验收标准**:
- [ ] 级联模式下生成结果流形率 > 90%
- [ ] 比单独 Transformer 模式的几何质量提升 > 20%

### 4.2 Week 3 风险与应对

**风险**: Transformer 上下文窗口不足  
**应对**: 减少 patch 数量（增加 patch size）或使用长序列 Transformer（Longformer/Performer）

**风险**: 生成过程中出现无效 token  
**应对**: 添加后处理过滤器，将无效索引映射到最近的有效 code

---

## 五、Week 4: 联合训练与端到端验证

### 5.1 任务清单

#### 任务 4.1: 点云条件化

**目标**: 让 Transformer 能够感知高模几何

**架构**:
```
条件化管线:
  1. 对高模进行 Poisson Disk Sampling 采样
  2. 使用 PointNet++ 编码点云为特征向量 (256-D)
  3. 将特征向量作为 Transformer 的条件输入
  4. Transformer 在生成每个 patch token 时都参考该特征
```

**验收标准**:
- [ ] 点云采样保留几何特征率 > 95%
- [ ] 条件化生成 vs 无条件化生成的几何相似度提升 > 30%

#### 任务 4.2: 端到端流管线

**目标**: 整合数据管道、VQ-VAE、Transformer 为完整管线

**步骤**:
1. 实现 `pipeline/inference.py`:
   ```python
   def generate_lowpoly(highpoly_mesh, vqvae, transformer, pointnet):
       # 1. 点云采样 + 编码
       point_cloud = sample_point_cloud(highpoly_mesh, n_points=2048)
       condition = pointnet.encode(point_cloud)
       
       # 2. Transformer 生成 patch tokens
       patch_tokens = transformer.generate(condition, max_length=50)
       
       # 3. VQ-VAE 解码每个 patch
       patches = [vqvae.decode(token) for token in patch_tokens]
       
       # 4. Mesh Assembler 合并
       mesh = assemble_mesh_from_quantized_patches(patches, use_topology_weld=True)
       
       return mesh
   ```
2. 在 10 个测试样本上运行
3. 人工审查生成质量

**验收标准**:
- [ ] 流管线可以在 1 分钟内完成单个 mesh 的生成
- [ ] 生成结果是有效的 mesh 文件
- [ ] 不再需要手工干预

#### 任务 4.3: 评估脚本与审美标准

**目标**: 建立量化评估体系

**指标**:
| 指标 | 计算方法 | 目标值 |
|------|----------|--------|
| 流形率 | 检查每个边是否被恰好两个面共享 | > 95% |
| UV 拉伸 | 计算 UV 面积与 3D 面积的比值 | < 2.0 |
| 几何质量 | Chamfer Distance 到原始高模 | < 1e-2 |
| 上下文一致性 | 生成结果与高模的洪差 | < 5% |
| 生成速度 | 每个 mesh 的生成时间 | < 60s |

**验收标准**:
- [ ] 评估脚本可以自动运行
- [ ] 输出 JSON 格式的评估报告
- [ ] 所有指标均可计算

### 5.2 Week 4 风险与应对

**风险**: 端到端流管线性能不达标  
**应对**: 实现模型量化（INT8/FP16），使用 TensorRT 加速

**风险**: 生成结果质量不稳定  
**应对**: 添加 temperature 调节 + top-p 采样，增加随机性控制

---

## 六、防御性设计的完整落地清单

### 6.1 与代码接口的对应关系

| 防御措施 | 代码位置 | 接口名 | 落地周次 | 验收标准 |
|----------|----------|---------|----------|----------|
| Topology Consistency Loss | `tokenizer/vqvae_tokenizer.py` | `compute_topology_consistency_loss()` | Week 2 | 共享顶点偏离 < 1e-6 |
| UV Density Kill Switch | `tokenizer/scripts/batch_preprocess.py` | `validate_uv_density()` | Week 1 | 丢弃率 < 30% |
| 拓扑升维焊接 | `tokenizer/mesh_assembler.py` | `_merge_patches_with_topology()` | Week 2 | 焊接成功率提升 |
| 渐进式 L_topo 权重 | `train_vqvae.py` | `get_lambda_topo(epoch)` | Week 2 | 模型收敛稳定 |
| Codebook EMA | `models/vqvae_gnn.py` | `update_codebook_ema()` | Week 2 | 利用率 > 70% |
| Boundary Conditioning | `models/transformer.py` | `BoundaryEncoder` | Week 3 | 边界洪差 < 1e-3 |
| 级联 Pipeline | `pipeline/inference.py` | `cascade_refine()` | Week 4 | 流形率 > 90% |
| 点云特征化 | `models/pointnet.py` | `encode_point_cloud()` | Week 4 | 传输体积 < 1MB |

### 6.2 未覆盖的风险（需要补充）

| 风险 | 等级 | 影响 | 解决方案 | 责任人 |
|------|------|------|----------|----------|
| Codebook Collapse | 🟠 HIGH | VQ-VAE 重建质量下降 | EMA + Code Restart | 算法团队 |
| GNN Over-smoothing | 🟡 MEDIUM | 面级特征趨于相同 | 限制层数 + GAT | 算法团队 |
| 测试资源不足 | 🟡 MEDIUM | 验证覆盖率不足 | 增加测试样本 | QA 团队 |
| 数据集偏差 | 🟡 MEDIUM | 模型泛化能力下降 | 多数据源融合 | 数据团队 |

---

## 七、关键正比正见（对原始建议的修正）

### 7.1 UV Density 阈值：从 50 降至 20

**原始建议**: "过滤阈值设为 min_patch_density = 50"

**CTO 判定**: ❌ 过于激进

**理由**:
- 若阈值 50，可能丢弃 60%~70% 的数据
- 训练集过小会导致模型泛化不足
- 工业级 UV 的合理分布:
  - Density > 50: 优质数据（占 30%~40%）
  - 20~50: 可接受数据（占 40%~50%）
  - < 20: 碎少数据（占 10%~30%）

**修正方案**:
```python
min_patch_density = 20.0   # 硬门槛：低于此值丢弃
# 分类策略:
#   - 优质（Density > 50）: 增加采样权重
#   - 正常（20~50）: 标准采样
#   - 低质（< 20）: 直接丢弃
```

### 7.2 L_topo 权重：从固定值改为渐进式

**原始建议**: "结入总损失：L_total = L_recon + β L_commitment + λ_topo L_topo"，未给出 λ_topo 具体值

**CTO 判定**: ⚠️ 需要渐进式调度

**理由**:
- 若 epoch 0 就设 λ_topo = 1.0，模型可能不收敛
- 因为 L_topo 会抗衡 L_recon，导致两者都无法优化
- 建议先让模型学会基本重建，再逐步加入拓扑约束

**修正方案**:
```python
def get_lambda_topo(epoch):
    """
    渐进式权重调度
    - epoch 0~10:  0.1  (让模型先学基本重建)
    - epoch 10~30: 0.5  (强化拓扑约束)
    - epoch 30+:   1.0  (全力收敛)
    """
    if epoch < 10:
        return 0.1
    elif epoch < 30:
        return 0.5
    else:
        return 1.0
```

### 7.3 Boundary Conditioning：需要更具体的特征设计

**原始建议**: "在每个 Patch Token 中嵌入显式的边界状态编码"，未给出具体特征

**CTO 判定**: ⚠️ 需要补充具体特征设计

**修正方案**:
已在上文 Week 3 任务 3.3 中详细定义了 5 个具体的边界特征：
- `boundary_vertex_count`
- `seam_edge_count`
- `is_closed_loop`
- `neighbor_patch_ids`
- `shared_boundary_length`

### 7.4 点云采样：从 Poisson Disk 改为 Uniform Grid + FPS

**原始建议**: "在 Blender 前端对高模进行快速的泊松磁盘采样"

**CTO 判定**: ⚠️ Blender Python API 中无原生 Poisson Disk Sampling

**修正方案**:
- **Phase 3 MVP**: 使用 Uniform Grid Sampling（Blender 原生支持 `bpy.ops.mesh.vertices_to_points`）
- **Phase 4 优化**: 引入 Open3D 的 Poisson Disk（需要额外依赖）
- **备选**: 自实现 Farthest Point Sampling (FPS)（轻量，无额外依赖）

---

## 八、资源与依赖

### 8.1 硬件需求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| GPU | RTX 3060 (12GB) | RTX 4080 (16GB) | VQ-VAE + Transformer 联合训练 |
| RAM | 16 GB | 32 GB | 数据加载和缓冲 |
| 存储 | 50 GB | 100 GB | 数据集 + 检查点 + 日志 |
| CPU | 8 核 | 16 核 | 数据预处理 |

### 8.2 软件依赖

```bash
# 核心框架
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# 训练基础设施
pytorch-lightning>=2.0.0
wandb>=0.15.0

# 数据处理
numpy>=1.24.0
trimesh>=4.0.0
xatlas>=0.1.0

# 工具
jupyter
tqdm
matplotlib
```

### 8.3 环境配置

```bash
# WSL2 + CUDA 环境
conda create -n mesh-topo-ai python=3.10
conda activate mesh-topo-ai

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 安装其他依赖
pip install -r requirements.txt
```

---

## 九、验收标准与发布准入

### 9.1 总体验收标准

Phase 3 完成时，项目必须满足以下所有条件才能进入 Phase 4：

| # | 验收项 | 标准 | 验证方法 |
|---|--------|------|----------|
| 1 | VQ-VAE 重建质量 | 重建 L2 < 1e-3 | 验证集上的平均 Chamfer Distance |
| 2 | 拓扑一致性 | 共享顶点偏离 < 1e-6 | `compute_topology_consistency_loss()` |
| 3 | 生成流形率 | > 95% | 检查每个边是否被恰好两个面共享 |
| 4 | 生成速度 | < 60s/mesh | 测试 50 个 mesh 的平均生成时间 |
| 5 | 代码覆盖率 | > 80% | 自动化测试覆盖所有模块 |
| 6 | 文档完整性 | 完整 | 所有模块有 docstring 和 type hint |

### 9.2 发布准入检查清单

**代码**:
- [ ] 所有模块通过单元测试
- [ ] 代码覆盖率 > 80%
- [ ] 所有警告和 TODO 已解决或记录
- [ ] 没有安全漏洞（硬编码凭证、敏感数据泄露）

**模型**:
- [ ] VQ-VAE 检查点已保存（codebook + 权重）
- [ ] Transformer 检查点已保存
- [ ] 生成结果的示例已存档

**文档**:
- [ ] API 文档已更新
- [ ] 训练指南已完善
- [ ] 常见问题解决方案已录入

---

## 十、附录

### 附录 A: 术语表

| 术语 | 定义 |
|------|------|
| UV Island | UV 空间中连通的面片组 |
| Patch | Mesh 中一组连通的面片（通常对应一个 UV island） |
| Seam | UV 展开时需要切开的边界 |
| Token | VQ-VAE 量化后的离散编码 |
| Weld | 将多个相同位置的顶点合并为一个 |
| Deduplicate | 去除重复的顶点 |
| Manifold | 每个边恰好被两个面共享的网格 |
| Codebook | VQ-VAE 中存储的离散向量码本 |
| EMA | 指数移动平均，用于平滑更新 |
| Over-smoothing | GNN 层数过多导致节点特征趋于相同 |

### 附录 B: 参考文档

- `docs/PROJECT_GUIDE.md` — 项目总体指南
- `docs/RISK_ASSESSMENT.md` — 风险盘点与防御性方案
- `docs/research-report-phase1.md` — Phase 1 技术调研报告
- `docs/data-pipeline-report-phase2.md` — Phase 2 数据管道报告

### 附录 C: 变更日志

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|----------|------|
| v1.0 | 2026-05-21 | 初始版本，包含完整的 4 周 Sprint 计划 | CTO |

---

*本文档由 Hermes CTO 维护，随着 Phase 3 进展实时更新。任何偏离本计划的决策必须经过 CTO 审批并记录在本文档的变更日志中。*
