# 🚨 核心技术风险盘点与防御性方案

> 盘点时间: 2026-05-21  
> 状态: Phase 2 → Phase 3 过渡关口  
> 风险等级: 🔴 CRITICAL / 🟠 HIGH / 🟡 MEDIUM  
> 依据: 对 Phase 2 代码实现的深度工程审查

---

## 风险 1：GNN Patch 局部重建时的"拓扑失忆" — 微裂缝危机

**风险等级**: 🔴 CRITICAL  
**风险频率**: Phase 3 训练完成后必然触发

### 风险描述

VQ-VAE Decoder 的输出是每个面独立的 `F_p × 3 × 3` corners，意味着：
- 同一个顶点被多个共享它的面**各自独立预测**
- VQ-VAE 的压缩是有损的，每个面的 latent 被量化后信息损失
- 共享顶点在不同面中的重建坐标可能偏离 `1e-5` ~ `1e-6`

**致命链路**:
```
Decoder 独立预测 face corners
    → 共享顶点坐标不一致
    → Mesh Assembler deduplicate 阈值 1e-6 无法合并
    → 产生物理裂缝（Cracks）
    → _remove_degenerate_faces 清理出大量孔洞
    → 模型报废
```

### 防御方案：Topology Consistency Loss

**核心思想**: 在训练目标中强行惩罚"原始 mesh 中是同一个顶点，重建后必须仍是同一个坐标"。

**形式化定义**:

```
L_topo = λ_topo · Σ_{adjacent faces (i,j)} Σ_{shared vertices v} 
         || corner_i[v] - corner_j[v] ||²
```

- `corner_i[v]`：面 i 中共享顶点 v 的重建坐标
- `corner_j[v]`：面 j 中同一顶点 v 的重建坐标
- `λ_topo`: 权重（建议 0.5 ~ 1.0）

**实现策略**:
1. 在 `vqvae_tokenizer.py` 中新增 `compute_topology_consistency_loss()` 方法
2. 训练时通过 face-adjacency 矩阵快速查找共享顶点
3. 总损失: `L = L_recon + L_commitment + L_topo`

**异常处理**:
- 若 patch 过小（< 3 faces），共享顶点约束可能不充分 → 可设置最小 patch size 门槛
- 非流形 patch 可能存在套瓦面次序 → 需要 vertex-winding 校验

### 验证标准

重建后的 mesh 满足:
- 共享顶点偏离中位数 < `1e-6`
- 无物理裂缝（通过 mesh 水密性检测）
- 退化面比例 < 0.1%

---

## 风险 2：Fusion 360 数据集的"无 UV 危机" — UV 碎屑爆炸

**风险等级**: 🔴 CRITICAL  
**风险频率**: 下载数据集后第一次运行 batch 预处理就会触发

### 风险描述

CAD 数据集的特点：
- 原生模型**无工业级 UV**
- 自动 UV 展开工具（xatlas、Blender Smart UV）面对复杂机械结构时，会生成**极其细碎的 UV islands**
- 一个简单的泛型零件可能被切出上百个微小 patch

**致命链路**:
```
CAD 模型自动展开 UV
    → 产生成千个细小 UV islands
    → Patch 数量爆炸
    → Transformer 上下文窗口不足（通常 1024~4096）
    → 自回归生成不可行
    → 数据集报废
```

### 防御方案：UV Patch 密度熔断指标

**核心思想**: 在数据管道中设置硬性过滤门槛，只保留"大块、规整"的 UV 分区模型。

**熔断指标**:

```
Density = 总面数 / UV Patches 数量

策略:
- Density > 50   → ✅ 保留（大块 UV，工业级分区）
- 20 < Density < 50 → ⚠️ 可保留（边界情况）
- Density < 20   → ❌ 丢弃（UV 碎少了，自动展开质量过差）
```

**实现策略**:
1. 在 `tokenizer/scripts/batch_preprocess.py` 中新增过滤步骤
2. 记录被丢弃模型的统计信息（用于后续评估数据质量）
3. 设置可配置阈值（`min_patch_density`）

**备选方案**：
- 若 Fusion 360 碎片率过高，转向 **ABC Dataset**，使用严格的四边形三角化 + RizomUV 级别的自动展开
- 或者使用**合成数据**：从简单几何体逐步增加复杂度，确保 UV 分区可控

### 验证标准

统计训练集的 patch 分布：
- 平均 patches/mesh: < 20
- 最大 patches/mesh: < 50
- 丢弃率: < 30% 为可接受

---

## 风险 3：自回归 Transformer 的"维数灾难与死锁"

**风险等级**: 🟠 HIGH  
**风险频率**: Phase 3 训练完成后，在复杂序列上必然遇到

### 风险描述

自回归生成的本质局限：
- Transformer 逐个预测 patch token，每一步都只能看到已生成的序列
- 当预测到第 10 个 patch 时，它**无法前瞻**第 1 个 patch 的几何位置
- 误差逐步累积，最终可能导致不同 patch 的边界**无法闭合**

**致命链路**:
```
Transformer 逐个生成 Patch
    → 第 10 个 Patch 的位置与第 1 个不匹配
    → 边界不能闭合
    → 几何扭曲 / 形状崩溃
    → 整体拓扑不可用
```

### 防御方案：多层次约束与条件增强

**方案 A: 层次化生成（Hierarchical Generation）**

不是一步到位预测所有 patch tokens，而是分层次：
1. **粗骨架层**: 预测粗略的拓扑骨架（粗网格 / edge loops）
2. **中层**: 在骨架约束下预测 patch 的大致位置和尺寸
3. **精细层**: 在位置约束下预测每个 patch 的具体几何

**方案 B: 边界条件化（Boundary Conditioning）**

在每个 patch token 中嵌入**边界编码**：
- 该 patch 的边界顶点数量
- 边界顶点的悬挂状态（是否已与前面的 patch 连接）
- 这让 Transformer 在生成新 patch 时"知道"哪些边界必须与现有结构对齐

**方案 C: 级联 Pipeline（最务实）**

如调研报告中所讨论：
1. Transformer 只生成"粗骨架"（大致的面片排列 + UV island 布局）
2. 对每个 patch 内部，使用**可微分优化**（Differentiable UV 展开 + Differentiable Mesh Processing）精化
3. 这样 Transformer 的"误差"被可微分优化器缓解

### 验证标准

- 自回归生成的网格在 patch 边界处的洪差 < `1e-3`
- 生成结果的 manifold 率 > 95%
- 对比实验：无条件生成 vs 有条件生成的几何质量

---

## 风险 4：Blender 插件的"IO 死锁"

**风险等级**: 🟡 MEDIUM  
**风险频率**: 大型高模（> 100MB OBJ）时必然触发

### 风险描述

当前实现的问题：
- Blender 插件**同步导出** 整个高模 OBJ
- 通过 HTTP POST **发送整个文件**
- 这对于几百 MB 的高模导致 Blender UI **阻塞假死**

### 防御方案：点云特征替代高模导出

**核心思想**: 不发送完整高模网格，只发送经过 PointNet++ 编码后的**轻量级点云特征**。

**新架构**:
```
Blender 插件
    ├── 异步导出高模为点云（采样到固定数量的点）
    ├── 将点云编码为 256-D 或 512-D 特征向量
    └── 通过 HTTP 发送轻量 JSON （< 1MB）

后端服务
    ├── 接收点云特征
    ├── Transformer 条件生成 patch 序列
    └── 返回重建的低模网格
```

**技术细节**:
- 点云采样: 使用 Poisson disk sampling 或 uniform grid sampling
- 特征编码: 在 Blender 中调用预训练的 PointNet++ 小模型（或使用轻量级后端）
- 通信: JSON payload 中仅包含 feature vector + 元数据（比例、转换矩阵）

**预期效果**:
- HTTP 请求体积从 > 100MB 降至 < 1MB
- Blender UI 无阻塞感
- 实时性显著提升

### 验证标准

- 点云采样保留几何特征率 > 95%（与完整高模对比）
- 传输延迟 < 100ms
- Blender UI 无卡顿（通过 frame time 监测）

---

## 总结：防御性路线图

```
Phase 2 后半 数据验证
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

## 附录：代码层面的防御性接口

以下接口已在代码库中预留：

| 防御措施 | 文件 | 接口 |
|----------|------|------|
| Topology Consistency Loss | `tokenizer/vqvae_tokenizer.py` | `compute_topology_consistency_loss()` |
| UV Density 熔断 | `tokenizer/scripts/batch_preprocess.py` | `filter_by_patch_density()` |
| 拓扑升维焊接 | `tokenizer/mesh_assembler.py` | `merge_patches_with_topology()` |
| 点云特征导出 | `blender-addon/__init__.py` | `export_point_cloud_features()` (待实现) |

---

*本文档与代码实现同步更新，任何 Phase 3 进展中的新风险应实时追加。*
