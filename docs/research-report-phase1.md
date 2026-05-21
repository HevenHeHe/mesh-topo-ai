# AI 智能拓扑与原生 UV 资产管线 — 技术调研报告 (Phase 1)

> 调研时间: 2026-05-21
> 调研范围: MeshGPT / PolyGen / Triangle Strip Tokenization / 开源数据集拓扑质量

---

## 1. MeshGPT 核心方法论

**论文**: *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers* (SIGGRAPH 2024)
**链接**: https://arxiv.org/abs/2311.15475

### Tokenizer 设计
MeshGPT 将三角形网格视为**离散 token 序列**，核心是一个**基于图神经网络（GNN）的 VQ-VAE tokenizer**：

- **编码器**: 在网格的**面片邻接图（face-adjacency graph）**上运行 GNN，每个节点代表一个三角形面片，边代表共享边。通过消息传递聚合局部几何信息，将每个面片压缩为 latent embedding。
- **量化（VQ）**: 使用向量量化码本将连续的 face embedding 映射为离散 code。整个网格被编码为**一个短序列的离散 token**（通常几十到几百个）。
- **解码器**: 从量化后的 face token 序列重建网格。预测的是**面片坐标（face corners）**，即每个三角形的 3 个顶点位置，而非直接预测顶点索引。
- **序列化**: 编码器输出的 face token 按**面片邻接顺序**排列，形成"面片行走"序列。Decoder-only GPT 在这些 token 上执行自回归预测。

**关键洞察**: Tokenizer 学习的是"网格局部几何模式"的离散字典，GPT 学习的是这些模式之间的组合规律。它操作的是**面片级别的 token**，而非顶点或边。

---

## 2. PolyGen 核心方法论

**论文**: *PolyGen: An Autoregressive Generative Model of 3D Meshes* (ICML 2020)
**链接**: https://arxiv.org/abs/2002.10880

### 序列化策略
PolyGen 采用**层次化自回归**：

- **顶点序列**: 首先自回归生成顶点坐标。每个顶点表示为量化后的三维坐标 $(x,y,z)$。
- **面片序列**: 顶点生成完毕后，再自回归生成面片。每个面片表示为**指向顶点序列的索引三元组** $(i, j, k)$。
- **顺序启发式**: 顶点按**从下到上（z轴）**生成；面片按确定性排序排列。
- **注意力掩码**: 顶点预测只看到前面的顶点，面片预测只看到前面的面片及全部顶点。

**关键洞察**: 拓扑（面片索引）和几何（顶点坐标）分离成两个自回归阶段。序列长度极长，计算成本高，复杂网格扩展性有限。

---

## 3. "Strips as Tokens" 可行性评估

### 贪婪三角带剥离（Greedy Triangle Strip Peeling）

| 维度 | 评估 |
|------|------|
| **序列化能力** | ✅ 强。三角带天然产生一维序列。 |
| **可逆性/解码** | ⚠️ 中等。一个网格存在多种 strip 分解方式，序列不唯一，增加学习难度。 |
| **拓扑表达力** | ⚠️ 有限。适合流形网格，对非流形边、多边界、洞的表达能力差。 |
| **UV/属性保持** | ❌ **差**。三角带剥离以几何邻接为驱动，**不感知 UV seams**。在 UV 边界处顶点需要拆分，破坏 strip 连续性。 |
| **序列长度** | ✅ 较好。相比逐面片序列，token 数可压缩到 1/3 ~ 1/2。 |

### 核心结论

**不建议直接将标准贪婪三角带剥离作为核心 Tokenizer**，原因：
1. **非唯一性**: 同一网格的多种 strip 分解造成数据分布模糊。
2. **UV/材质盲**: 项目目标是"原生 UV 分区"，而三角带算法完全不感知 UV 拓扑。
3. **复杂拓扑脆弱**: 工业模型常有非流形结构。

**若坚持"Strips"思路**，应设计**语义感知的三角带**——在 UV island 边界和材质边界处强制切断，使每个 strip 落在单一 UV 区域内。这实际上更接近 **MeshGPT 的 face-token + UV 约束** 的混合方案。

---

## 4. 开源数据集评估

### Objaverse / Objaverse-XL
- **规模**: >800k 模型
- **拓扑质量**: ❌ **极差**。UGC 拓扑极不一致，三角/四边/多边形混合、非流形、内部面普遍存在。
- **UV**: ❌ 绝大多数无可用 UV，或严重重叠/拉伸。
- **适用性**: 仅适合点云/NeRF。用于本项目需重度清洗，成本极高。

### ShapeNet
- **规模**: ~51k 模型
- **拓扑质量**: ⚠️ **中等偏低**。mostly manifold，但存在 T-junctions、孤立顶点、退化面。
- **UV**: ❌ 大多数无原生 UV，或 UV 为自动展开（atlasing），不适合工业管线。
- **适用性**: 适合 occupancy network，不适合高质量 mesh/UV 生成。

### ABC Dataset (A Big CAD Model Dataset)
- **规模**: ~1M CAD 模型
- **拓扑质量**: ✅ **高**。B-rep 通常是精确的流形曲面。
- **UV/布线**: ✅ 有良好的结构线（feature curves），但原生表示是 NURBS/B-rep，需高质量三角化。

### 推荐策略
**现有开源数据集均不完美**。建议：
- **Phase 1-2**: 从 **ABC Dataset** 或 **Fusion 360 Gallery** 提取 B-rep，使用严格四边/三角重拓扑 + 自动 UV 展开构建**合成训练数据**。
- **Phase 3**: 用 **Objaverse** 进行对比预训练，但需依赖上游几何理解模型过滤低质量样本。

---

## 5. 综合评估与 Phase 2 建议

### 技术难度: 8/10
MeshGPT 证明了高质量 mesh 生成可行，但将"工业级布线"和"UV 分区"作为硬约束加入自回归框架，需要重新设计：
1. **联合 Tokenizer**: 同时编码几何、拓扑、UV seams 的 token。
2. **结构化解码器**: 保证生成的面片在 UV 空间不重叠，且边界精确对齐。

### 最大瓶颈
1. **数据瓶颈 > 算法瓶颈**。没有大规模、高质量、带原生 UV 和一致拓扑的公开数据集。
2. **序列长度与复杂性**。UV seams 引入大量拓扑约束，自回归模型在长序列上保持全局 UV 一致性极其困难。
3. **非唯一性表示**。显式三角带/拓扑序列化可能存在多对一映射，增加模型困惑度。

### Phase 2 切入点建议

**方案 A（推荐）: UV-Guided Face Cluster as Token**
- 先用 **UV atlas 分区**将 mesh 切分为若干连通的 face cluster（每个 cluster 是一个 UV island）。
- 每个 cluster 内部使用**基于 GNN 的 VQ-VAE tokenizer**（类似 MeshGPT），但编码时加入 UV 边界信息。
- GPT 自回归生成的是 **cluster 序列**，而非单个面片。大幅降低序列长度，且天然保证 UV island 级别的结构。

**方案 B: Graph Diffusion 替代纯自回归**
- 若自回归在长距离 UV 接缝约束下表现差，可转向**离散图扩散模型**（如 DiT for Mesh）在 face-adjacency graph 上操作，更容易维持全局拓扑一致性。

**方案 C: 级联 Pipeline（最务实）**
- **Step 1**: 自回归生成"粗拓扑骨架"（edge loops + face patches）。
- **Step 2**: 对每个 patch 内部，使用轻量级优化器（可微分渲染 / Differentiable UV）精化布线和 UV 展开。

---

## 核心参考链接

| 论文 | 链接 |
|------|------|
| MeshGPT | https://arxiv.org/abs/2313.15475 |
| PolyGen | https://arxiv.org/abs/2002.10880 |
| ABC Dataset | https://arxiv.org/abs/1905.07553 |
| Objaverse | https://arxiv.org/abs/2212.08051 |

---

*报告由 Hermes CTO 调研代理产出*
