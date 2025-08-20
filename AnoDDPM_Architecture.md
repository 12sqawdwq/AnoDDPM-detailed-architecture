# AnoDDPM 模型架构图（可编辑 Mermaid）

> 目标：帮助快速理解 AnoDDPM（Anomaly Detection with Denoising Diffusion Probabilistic Models Using Simplex）的核心组件与训练/推理流程。可在 VS Code 中直接预览并导出为 SVG/PNG（Markdown 预览支持 Mermaid）。

---

## 1) 总览（训练阶段）

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  %% 训练：在正常样本上拟合扩散逆过程（预测噪声）
  I[训练输入: 正常样本 x₀] --> Q[前向扩散 q(x_t|x_{t-1})<br/>(β 调度, T 步)]
  Q -->|t = 1..T| XT[x_t]
  XT --> U[UNet ε_θ(x_t, t)<br/>(预测噪声 ε̂)]
  U --> L[损失: L_simple = ||ε - ε̂||]
  L --> M[得到训练好的去噪模型 ε_θ*]
```

简述：在仅包含“正常”数据的训练集上，使用 DDPM 噪声预测目标（noise prediction objective）训练 UNet 去噪器。

---

## 2) 推理阶段（异常检测）

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  %% 推理：对测试图像在多个时间步上评估残差并聚合
  X[测试图像 x] --> N1[添加噪声: 采样 t ∈ T_s<br/>(少量代表性噪声步)]
  N1 -->|并行| D1[ε_θ*(x_t, t)<br/>(得到 ε̂ 或 x̂₀)]
  D1 --> R1[残差/重建误差 r_t<br/>例如: ||x - x̂₀,t|| 或 ||ε - ε̂||]
  R1 --> Agg[Simplex 组合与归一化<br/>权重 w_t, ∑w_t=1, w_t≥0]
  Agg --> Heat[像素级异常热力图 A]
  Agg --> Score[整体异常分数 S]
```

要点：在多个噪声步 t 上得到的像素级残差 r_t 通过“Simplex 组合”（可理解为在单纯形约束下的加权聚合）形成稳定、可分离的异常热力图与全局分数。

---

## 3) 组件分解（UNet 去噪器）

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart TB
  subgraph UNet[UNet ε_θ 架构(示意)]
    TE[时间步嵌入 t-embed]
    DS[下采样阶段: ResBlocks + (可选)Attention]
    BN[瓶颈: ResBlocks + Self-Attention]
    US[上采样阶段: ResBlocks + (可选)Attention]
  end
  IN[x_t, t] --> TE --> DS --> BN --> US --> OUT[输出: 预测噪声 ε̂]
```

说明：以上为通用 DDPM-UNet 结构示意，具体层数、通道数、注意力位置可据实现细节调整。

---

## 4) 可调参数与开关（放在图中的位置）
- T（总扩散步数）：位于“前向扩散”与训练流程。
- T_s（推理所选噪声步集）：位于推理图的“添加噪声”节点。
- 残差定义 r_t：位于“残差/重建误差”节点（x vs x̂₀ 或 ε vs ε̂）。
- Simplex 组合权重 w_t：位于“Simplex 组合与归一化”节点（单纯形约束）。
- 归一化/后处理：可在 Agg 前后增加 min-max、平滑或 CRF 等（按需要）。

---

## 5) 备注与假设
- 假设使用标准 DDPM 目标（预测噪声）与 UNet 骨干；注意力层可选。
- 异常图通常来自多尺度（多 t）残差的凸组合，以提升鲁棒性与可分离性。
- 如需严格复现实验，请以论文与原实现的具体设定为准（β 调度、通道数、t 选择策略、聚合细节等）。

---

## 6) 使用与导出
- 预览：在 VS Code 中打开本文件，使用“Markdown 预览”查看 Mermaid 图。
- 导出：通过预览或安装 Mermaid 相关扩展导出为 SVG/PNG 以用于论文插图。
- 修改：直接编辑代码块中的节点/文案，即可快速定制结构示意。

