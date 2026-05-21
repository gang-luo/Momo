# Momo MolOPD 设计说明

## 1. ProteinOPD 思路迁移
- Teacher 构建：先用目标数据子集（FDA/target1/target2）做轻量 SFT（本项目用 LoRA）。
- OPD 核心：student 在自身 rollout 轨迹上，对齐 teacher 的 token-level 分布（JSD/KL）。
- 多目标：多个 teacher 使用 PoE（logits 加权几何平均）形成共识分布，再进行蒸馏。

## 2. NovoMolGen 损失理解
- 基础训练目标是自回归 Causal LM NLL（交叉熵），即 `labels=input_ids` 的 next-token prediction。
- 本项目 teacher 微调保持该目标不变，仅更新 LoRA 参数。
- OPD 阶段将监督信号改为 student logits 与 teacher/PoE logits 的 token-level JSD。

## 3. 关键评估指标（除 loss）
- Validity：RDKit 可解析比例。
- Uniqueness：去重后分子占比。
- Novelty：相对训练集新颖性。
- Diversity：分子间平均 Tanimoto 距离。
- Property 指标：QED/SA/logP + 两靶标打分器。
- Pareto/Hypervolume：多目标整体权衡能力。

## 4. 训练流程
1. 训练 3 个 teacher LoRA（FDA、target1、target2）。
2. 单目标 OPD：仅 FDA teacher。
3. 多目标 OPD：FDA+target1+target2（可调权重）。
4. 用 infer notebook 导出分子 JSON，供后续物化性质计算。
