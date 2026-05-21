# Momo MolOPD

基于 ProteinOPD 思路，将 OPD（On-Policy Distillation）迁移到小分子生成（NovoMolGen）场景。

## 核心功能
- 基于 `chandar-lab/NovoMolGen_32M_SMILES_AtomWise` 的 student。
- 三个 PEFT/LoRA teacher：FDA、Target1、Target2。
- 单目标 OPD（FDA teacher）与多目标 OPD（FDA+Target1+Target2）训练。
- PyTorch Lightning 单卡/多卡、W&B、推理 JSON 导出。

## 快速开始
```bash
pip install -e .
python scripts/train.py --config configs/train_fda_teacher.yaml
python scripts/train.py --config configs/train_opd_multi.yaml
```

## 数据组织
- `pretrained_mol/`：模型与数据集下载目录
- `momo_molopd/data/`：训练/验证/测试 jsonl
