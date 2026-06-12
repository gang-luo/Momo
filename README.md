# Momo MolOPD

MolOPD 是一个面向小分子 SMILES 自回归生成器的 **post-training / on-policy distillation** 框架。当前实现以 Hugging Face 上的 NovoMolGen SMILES Transformer 为 base molecular generator，参考 `ProteinOPD-main/` 的 teacher construction + OPD training 思路，但保持 molecule-specific 的数据、采样、评价与 LoRA teacher 边界。

## 方法边界

- **Base**：原始 NovoMolGen 直接采样，不做 post-training。
- **FDA-SFT / FDA-LoRA**：在 FDA/approved-drug 分子集合上对 base model 做 LoRA/SFT，得到 developability prior teacher；它不是最终 student 的唯一目标。
- **Target-LoRA**：在单靶标活性分子集合上对 base model 做 LoRA/SFT，得到 `target1`、`target2` 等方向性 teacher。
- **RL**：只用 sequence-level reward 优化的模型；第一版不实现，仅保留 evaluator/reward 接口。
- **RL+KL**：reward 优化时加入 KL-to-base 的 baseline；第一版不实现，仅保留配置边界。
- **MolOPD**：student 从 base generator 初始化，在 student on-policy samples 上同时查询 `π_base`、`π_FDA`、`π_Target_i` 的 token logits/log-probs，用 token-level multi-teacher OPD 更新 student：FDA 是 developability prior，target teachers 是靶向设计方向，base retention model 用于保持探索能力。

第一版默认 **不启用 ADMET/reward/evaluator 参与 loss**，只实现 token-level retention + FDA prior + one/multiple target priors。

## 配置与训练入口约定

所有训练脚本都采用 **YAML-first** 约定：命令行只传入 `--config`，不再额外传入数据路径、模型路径、LoRA 输出路径或优化器参数。这样可以保证每一次实验的 data/model/lora/loss/optimizer/scheduler/trainer/logging/runtime 都被完整记录在同一个 YAML 中。

每个训练 YAML 均按以下职责分块：

- `data`: 数据路径、SMILES 处理、batch size、num workers、max length。
- `model`: base/student/tokenizer/adapter 输出等模型与权重路径。
- `lora`: teacher 或 student LoRA 结构参数。
- `teachers`: MolOPD 所需 FDA 与 target adapters。
- `loss`: 明确损失权重与 KL 方向；teacher SFT 使用 shifted-token NLL。
- `optimizer`: optimizer、lr、weight decay、betas、grad clip。
- `scheduler`: scheduler 类型与参数。
- `trainer`: Lightning Trainer 设置，如 precision、accelerator、devices、DDP。
- `checkpoint`: checkpoint monitor/save 策略。
- `logging`: wandb/CSV logger 设置。
- `runtime`: output directory、seed、resume、run_test。

## 核心模块

- `molopd.policies`: `BasePolicyWrapper`、`StudentPolicyWrapper`、`TeacherPolicyWrapper`，统一 sample/logits/token log-prob/sequence log-prob 接口。
- `molopd.data`: JSON/JSONL/CSV SMILES 读取，自动识别 `smiles`、`SMILES`、`canonical_smiles`、`generated_smiles` 字段，RDKit canonicalization、invalid 过滤、去重与 causal-LM labels。
- `molopd.training.teacher_lightning_module`: FDA/target teacher LoRA SFT 的 LightningModule，使用显式 shifted-token NLL loss。
- `molopd.sampling`: `OnPolicySampler` 从当前 student 采样、valid/canonicalize、去重并 tokenization。
- `molopd.opd`: masked token-level KL 与 `MolOPDLightningModule`。
- `molopd.evaluators`: RDKit logging evaluator 与 ADMET/FDA-likeness/target evaluator 占位接口，默认关闭。
- `molopd.training.replay_buffer`: 每轮样本 CSV 与 replay buffer 日志。

## 1. FDA teacher construction

先编辑 `configs/train_fda_teacher_novomolgen.yaml` 中的 `data.train_path`、`data.val_path`、`model.base_model_name_or_path`、`model.output_adapter_dir` 等字段，然后运行：

```bash
python scripts/train_fda_teacher.py --config configs/train_fda_teacher_novomolgen.yaml
```

训练逻辑：`MolTeacherSFTLightningModule` 对 FDA SMILES 做 causal LM LoRA/SFT，loss 是 shifted-token negative log-likelihood。输出：

- `model.output_adapter_dir/adapter_model.*`
- `model.output_adapter_dir/adapter_config.json`
- tokenizer 文件
- `runtime.output_dir/checkpoints/`
- wandb 或 CSV logs

## 2. Target teacher construction

先编辑 `configs/train_target_teacher_novomolgen.yaml`：

- `teacher.target_name`: 例如 `target1`、`target2`。
- `data.train_path`: 该靶标活性分子集合。
- `model.output_adapter_dir`: 对应 target LoRA 输出目录。

然后运行：

```bash
python scripts/train_target_teacher.py --config configs/train_target_teacher_novomolgen.yaml
```

对多个靶标重复复制/修改 YAML，并在 MolOPD 配置的 `teachers.target_adapters` 中列出多个 adapter。

## 3. MolOPD token-level multi-teacher training

Debug（小步数验证 pipeline）：

```bash
python scripts/run_molopd.py --config configs/molopd_novomolgen_debug.yaml
```

Main：

```bash
python scripts/run_molopd.py --config configs/molopd_novomolgen_main.yaml
```

训练每一步会：

1. 从当前 student on-policy 采样 SMILES。
2. RDKit canonicalize 并过滤 invalid 样本。
3. 对 valid canonical SMILES tokenization。
4. 查询 student/base/FDA/target teacher logits。
5. 计算 masked token-level OPD loss。
6. 只更新 student。
7. 记录 wandb/CSV metrics 与样本 CSV。

样本日志默认保存到：

```text
runtime.output_dir/logging.run_name/samples_iter_0000.csv
runtime.output_dir/logging.run_name/replay_buffer.csv
```

## 4. Unified generation

Base generation：

```bash
python scripts/sample_model.py \
  --model_name_or_path chandar-lab/NovoMolGen_32M_SMILES_AtomWise \
  --label base \
  --num_samples 1000 \
  --output outputs/eval/base_samples.csv
```

FDA-LoRA generation：

```bash
python scripts/sample_model.py \
  --model_name_or_path chandar-lab/NovoMolGen_32M_SMILES_AtomWise \
  --adapter_path pretrained_mol/teachers/fda_lora \
  --label fda_lora \
  --num_samples 1000 \
  --output outputs/eval/fda_lora_samples.csv
```

Target-LoRA generation：

```bash
python scripts/sample_model.py \
  --model_name_or_path chandar-lab/NovoMolGen_32M_SMILES_AtomWise \
  --adapter_path pretrained_mol/teachers/target1_lora \
  --label target1_lora \
  --num_samples 1000 \
  --output outputs/eval/target1_lora_samples.csv
```

MolOPD student generation：

```bash
python scripts/sample_model.py \
  --model_name_or_path chandar-lab/NovoMolGen_32M_SMILES_AtomWise \
  --adapter_path outputs/molopd_main/novomolgen_molopd_main/final_student \
  --label molopd \
  --num_samples 1000 \
  --output outputs/eval/molopd_samples.csv
```

## 5. Four-block evaluation notebook

当前第一版至少支持生成以下四类样本用于 notebook 评估：

1. Base generation
2. FDA-LoRA generation
3. Target-LoRA generation
4. MolOPD generation

评估沿用已有四模块 notebook：

1. Basic physchem
2. ADMET-AI
3. Scaffold
4. Fingerprint/FDA-neighborhood

推荐将 `scripts/sample_model.py` 输出的 CSV 统一放到 `outputs/eval/`，在 notebook 中按 `label/model_name` 分组比较。

## 测试

```bash
pytest tests/test_smiles_dataset.py tests/test_kl_loss.py
pytest tests/test_on_policy_sampler.py tests/test_teacher_loading.py
pytest tests/test_run_debug.py
```
