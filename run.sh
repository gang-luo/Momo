# teacher model

python scripts/train_fda_teacher.py \
  --config configs/train_fda_teacher_novomolgen.yaml \
  --base_model_name_or_path infer/pretrained_mol/NovoMolGen_32M_SMILES_AtomWise \
  --fda_data_path data/ZINC/fda.json \
  --output_adapter_dir infer/teachers/fda_lora



python scripts/train_target_teacher.py \
  --config configs/train_target_teacher_novomolgen.yaml \
  --base_model_name_or_path infer/pretrained_mol/NovoMolGen_32M_SMILES_AtomWise \
  --target_data_path data/bindingDB/result/EGFR_lora_smiles_1000nM.json \
  --target_name EGFR \
  --output_adapter_dir infer/teachers/EGFR_lora