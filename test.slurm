#!/bin/bash
#SBATCH -J prompt_mol_property
#SBATCH -o pred_cyp_%j.log
#SBATCH -e pred_cyp_%j.err
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Submit from the project root, or export PROMPT_MOL_ROOT before sbatch.
# Adjust partition and resources to match your cluster configuration.
set -euo pipefail
PROMPT_MOL_ROOT="${PROMPT_MOL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "$PROMPT_MOL_ROOT/run"
cd "$PROMPT_MOL_ROOT/run"

python "$PROMPT_MOL_ROOT/pred_prompt_property.py" \
  --lr=1e-5 \
  --train_raw_path "$PROMPT_MOL_ROOT/data/cyp/train.csv" \
  --test_raw_path "$PROMPT_MOL_ROOT/data/cyp/test.csv" \
  --model_path "$PROMPT_MOL_ROOT/fine_models/cyp/pytorch_model.pt" \
  --vocab_path "$PROMPT_MOL_ROOT/prompt_reg_voc.csv" \
  --k 3 --property_name cyp_2c9 \
  --cat_model_path "$PROMPT_MOL_ROOT/cat_models/cyp"
