#!/bin/bash
#SBATCH -J prompt_mol_protein
#SBATCH -o pred_protein_%j.log
#SBATCH -e pred_protein_%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

# Submit from the project root, or export PROMPT_MOL_ROOT before sbatch.
# Adjust partition and resources to match your cluster configuration.
set -euo pipefail
PROMPT_MOL_ROOT="${PROMPT_MOL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "$PROMPT_MOL_ROOT/run" "$PROMPT_MOL_ROOT/results/protein"
cd "$PROMPT_MOL_ROOT/run"

python "$PROMPT_MOL_ROOT/pred_protein_mol.py" \
  --lr=1e-5 \
  --train_raw_path "$PROMPT_MOL_ROOT/data/SA/chembl_8k_mw.csv" \
  --test_raw_path "$PROMPT_MOL_ROOT/data/protein/test_mol_input_sa.csv" \
  --model_path "$PROMPT_MOL_ROOT/fine_models/protein/gen_8k_mw/pytorch_model.pt" \
  --vocab_path "$PROMPT_MOL_ROOT/prompt_reg_voc.csv" \
  --k 4 \
  --save_results "$PROMPT_MOL_ROOT/results/protein/gen_8k_mwtrain_k41.csv"
