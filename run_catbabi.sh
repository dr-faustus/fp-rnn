echo CONDA_PREFIX=$CONDA_PREFIX
echo DATA=$DATA
echo LOGS=$LOGS
echo RESULTS=$RESULTS

MODEL_NAME=${1:-FPMamba1}
MIXER_TYPE=${2:-householderproduct}
MIXER_RANK=${3:-4}
N_LAYERS=${4:-4}
LEARNING_RATE=${5:-5e-4}
TRAIN_BATCH_SIZE=${6:-256}
MAX_ITER=${7:-100}

echo "Argument 1 (model_name): $MODEL_NAME"
echo "Argument 2 (mixer_type): $MIXER_TYPE"
echo "Argument 3 (mixer_rank): $MIXER_RANK"
echo "Argument 4 (n_layers): $N_LAYERS"
echo "Argument 5 (lr): $LEARNING_RATE"
echo "Argument 6 (train_batch_size): $TRAIN_BATCH_SIZE"
echo "Argument 7 (max_iter): $MAX_ITER"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bedouin

python main.py \
  --model_name "$MODEL_NAME" \
  --mixer_type "$MIXER_TYPE" \
  --mixer_rank "$MIXER_RANK" \
  --n_layers "$N_LAYERS" \
  --lr "$LEARNING_RATE" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --max_iter "$MAX_ITER" \
  --mixer_h_dep \
  --use_short_conv \
  --use_wandb
