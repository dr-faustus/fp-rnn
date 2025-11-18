echo CONDA_PREFIX=$CONDA_PREFIX
echo DATA=$DATA
echo LOGS=$LOGS
echo RESULTS=$RESULTS

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bedouin

# Print the values of the arguments
echo "Argument 1 (config_file): $1"
echo "Argument 2 (seed): $2"


cd arithmetic_exp
python main.py --config=$1 --wandb_dir '/fast/smovahedi/logs/wandb' --use_wandb --seed $2