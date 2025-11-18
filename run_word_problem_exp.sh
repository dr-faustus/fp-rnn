echo CONDA_PREFIX=$CONDA_PREFIX
echo DATA=$DATA
echo LOGS=$LOGS
echo RESULTS=$RESULTS

# Print the values of the arguments
echo "Argument 1 (group): $1"
echo "Argument 2 (k): $2"
echo "Argument 3 (model): $3"
echo "Argument 4 (mixer_rank): $4"
echo "Argument 5 (d_mixer): $5"
echo "Argument 6 (k_eval_min): $6"
echo "Argument 7 (k_eval_max): $7"
echo "Argument 8 (layers): $8"
echo "Argument 9 (epochs): $9"
echo "Argument 10 (seed): ${10}"
echo "Argument 11 (max_iter): ${11}"
echo "Argument 12 (lr): ${12}"
echo "Argument 13 (weight_decay): ${13}"
echo "Argument 14 (batch_size): ${14}"
echo "Argument 15 (mixer_type): ${15}"

cd word_problem_exp
python main.py --group $1 --k $2 --model $3 --mixer_rank $4 --d_mixer $5 --k_eval_min $6 --k_eval_max $7 --n_layers $8 --epochs $9 --seed ${10} --max_iter ${11} --lr ${12} --weight_decay ${13} --batch_size ${14} --mixer_type ${15} --use_wandb --gradient_clip 1.0 --layer_norm_eps 1e-6 --mixer_h_dep --n_backwards 1 --k_eval_step 5
