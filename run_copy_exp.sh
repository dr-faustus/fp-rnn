echo CONDA_PREFIX=$CONDA_PREFIX
echo DATA=$DATA
echo LOGS=$LOGS
echo RESULTS=$RESULTS

# Print the values of the arguments
echo "Argument 1 (model): $1"
echo "Argument 2 (layers): $2"
echo "Argument 3 (mixer_rank): $3"
echo "Argument 4 (lr): $4"
echo "Argument 5 (epochs): $5"
echo "Argument 6 (train_batch_size): $6"
echo "Argument 7 (eval_batch_size): $7"
echo "Argument 8 (train_task): $8"
echo "Argument 9 (eval_task): $9"
echo "Argument 10 (vocab_size): ${10}"
echo "Argument 11 (min_train_len): ${11}"
echo "Argument 12 (max_train_len): ${12}"
echo "Argument 13 (min_eval_len): ${13}"
echo "Argument 14 (max_eval_len): ${14}"
echo "Argument 15 (seed): ${15}"
echo "Argument 16 (mixer_type): ${16}"


cd copy_exp
python main.py --model $1 --layers $2 --mixer_rank $3 --lr $4 --epochs $5 --train_batch_size $6 --eval_batch_size $7 --train_task $8 --eval_task $9 --vocab_size ${10} --min_train_len ${11} --max_train_len ${12} --min_eval_len ${13} --max_eval_len ${14} --seed ${15} --mixer_type ${16} --use_wandb --layer_norm_eps 1e-6 --mixer_h_dep --use_short_conv --symm_mixer