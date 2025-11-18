import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow running as a script or module
import wandb
import argparse

import torch

from copy_exp.model_utils import get_model
from copy_exp.data_utils import get_train_dataset, get_tokenizer, get_eval_dataset
from copy_exp.train_utils import train, save_model
from copy_exp.test_utils import evaluation

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()

    ##task
    parser.add_argument('--train_task',choices=["copy","prefix_ngram","suffix_ngram"],
                        required=True, help="tasks to train the model")
    parser.add_argument('--eval_task',choices=["copy","prefix_ngram","suffix_ngram","duplicate_ngram"],
                        required=True, help="tasks to evaluate the model")
    
    parser.add_argument('--vocab_size', default=26, type=int, help="vocabulary size in the strings. maximum is 26.")
    
    parser.add_argument('--n_gram', default=0, type=int, 
            help='''length of the n-gram when training/evaluating on 'prefix_ngram', 'suffix_ngram', 'duplicate_ngram'.
            Set 0 for 'copy' task.''')
    parser.add_argument('--length_answer', default=0, type=int, 
            help="length of the answer to be returned. Set 0 if no constraint on the length of the answer.")

    #model
    parser.add_argument('--model', choices=['T_nope', 'T_rope', 'T_alibi', "T_hard_alibi", 'lstm', 'Mamba1', 'Mamba2', 'GatedDeltaNet', 'GatedDeltaProduct', 'FPRNN', 'FPMamba1', 'FPMamba2'],
            required=True, help='''models starting by 'T' are transformers with different positional embeddings. Other choices
            are mamba and lstm.''')
    parser.add_argument('--hidden_size', default=1024, type=int, help="Hidden size of the models")
    parser.add_argument('--layers', default=12, type=int, help="Number of layers in the models.")
    parser.add_argument('--heads', default=16, type=int, help="Number of heads in the transformer models.")
    parser.add_argument('--num_masked_heads', default=8, type=int, help='''Only when model = ''T_hard_alibi''. 
            Number of heads where we apply hard alibi. The remaining heads are set to nope.''')
    parser.add_argument('--state_dim', default=32, type=int, help='''Only when model = ''mamba''. 
            Sets the state dimension of the model.''')
    parser.add_argument('--d_mixer', default=1024, type=int, help="mixer dimension")
    parser.add_argument('--mixer_rank', default=1, type=int, help="number of components in mixer, only for bedouin and gated deltaproduct")
    parser.add_argument('--mixer_type', default='dplr', type=str, choices=['householderproduct', 'dplr', 'nonselective', 'kronecker', 'monarch', 'mixture'])
    parser.add_argument('--mixer_diag_h', action='store_true', help='use diagonal scaling for hidden state in mixer')
    parser.add_argument('--mixer_proj_rank', type=int, default=-1, help='mixer projection rank, default=full-rank')

    parser.add_argument('--symm_mixer', action='store_true', help="use symmetric mixer for hidden state dependence")
    parser.add_argument('--mixer_h_dep', action='store_true', help="previous hidden-state dependence for the mixer")
    parser.add_argument('--n_backwards', type=int, default=1, help="num of backward passes.")
    parser.add_argument('--use_short_conv', action='store_true', help='use short convolution in fp models')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)

    parser.add_argument('--max_iter', type=int, default=10, help="maximum number of iterations for fixed-point")

    #optimization
    parser.add_argument('--lr', default=1e-5, type=float, help="choice of learning rate")
    parser.add_argument('--epochs', default=1, type=int, help="number of epochs")
    parser.add_argument('--steps', default=2000, type=int, help="number of steps for each epoch")
    

    parser.add_argument('--train_batch_size', default=8, type=int, help="training batch size")
    parser.add_argument('--eval_batch_size', default=8, type=int, help="evaluation batch size")
    parser.add_argument('--eval_num_batches', default=3, type=int, help='''number of batches to use for evaluation.
            useful to have a mean + std over results.''')
    
    parser.add_argument('--min_train_len', default=5, type=int, help="minimum length of a training example")
    parser.add_argument('--max_train_len', default=20, type=int, help="maximum length of a training example")
    parser.add_argument('--min_eval_len', default=10, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--max_eval_len', default=20, type=int, help="maximum length of an evaluation example")
    
    ##context length
    parser.add_argument('--context_len', default=220, type=int, help="context length during training")
    parser.add_argument('--eval_context_len', default=220, type=int, help="context length at evaluation time")
    
    ##other stuff
    parser.add_argument('--ampdtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--gpu', default='0', type=str, help="gpu num")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--use_wandb', action='store_true', help='use weights and biases or not')
    parser.add_argument('--wandb_dir', type=str, default=Path(os.environ['LOGS']) / "wandb", help='directory for weights and biases output')
    parser.add_argument('--update_wandb_interval', type=int, default=10, help='weights and biases update interval, per batch')
    
    return parser.parse_args()


args = parse_args()
args.run_name = args.model + '-mixerrank=' + str(args.mixer_rank) + '-' + args.mixer_type + '-' + str(args.seed)

print(args)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(args.seed)

if args.use_wandb:
    """Initalizes a wandb run"""
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key='8c80a699015158d475e03c1a11794f578631adf7')
    wandb.init(
        project='copy', 
        name=args.run_name, 
        dir=args.wandb_dir,
        config=vars(args)
    )

## Get train dataset & tokenizer
tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args)
train_dataset = get_train_dataset(args, tokenizer)

batch = next(iter(train_dataset))

print("-"*100)
print(f"EXAMPLE {batch['input'][0]}")
print("-"*100)
print(batch['input_ids'][-1][batch['mask'][-1]==1], batch['input_ids'][-1], batch['input'][-1])
print("*"*100)

## Get model
model = get_model(args, tokenizer)

print("^"*100)
print(model)
print(f"Number of parameters of the model: {count_parameters(model)}")
print("^"*100)


## train the model
train(args,model,train_dataset,TO_TOKEN)



## save model
save_model(args, model)


## evaluation of the model

# model = torch.load('/is/sg2/smovahedi/codes/transformers_ssm_copy/transformers_ssm_copy/synthetic_exps/output_dir/model_EqRNN_layer_4_hidden_1024_train_copy_lr_5e-05_epochs_15_steps_2000/model.pt', weights_only=False)

print("###EVALUATION")

model.eval()

str_acc_mean_list, str_acc_std_list, char_acc_mean_list = evaluation(args, model,tokenizer,TO_TOKEN)


print(args)
print(str_acc_mean_list)
print(char_acc_mean_list)

print("DONE")

