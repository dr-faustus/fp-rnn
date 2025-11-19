import os

import importlib
import random
import torch
import torch.nn as nn
import numpy as np

if os.environ.get("FP_RNN_ENABLE_TORCH_COMPILE", "0") != "1":
    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(func):
                return func
            return decorator
        return fn
    if hasattr(torch, "compile"):
        torch.compile = _identity_compile

from argparse import ArgumentParser
from construct import construct_model
from datasets import create_iterator

from utils.lib import setup_log_folder, save_current_script, setup_logger, \
  count_parameters

import wandb

MODELS = "models"
TRAINERS = "trainers"
DATASETS = "datasets"

# general
parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device choice')

# model params
parser.add_argument('--model_name', type=str, default='Mamba1', choices=['Mamba1', 'Mamba2', 'FPMamba1', 'FPMamba2'])
parser.add_argument('--d_model', default=512, type=int, help="hidden size of the models, d_model for mamba/bedouin")
parser.add_argument('--d_state', default=16, type=int, help="sets the state dimension of the model.")
parser.add_argument('--d_mixer', default=None, type=int, help="the number of dimensions spanned by a mixer, only for bedouin")
parser.add_argument('--mixer_rank', default=1, type=int, help="number of components in mixer, only for bedouin and gated deltaproduct")
parser.add_argument('--mixer_type', default='kronecker', type=str, choices=['householderproduct', 'dplr', 'nonselective', 'kronecker', 'monarch', 'mixture', 'selectivedense'])
parser.add_argument('--max_iter', type=int, default=1000, help="maximum number of iterations for fixed-point")
parser.add_argument('--symm_mixer', action='store_true', help="use symmetric mixer for hidden state dependence")
parser.add_argument('--mixer_proj_rank', type=int, default=-1, help="mixer projection rank, default=full-rank")
parser.add_argument('--mixer_h_dep', action='store_true', help="previous hidden-state dependence for the mixer")
parser.add_argument('--n_backwards', type=int, default=1, help="num of backward passes.")
parser.add_argument('--use_short_conv', action='store_true', help='use short convolution in fp models')
parser.add_argument('--xlstm_setup', type=str, choices=["00", "01", "10", "11"], default="00", help='xlstm configuration')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
parser.add_argument('--n_layers', type=int, default=1, help="number of layers in the models")
parser.add_argument('--n_heads', type=int, default=8, help="number of heads, for mamba2, deltanet, deltaproduct, transformers")
parser.add_argument('--rms_norm', type=bool, default=False)
parser.add_argument('--fused_add_norm', type=bool, default=False)
parser.add_argument('--residual_in_fp32', type=bool, default=False)
parser.add_argument('--bias', type=bool, default=True)

# train parameters
parser.add_argument('--train_batch_size', type=int, default=32, help='train data batch size')
parser.add_argument('--eval_batch_size', type=int, default=128, help='test data batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay coefficient')
parser.add_argument('--beta1', type=float, default=0.9, help='adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--n_workers', type=int, default=2, help='num of workers on the dataloader')
parser.add_argument('--max_steps', type=int, default=15000, help='maximum number of steps, otherwise runs forever')
parser.add_argument('--early_stopping_steps', type=int, default=-1, help='early stopping patience')

# data parameters
parser.add_argument('--dataset_name', type=str, default='catbAbI_dataset', help='dataset name')
parser.add_argument('--trainer_name', type=str, default='catbAbI_trainer', help='trainer name')
parser.add_argument('--dataset_variation', type=str, default='catbAbI10k', help='variation of the catbabi dataset')
parser.add_argument('--whitelist', type=str, default="", help='which tasks to train on. default is all')
parser.add_argument('--ra_mode', type=bool, default=True, help='request answer only')
parser.add_argument('--seq_len', type=int, default=200, help='sequence length to train on')
parser.add_argument('--lr_warmup', type=int, default=-1, help='learning rate warmup iterations. default is none')

# log parameters
parser.add_argument('--use_wandb', action='store_true', help='write logs or not')
parser.add_argument('--log_every_n_steps', type=int, default=25, help='log step size')
parser.add_argument('--eval_every_n_steps', type=int, default=250, help='eval step size')
parser.add_argument('--eval_steps', type=int, default=-1, help='eval step size')
parser.add_argument('--log_folder', type=str, default="logs/", help='log folder path')

args = parser.parse_args()

if args.seed < 0:
    args.seed = int(10000 * np.random.rand(1)[0])

# build folder name
path = "{}/{}/{}_lr{}_bs{}/{}".format(
        "logs",
        "catbabi",
        args.model_name,
        args.lr,
        args.train_batch_size,
        args.seed
    )

setup_log_folder(path)
save_current_script(args.log_folder)

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # setup log folder and backup source code
    if args.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        os.environ["WANDB_SILENT"] = "true"                         # set this in .bashrc if needed
        wandb.login(key='8c80a699015158d475e03c1a11794f578631adf7') # in your terminal run: `wandb login --relogin`
        wandb.init(
            project='catbabi', 
            name="{}-{}-{}-{}-d_model-{}-n_layers-{}-lr-{}".format(args.model_name, args.seed, args.mixer_type, args.mixer_rank, args.d_model, args.n_layers, args.lr), 
            dir=args.log_folder,
            config=vars(args)
        )

    # setup logger
    log, logger = setup_logger(args.log_folder)
    log("{}".format(args))

    # import dataset
    log("load datasets ...")
    train_generator = create_iterator(args=args, partition="train", batch_size=args.train_batch_size)
    eval_generator = create_iterator(args=args, partition="valid", batch_size=args.eval_batch_size, random=False)
    test_generator = create_iterator(args=args, partition="test", batch_size=1, random=False)

    vocab_size = len(train_generator.dataset.idx2word)
    args.vocab_size = vocab_size
    log("dataset vocab size: {}".format(vocab_size))
    log("Number of train batches: {}".format(len(train_generator)))
    log("Number of test batches: {}".format(len(eval_generator)))

    # build model
    log("load model ...")
    model = construct_model(args)

    log("skipping model print ...")
    #log("{}".format(model))
    log("{} trainable parameters found. ".format(count_parameters(model)))

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # loss
    criterion = nn.CrossEntropyLoss(ignore_index=args.PAD)

    # create trainer
    log("load trainer ...")
    _module = importlib.import_module(TRAINERS + "." + args.trainer_name)
    trainer = _module.Trainer(model=model,
                              params=args,
                              train_generator=train_generator,
                              eval_generator=eval_generator,
                              optimizer=optimizer,
                              criterion=criterion,
                              log=log)

    # begin training
    trainer.train()

    log("\nloading best mode from: ", trainer.best_eval_state_path)
    trainer.load_state(trainer.best_eval_state_path)

    log("\nfinal batch_size=1 evaluation ...")
    trainer.evaluate(generator=test_generator, progress=True, eval_type='test')

if __name__ == '__main__':
    run(args, )
