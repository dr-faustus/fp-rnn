import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow running as a script or module
from pprint import pformat
from functools import partial
from contextlib import nullcontext
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
log = get_logger(__name__)

from word_problem_exp.construct import construct_model
from data.word_problem.dataset import get_dataset, get_single_dataset, get_dataloaders, pad_collate
from word_problem_exp.utils import init_wandb, save_checkpoint, load_checkpoint, count_parameters, get_param_groups
from word_problem_exp.metrics import (
    ce_loss,
    reduce_metrics,
    sequence_accuracy,
    token_accuracy,
    compute_metrics
)

@torch.no_grad()
def eval(model, dataloader, accelerator, tokenizer, metric_fns, ctx, prefix='val', k_min=None, k_step=1):
    k_max = next(iter(dataloader))["input_ids"].shape[1] # get dataloader seqlen
    k_min = k_max if k_min is None else (k_min+1) # +1 for BOS token
    assert k_min <= k_max

    ks = list(range(k_min, k_max+1, k_step))
    eval_results = {k:[] for k in ks}
    
    model.eval()
    for k in (ks):
        for batch in tqdm(dataloader, desc="Eval", position=1, leave=False, mininterval=1, maxinterval=30):
            source = batch["input_ids"][:,:k]
            target = batch["labels"][:,:k]
            with ctx:
                output = model(source)
            if not torch.is_tensor(output):
                output = output[0]
                
            predictions, references = accelerator.gather_for_metrics((output, target))

            # for k in (ks): # if len(ks)==0 else tqdm(ks, desc=f"k", position=2, leave=False)):
            eval_results[k].append(compute_metrics(
                        [(predictions,#[:,:k], 
                          references,#[:,:k])
                        )],
                        prefix=prefix,
                        tokenizer=tokenizer,
                        model=model,
                        metric_fns=metric_fns,
            ))

    eval_results = {k:reduce_metrics(eval_results[k]) for k in ks}

    if len(ks) == 1: # if k_min is None
        eval_results = eval_results[k_max]
    return eval_results

def train(cfg):
    """Run word problem experiment."""
    accelerator = Accelerator()
    log.info('Args: ' + str(cfg))
    log.info('Device name: ' + torch.cuda.get_device_name())
    
    set_seed(cfg.seed)

    # Load dataset
    datadict = get_dataset(
        group=cfg.group,
        k=cfg.k,
        strict_len=cfg.strict_len,
        train_size=cfg.train_size,
        data_dir=cfg.data_train_dir,
        supervised=cfg.tagging,
        max_samples=cfg.max_samples,
    )
    dataset = datadict["dataset"]
    cfg.n_vocab = datadict["n_vocab"]
    tokenizer = datadict["tokenizer"]
    collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    # Set up logger
    log.info(f"Config:\n {pformat(vars(cfg))}")
    log.info(f"Dataset:\n {dataset}")

    # Construct model
    model = construct_model(cfg)
    optimizer = optim.AdamW(
        get_param_groups(model), # set weight_decay=0 for bias, norm and p._no_weight_decay=True flag
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=True
    )

    log.info(model)
    
    train_dataloader, eval_dataloader = get_dataloaders(cfg.batch_size, cfg.train_size, datadict, drop_last=(cfg.model == 'FPRNN'))

    ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, cfg.ampdtype), cache_enabled=False)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if cfg.compile:
        log.info("Compiling model...")
        model = torch.compile(model)
        log.info("Model compiled!")

    # with ctx: # print model summary and test/warmup one forward pass
    #     batch = next(iter(train_dataloader))
    #     summary(model=model, input_data=batch["input_ids"], depth=10, mode='eval')

    metric_fns = {
        "loss": ce_loss,
        "sequence_accuracy": token_accuracy,
    }

    if cfg.tagging:
        metric_fns["sequence_accuracy"] = sequence_accuracy
        metric_fns["token_accuracy"] = token_accuracy


    global_step = 0
    best_val_acc = 0.0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * cfg.epochs)
    for epoch in (n_bar := tqdm(range(cfg.epochs), desc="Epochs")):
        # break
        model.train()
        if 'FP' in cfg.model:
            for layer in model.backbone.layers:
                layer.mixer.fixed_point_iter = []
        for batch_num, batch in enumerate(
            t_bar := tqdm(train_dataloader, desc="Train", position=1, leave=False)#, mininterval=1, maxinterval=30)
        ):
            if 'FP' in cfg.model and cfg.max_iter < 0:
                shape = 4.0
                for layer in model.backbone.layers:
                    layer.mixer.set_max_iter(int(np.ceil(np.random.gamma(shape=shape, scale=(cfg.k / 4) / shape + 1))))


            optimizer.zero_grad()

            source = batch["input_ids"]
            target = batch["labels"]

            with ctx:
                output = model(source)
                if not torch.is_tensor(output):
                    output = output[0]

                #print(target.dtype, output.dtype) 
                loss = F.cross_entropy(output.flatten(end_dim=-2), target.flatten())
            accelerator.backward(loss)

            if cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip, norm_type=2.0
                )

            optimizer.step()

            predictions, references = accelerator.gather_for_metrics((output, target))
            metrics = compute_metrics(
                    [(predictions, references)],
                    prefix="train",
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    model=model,
                    metric_fns=metric_fns,
                    reduce_batch=True
                )

            # if batch_num % 1000 == 0:
            postfix = {"loss": f"{loss.item():.5f}"}
            if 'FP' in cfg.model:
                postfix.update({'FP iters': [v['median'] for k,v in metrics.items() if 'fp_iters_layer' in k]})
            t_bar.set_postfix(postfix)

            if cfg.use_wandb and batch_num % cfg.update_wandb_interval == 0:
                wandb.log(metrics, step=global_step)
            global_step += 1
        
            scheduler.step()
        
        if 'FP' in cfg.model:
            for layer in model.backbone.layers:
                layer.mixer.fixed_point_iter = []
        eval_metrics = eval(model, eval_dataloader, accelerator, tokenizer, metric_fns, ctx=ctx)
        
        eval_metrics['epoch'] = epoch
        if eval_metrics["val/sequence_accuracy"] > best_val_acc:
            best_val_acc = eval_metrics["val/sequence_accuracy"]
        eval_metrics["val/best_sequence_accuracy"] = best_val_acc

           
        if cfg.use_wandb:
            wandb.log(eval_metrics, step=global_step)

        n_bar.set_postfix({"val/acc": f"{eval_metrics['val/sequence_accuracy']:.3f}"})
        save_checkpoint(epoch * len(train_dataloader), model, optimizer, scheduler, cfg.output_dir, cfg.run_name)

    if 'FP' in cfg.model:
        for layer in model.backbone.layers:
            layer.mixer.fixed_point_iter = []
            layer.mixer.optimizer.max_iter = cfg.max_iter
    
    # model = load_checkpoint(micro_step=1 * len(train_dataloader), model=model, output_dir=cfg.output_dir, exp_name=cfg.run_name, device='cuda')

    datadict = get_single_dataset(
            group=cfg.group,
            k=cfg.k_eval_max,
            train_size=1,
            data_dir=cfg.data_test_dir,
            supervised=cfg.tagging,
            max_samples=cfg.max_samples,
        )

    eval_dataloader = accelerator.prepare(DataLoader(
        datadict['dataset'],
        shuffle=False,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True, # workaround: sometimes the compiler complains with varying batchsizes
    ))

    k_eval_metrics = eval(model, eval_dataloader, accelerator, tokenizer, metric_fns, ctx=ctx, k_min=cfg.k_eval_min, k_step=cfg.k_eval_step, prefix='test')
    
    for k_eval, metrics in k_eval_metrics.items():
        metrics['k_eval'] = k_eval
        if cfg.use_wandb:
            wandb.log(metrics, step=global_step)

    print(k_eval_metrics)
    print({k_eval: metrics['test/sequence_accuracy'] for k_eval, metrics in k_eval_metrics.items()})
    print({k_eval: [v['median'] for k,v in metrics.items() if 'fp_iters_layer' in k] for k_eval, metrics in k_eval_metrics.items()})
    print(cfg)
    print(f"Number of parameters of the model: {count_parameters(model)}")

    accelerator.end_training()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--group', choices=["S5", "Z60", "A4", "A5"], required=True, help="the task group")
    parser.add_argument('--k', type=int, default=10, help="sequence length for the training task")
    parser.add_argument('--k_eval_min', default=2, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--k_eval_max', default=50, type=int, help="maximum length of an evaluation example")
    parser.add_argument('--k_eval_step', default=1, type=int, help="inteval of sequence length between evaluation example")
    parser.add_argument('--data_train_dir', type=str, default=Path(os.environ['DATA']) / 'word_problem')
    parser.add_argument('--data_test_dir', type=str, default=Path(os.environ['DATA']) / 'word_problem')
    parser.add_argument('--strict_len', type=bool, default=False)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--tagging', type=bool, default=True)
    parser.add_argument('--max_samples', type=int, default=None)

    # Model parameters
    parser.add_argument('--model', choices=['Mamba1', 'Mamba2', 'GatedDeltaNet', 'LSTM', 'xLSTM', 'Transformers', 'GatedDeltaProduct', 'DenseRNN', 'FPRNN', 'FPMamba1', 'FPMamba2'], required=True, help="model choice")
    parser.add_argument('--d_model', default=512, type=int, help="hidden size of the models, d_model for mamba/bedouin")
    parser.add_argument('--d_state', default=16, type=int, help="sets the state dimension of the model.")
    parser.add_argument('--d_mixer', default=None, type=int, help="the number of dimensions spanned by a mixer, only for bedouin")
    parser.add_argument('--mixer_rank', default=1, type=int, help="number of components in mixer, only for bedouin and gated deltaproduct")
    parser.add_argument('--mixer_type', default='dplr', type=str, choices=['householderproduct', 'dplr', 'nonselective', 'kronecker', 'monarch', 'mixture', 'selectivedense'])
    parser.add_argument('--max_iter', type=int, default=10, help="maximum number of iterations for fixed-point")
    parser.add_argument('--symm_mixer', action='store_true', help="use symmetric mixer for hidden state dependence")
    parser.add_argument('--mixer_proj_rank', type=int, default=-1, help="mixer projection rank, default=full-rank")
    parser.add_argument('--mixer_h_dep', action='store_true', help="previous hidden-state dependence for the mixer")
    parser.add_argument('--n_backwards', type=int, default=1, help="num of backward passes.")
    parser.add_argument('--use_short_conv', action='store_true', help='use short convolution in fp models')
    parser.add_argument('--xlstm_setup', type=str, choices=["00", "01", "10", "11"], default="00", help='xlstm configuration')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('--n_layers', type=int, default=1, help="number of layers in the models")
    parser.add_argument('--n_heads', type=int, default=12, help="number of heads, for mamba2, deltanet, deltaproduct, transformers")
    parser.add_argument('--rms_norm', type=bool, default=False)
    parser.add_argument('--fused_add_norm', type=bool, default=False)
    parser.add_argument('--residual_in_fp32', type=bool, default=False)
    parser.add_argument('--bias', type=bool, default=True)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="choice of learning rate")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--op_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay coefficient')
    parser.add_argument('--compile', type=bool, default=False)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--max_val_acc', type=float, default=0.99)
    parser.add_argument('--ampdtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')

    # Misc
    parser.add_argument('--gpu', default='0', type=str, help="gpu num")
    parser.add_argument('--seed', default=0, type=int, help="random seed") # randint(0, 2**32 - 1)
    parser.add_argument('--use_wandb', action='store_true', help='use weights and biases or not')
    parser.add_argument('--wandb_dir', type=str, default=Path(os.environ['LOGS']) / "wandb", help='directory for weights and biases output')
    parser.add_argument('--update_wandb_interval', type=int, default=10, help='weights and biases update interval, per batch')
    parser.add_argument('--output_dir', type=str, default=Path(os.environ['RESULTS']) / 'word_problem', help='output directory for saving the model')
    
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.run_name = args.group + '-k=' + str(args.k) + '-' + args.model + '-mixerrank=' + str(args.mixer_rank) + '-' + args.mixer_type +  '-' + str(args.seed)

    if args.model in ['GatedDeltaProduct']:
        args.ampdtype = 'bfloat16' # force bfloat16

    if args.use_wandb:
        init_wandb(wandb_project='wordproblem',
                wandb_run_name=args.run_name,
                wandb_dir=args.wandb_dir,
                config=vars(args))

    train(cfg=args)
