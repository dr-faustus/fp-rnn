# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Type

import numpy as np

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

parser = ArgumentParser()
parser.add_argument("--config", default="parity_xlstm11")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument('--gpu', default='0', type=str, help="gpu num")
parser.add_argument('--use_wandb', action='store_true', help='use weights and biases or not')
parser.add_argument('--wandb_dir', type=str, help='directory for weights and biases output')
parser.add_argument('--update_wandb_interval', type=int, default=10, help='weights and biases update interval, per batch')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.optim as optim
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from arithmetic_exp.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from arithmetic_exp.data.utils import DataGen
from arithmetic_exp.lr_scheduler import LinearWarmupCosineAnnealing

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def create_save_directory(cfg, seed):
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory name
    dir_name = f"results/saved_models_{cfg.model.name}_{timestamp}_seed_{seed}"

    # Create the directory
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def save_wandb_run_id(save_dir):
    run_id = wandb.run.id
    with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run_id)


def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN or Inf detected in {name}")
        return True
    return False


def get_available_dtype(device):
    if device == 'cuda':
        # check that device supports bfloat16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print('Warning: bfloat16 is not supported on this device. Using float32 instead.')
            return torch.float16
    elif device == 'mps':
        return torch.float32
    else:
        return torch.bfloat16


def load_dataset(name, kwargs):
    cls = dataset_registry[name]
    return cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: DictConfig, seed: int, lr: float, batch_size: int):
    print(OmegaConf.to_yaml(cfg))
    save_dir = create_save_directory(cfg, seed)
    cfg.dataset.kwargs.seed = seed
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    if 'FP' in cfg.model.name:
        args.run_name = cfg.model.name + '-mixerrank=' + str(cfg.model.mixer_rank) + '-' + cfg.model.mixer_type + '-' + str(cfg.dataset.kwargs.seed)
    else:
        args.run_name = cfg.model.name + '-' + str(cfg.dataset.kwargs.seed)
    # Initialize wandb
    if args.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        os.environ["WANDB_SILENT"] = "true"
        wandb.login(key='8c80a699015158d475e03c1a11794f578631adf7')
        wandb.init(
            project='mod_arithmetic',
            name=args.run_name,
            dir=args.wandb_dir,
            config=OmegaConf.to_container(cfg)
        )
    seed_everything(cfg.dataset.kwargs.seed)
    # cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)

    train_loader = DataLoader(dataset.train_split, batch_size=cfg.training.batch_size)
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size) for key, val_ds in dataset.validation_split.items()
    }
    train_metrics = dataset.train_metrics.to(device=cfg.training.device)
    val_metrics = dataset.validation_metrics.to(device=cfg.training.device)
    if cfg.model.name == "SimpleRecurrent":
        from lm_model import SimpleRecurrentNet
        model = SimpleRecurrentNet(cfg.model).to(cfg.training.device)
    elif 'FP' in cfg.model.name:
        from models import FPLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
            d_model=cfg.model.d_model,
            n_layer=cfg.model.n_layers,
            vocab_size=cfg.dataset.kwargs.vocab_size,
            pad_vocab_size_multiple=4,
            ssm_cfg=dict(layer=cfg.model.name, 
                         d_mixer=cfg.model.d_mixer,
                         mixer_type=cfg.model.mixer_type,
                         mixer_rank=cfg.model.mixer_rank,
                         mixer_proj_rank=cfg.model.mixer_proj_rank,
                         max_iter=cfg.model.max_iter,
                         symm_mixer=False,
                         mixer_h_dep=True,
                         n_backwards=1,
                         norm_eps=1e-6,
                         use_short_conv=True),
        )
        if 'Mamba' in cfg.model:
            config.ssm_cfg['d_state'] = cfg.model.d_state
        model = FPLMHeadModel(config)
    elif cfg.model.name == 'Mamba1' or cfg.model.name == 'Mamba2':
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        config = MambaConfig(
            d_model=cfg.model.d_model,
            n_layer=cfg.model.n_layers,
            vocab_size=cfg.dataset.kwargs.vocab_size,
            pad_vocab_size_multiple=4,
            ssm_cfg=dict(layer=cfg.model.name),
        )
        model = MambaLMHeadModel(config)
    elif cfg.model.name == 'GatedDeltaProduct':
        from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM
        from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
        config = GatedDeltaProductConfig(hidden_size=cfg.model.d_model,
                                            vocab_size=cfg.dataset.kwargs.vocab_size,
                                            num_heads=cfg.model.n_heads,
                                            head_dim=32, # from DeltaProduct Paper
                                            num_hidden_layers=cfg.model.n_layers,
                                            num_householder=cfg.model.mixer_rank)
        model = GatedDeltaProductForCausalLM(config)
    elif cfg.model.name == 'GatedDeltaNet':
        from models import GatedDeltaNetForCausalLM
        from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
        config = GatedDeltaNetConfig(hidden_size=cfg.model.d_model,
                                     vocab_size=cfg.dataset.kwargs.vocab_size,
                                     num_heads=cfg.model.n_heads,
                                     head_dim=32,
                                     num_hidden_layers=cfg.model.n_layers,
                                     attn_mode="chunk")
        model = GatedDeltaNetForCausalLM(config)
    else:
        from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
        model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
            device=cfg.training.device
        )
    if cfg.training.compile:
        print('Compiling model...')
        model = torch.compile(model, mode='default')
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    
    weights_dtype = torch_dtype_map[cfg.training.weight_precision]
    available_dtype = get_available_dtype(cfg.training.device)
    model = model.to(dtype=weights_dtype, device=cfg.training.device)

    if args.use_wandb:
        wandb.config.update({"dtype_used": str(available_dtype), "weights_dtype": str(weights_dtype)})

    if hasattr(model, '_create_weight_decay_optim_groups'):
        optim_groups = model._create_weight_decay_optim_groups()
    else:
        optim_groups = []
        optim_groups.append(
            [param for name, param in model.named_parameters() if not hasattr(param, '_no_weight_decay')])
        optim_groups.append([param for name, param in model.named_parameters() if hasattr(param, '_no_weight_decay')])

    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )

    # Training loop
    step = 0
    epoch = 1
    running_loss = 0.0

    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        for inputs, labels in monitoring:
            monitoring.set_description_str(f"Steps {step + 1}/{cfg.training.num_steps} (Epoch: {epoch})")
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            model.train()
            optimizer.zero_grad()
            with torch.autocast(
                    device_type=cfg.training.device,
                    dtype=available_dtype,
                    enabled=cfg.training.enable_mixed_precision,
                    cache_enabled=False
            ):
                # Inside the training loop
                if check_nan_inf(inputs, "inputs") or check_nan_inf(labels, "labels"):
                    print(f"Warning: NaN or Inf in input data at step {step}. Skipping this batch.")
                    continue

                outputs = model(inputs.to(device=cfg.training.device))
                if not torch.is_tensor(outputs):
                    try:
                        outputs = outputs['logits']
                    except:
                        outputs = outputs[0]
                loss = nn.functional.cross_entropy(outputs.view(-1, cfg.model.vocab_size), labels.view(-1), ignore_index=-1)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss: {loss} encountered at step {step}. Skipping this batch.")
                    continue
                loss.backward()
                optimizer.step()
                # After optimizer.step()
                for name, param in model.named_parameters():
                    if check_nan_inf(param.data, f"parameter {name}"):
                        print(f"Warning: NaN or Inf in model parameters after update at step {step}")
                        break
                lr_scheduler.step()
                running_loss = loss
            step += 1
            train_metrics.update(outputs, labels)
            if step % cfg.training.val_every_step == 0:
                if 'FP' in cfg.model.name:
                    print(
                        f"\nStep [{step + 1}/{cfg.training.num_steps}] (Epoch: {epoch}), Loss: {running_loss:.4f},"
                        f" Metrics: {train_metrics.compute()},"
                        f" FP itrs: {int(np.median(model.backbone.layers[0].mixer.fixed_point_iter))}"
                    )
                else:
                    print(
                        f"\nStep [{step + 1}/{cfg.training.num_steps}] (Epoch: {epoch}), Loss: {running_loss:.4f},"
                        f" Metrics: {train_metrics.compute()}"
                    )
                if args.use_wandb:
                    # Log training metrics to wandb
                    metric_dict = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": running_loss,
                        **{f"train_{k}": v for k, v in train_metrics.compute().items()}
                    }

                    if 'FP' in cfg.model.name:
                        for idx, layer in enumerate(model.backbone.layers):
                            metric_dict['FP_itrs_' + str(idx)] = layer.mixer.fixed_point_iter[-1]
                    wandb.log(metric_dict)
                train_metrics.reset()

                # Validation loop
                for vl_name, val_loader in val_loaders.items():
                    model.eval()
                    val_loss = 0.0
                    val_metrics.reset()
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(device=cfg.training.device)
                            val_labels = val_labels.to(device=cfg.training.device)
                            with torch.autocast(
                                    device_type=cfg.training.device,
                                    dtype=available_dtype,
                                    enabled=cfg.training.enable_mixed_precision,
                            ):
                                val_outputs = model(val_inputs)
                                if not torch.is_tensor(val_outputs):
                                    try:
                                        val_outputs = val_outputs['logits']
                                    except:
                                        val_outputs = val_outputs[0]
                                loss = nn.functional.cross_entropy(
                                    val_outputs.view(-1, cfg.model.vocab_size),
                                    val_labels.view(-1),
                                    ignore_index=-1,
                                )
                                val_loss += loss.item()
                                val_metrics.update(val_outputs, val_labels)
                        val_loss /= len(val_loader)
                        print(
                            f"Validation[{vl_name}] Loss: {val_loss:.4f},"
                            f" Metrics: {val_metrics.compute()}"
                        )
                        metric_dict = {
                            "step": step,
                            f"val_{vl_name}_loss": val_loss,
                            **{f"val_{vl_name}_{k}": v for k, v in val_metrics.compute().items()}
                        }
                        if 'FP' in cfg.model.name:
                            for idx, layer in enumerate(model.backbone.layers):
                                metric_dict['FP_itrs_' + str(idx)] = layer.mixer.fixed_point_iter[-1]
                        '''
                        if cfg.model.name == "simple_recurrent" and cfg.model.layer_type == "diagonal":
                            sharpening_factors = {
                                f'sharpening_factor_{i}': layer.recurrent_layer.sharpening_factor.item() for
                                i, layer in zip(range(len(model.block_stack)), model.block_stack)}
                            metric_dict.update(sharpening_factors)

                            sharpening_factors_grad = {
                                f'sharpening_factor_{i}_grad': layer.recurrent_layer.sharpening_factor.grad.item() for
                                i, layer in zip(range(len(model.block_stack)), model.block_stack)}
                            metric_dict.update(sharpening_factors_grad)
                        '''
                        if args.use_wandb:
                            # Log validation metrics to wandb
                            wandb.log(metric_dict)

            if step >= cfg.training.num_steps:
                break
        epoch += 1

    # Save the model at the end of training
    model_save_path = os.path.join(save_dir, f"model_{cfg.model.name}_seed_{seed}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    if args.use_wandb:
        # Log the saved model file to wandb
        wandb.save(model_save_path)

    # Save the configuration
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to {config_save_path}")
    if args.use_wandb:
        save_wandb_run_id(save_dir)

        # Log the saved configuration file to wandb
        wandb.save(config_save_path)

        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    with open('./configs/' + str(args.config), "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg, args.seed, args.lr, args.batch_size)
