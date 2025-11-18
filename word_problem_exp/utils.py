import os
import wandb
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_wandb(wandb_project, wandb_run_name, wandb_dir, config):
    """Initalizes a wandb run"""
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"                         # set this in .bashrc if needed
    wandb.login(key='8c80a699015158d475e03c1a11794f578631adf7') # in your terminal run: `wandb login --relogin`
    wandb.init(
        project=wandb_project, 
        name=wandb_run_name, 
        dir=wandb_dir,
        config=config
    )

def save_checkpoint(micro_step, model, optimizer, scheduler, output_dir, exp_name):    
    state = {
        "micro_step": micro_step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else {},
    }

    exp_dir = os.path.join(output_dir, exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    
    save_path = os.path.join(exp_dir, f'ckpt_micro_step_{micro_step}.pth')
    print(f"Saving checkpoint to {save_path}")
    torch.save(state, save_path)
    print(f"Successfully saved checkpoint!")

def load_checkpoint(micro_step, model, output_dir, exp_name, device):
    exp_dir = os.path.join(output_dir, exp_name)
    load_path = os.path.join(exp_dir, f'ckpt_micro_step_{micro_step}.pth')
    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device=device)

    return model

def get_param_groups(model):
    """ Create param groups without and with weight_decay.
    Doc: https://docs.pytorch.org/docs/stable/optim.html#per-parameter-options """

    # filter out parameters that do not require grad
    named_param_dict = {n: p for n,p in model.named_parameters() if p.requires_grad}

    # filter out parameters with names containing 'bias', 'norm', etc
    decay_params_names = [n for n, p in model.named_parameters() if not getattr(p, '_no_weight_decay', False)] # exclude mamba 'A_log', 'D'
    decay_params_names = [n for n in decay_params_names if "bias" not in n] # exclude bias
    decay_params_names = [n for n in decay_params_names if "norm" not in n] # exclude normalization layers

    decay_params = [p for n, p in named_param_dict.items() if n in decay_params_names]
    no_decay_params = [p for n, p in named_param_dict.items() if n not in decay_params_names]

    # # sanity check
    # no_decay_param_names = [n for n, p in named_param_dict.items() if n not in decay_params_names]
    # print(f"\nParameters with no weight decay:")
    # print(*no_decay_param_names, sep='\n')
    # print(f"\nParameters with weight decay:")
    # print(*decay_params_names, sep='\n')

    return [dict(params=decay_params), dict(params=no_decay_params, weight_decay=0.0)]