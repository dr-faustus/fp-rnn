
from fractions import Fraction
import wandb


def construct_model(cfg):
  """Initalize a model from config. Counts parameters."""

  vocab_size = getattr(cfg, 'n_vocab', None) or getattr(cfg, 'vocab_size', None)
  if vocab_size is None:
    raise AttributeError("Config missing vocab size; expected `n_vocab` or `vocab_size`.")

  if 'FP' in cfg.model:
    from models import FPLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    config = MambaConfig(
        d_model=cfg.d_model,
        n_layer=cfg.n_layers,
        vocab_size=vocab_size,
        ssm_cfg=dict(layer=cfg.model, 
                      d_mixer=cfg.d_mixer,
                      mixer_type=cfg.mixer_type,
                      mixer_rank=cfg.mixer_rank,
                      mixer_proj_rank=cfg.mixer_proj_rank,
                      symm_mixer=cfg.symm_mixer,
                      mixer_h_dep=cfg.mixer_h_dep,
                      n_backwards=cfg.n_backwards,
                      max_iter=cfg.max_iter,
                      norm_eps=cfg.layer_norm_eps,
                      use_short_conv=cfg.use_short_conv,
                      glu_in_loop=cfg.glu_in_loop,
                      bc_norm=cfg.bc_norm,),
    )
    if 'Mamba' in cfg.model:
        config.ssm_cfg['d_state'] = cfg.d_state
    model = FPLMHeadModel(config)

  elif cfg.model == 'Mamba1' or cfg.model == 'Mamba2':
      from mamba_ssm.models.config_mamba import MambaConfig
      from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
      config = MambaConfig(
          d_model=cfg.d_model,
          n_layer=cfg.n_layers,
          vocab_size=vocab_size,
          ssm_cfg=dict(layer=cfg.model, 
                        d_state=cfg.d_state),
      )
      model = MambaLMHeadModel(config)

  else:
    raise NotImplementedError(f"Not implemented model: {cfg.model}.")

  model_cfg = config
  
  if hasattr(model, 'count_params'):
    n_params = model.count_params(non_embedding=False)
    n_params_no_embed = model.count_params(non_embedding=True)
    print(f"Number of parameters: {n_params:_}")
    print(f"Number of non-embedding parameters: {n_params_no_embed:_}")
    if wandb.run is not None:
      wandb.log({
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed
      })
  
  return model, model_cfg


def get_param_groups(model, weight_decay):
  """Create param groups with and withou weight_decay."""
  
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
  
  param_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": no_decay_params, "weight_decay": 0.0},
  ]
  
  return param_groups
