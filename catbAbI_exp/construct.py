from torch import nn
from collections import OrderedDict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def construct_model(cfg):
    """Initalize a model from config."""

    if 'FP' in cfg.model_name:
        from models import FPLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
            d_model=cfg.d_model,
            n_layer=cfg.n_layers,
            vocab_size=cfg.vocab_size,
            ssm_cfg=dict(layer=cfg.model_name, 
                         d_mixer=cfg.d_mixer,
                         mixer_type=cfg.mixer_type,
                         mixer_rank=cfg.mixer_rank,
                         mixer_proj_rank=cfg.mixer_proj_rank,
                         symm_mixer=cfg.symm_mixer,
                         mixer_h_dep=cfg.mixer_h_dep,
                         n_backwards=cfg.n_backwards,
                         max_iter=cfg.max_iter,
                         norm_eps=cfg.layer_norm_eps,
                         use_short_conv=cfg.use_short_conv),
            )
        if 'Mamba' in cfg.model_name:
            config.ssm_cfg['d_state'] = cfg.d_state
        if 'Mamba2' in cfg.model_name:
            config.ssm_cfg['ngroups'] = cfg.n_heads
        model = FPLMHeadModel(config)

    elif cfg.model_name == 'Mamba1' or cfg.model_name == 'Mamba2':
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        config = MambaConfig(
            d_model=cfg.d_model,
            n_layer=cfg.n_layers,
            vocab_size=cfg.vocab_size,
            ssm_cfg=dict(layer=cfg.model_name, 
                         d_state=cfg.d_state),
        )
        model = MambaLMHeadModel(config)
    
    return model
    
