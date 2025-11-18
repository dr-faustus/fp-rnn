import os
from models import (
        LSTM,
        GPTNeoXAlibiForCausalLM,
        GPTNeoXHardAlibiForCausalLM,
        GPTNeoXNoPEForCausalLM
        )
from transformers import  GPTNeoXForCausalLM, GPTNeoXConfig

def get_model(args, tokenizer):
    if 'FP' in args.model:
        from models import FPLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
                d_model=args.hidden_size,
                n_layer=args.layers,
                vocab_size=len(tokenizer),
                ssm_cfg=dict(layer=args.model, 
                             d_mixer=args.d_mixer,
                             mixer_type=args.mixer_type,
                             mixer_rank=args.mixer_rank,
                             max_iter=args.max_iter,
                             symm_mixer=args.symm_mixer,
                             mixer_proj_rank=args.mixer_proj_rank,
                             mixer_h_dep=args.mixer_h_dep,
                             n_backwards=args.n_backwards,
                             use_short_conv=args.use_short_conv),
            )
        if 'Mamba' in args.model:
            config.ssm_cfg['d_state'] = args.state_dim
        model = FPLMHeadModel(config)

    elif args.model == "Mamba1" or args.model == "Mamba2":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
                d_model=args.hidden_size,
                n_layer=args.layers,
                vocab_size=len(tokenizer),
                ssm_cfg=dict(layer=args.model, 
                             d_state=args.state_dim),
            )
        model = MambaLMHeadModel(config)
    
    elif args.model == 'GatedDeltaProduct':
        from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM
        from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
        config = GatedDeltaProductConfig(hidden_size=args.hidden_size,
                                            vocab_size=len(tokenizer),
                                            num_heads=args.heads,
                                            head_dim=32, # from DeltaProduct Paper
                                            num_hidden_layers=args.layers,
                                            num_householder=args.mixer_rank)
        model = GatedDeltaProductForCausalLM(config)

    elif args.model == 'GatedDeltaNet':
        from models import GatedDeltaNetForCausalLM
        from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
        config = GatedDeltaNetConfig(hidden_size=args.hidden_size,
                                        vocab_size=len(tokenizer),
                                        num_heads=args.heads,
                                        head_dim=32,
                                        num_hidden_layers=args.layers,
                                        attn_mode="chunk")
        model = GatedDeltaNetForCausalLM(config)

    elif args.model == "lstm":
        model = LSTM(
                embedding_dim=args.hidden_size,
                vocab_size=len(tokenizer),
                num_layers=args.layers,
                dropout_rate=0.65
                )
    
    if args.model in ["T_nope","T_rope","T_alibi"]:
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=args.heads,
                    num_hidden_layers=args.layers,
                    vocab_size=len(tokenizer),
                    )
    elif args.model == "T_hard_alibi":
        config = GPTNeoXConfig(
                    bos_token_id=0,
                    eos_token_id=0,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.hidden_size*4,
                    num_attention_heads=args.heads,
                    num_hidden_layers=args.layers,
                    num_masked_heads=args.num_masked_heads,
                    vocab_size=len(tokenizer),
                    )
    
    if args.model == "T_rope":
        model = GPTNeoXForCausalLM(config)
    elif args.model == "T_nope":
        model = GPTNeoXNoPEForCausalLM(config)
    elif args.model == "T_alibi":
        model = GPTNeoXAlibiForCausalLM(config)
    elif args.model == "T_hard_alibi":
        model = GPTNeoXHardAlibiForCausalLM(config)

    return model


