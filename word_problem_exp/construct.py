from torch import nn
from collections import OrderedDict

def construct_model(cfg):
    """Initalize a model from config."""

    if cfg.tagging:

        if 'DenseRNN' in cfg.model:
            from models.densernn import DenseRNN
            model = nn.Sequential(OrderedDict(
                embedding = nn.Embedding(cfg.n_vocab, cfg.d_model),
                layers = nn.Sequential(*[DenseRNN(d_model=cfg.d_model, 
                                                    d_mixer=cfg.d_mixer,
                                                    mixer_type=cfg.mixer_type,
                                                    mixer_rank=cfg.mixer_rank,
                                                    mixer_proj_rank=cfg.mixer_proj_rank,
                                        ) for i in range(cfg.n_layers)]),
                lm_head =  nn.Linear(cfg.d_model, cfg.n_vocab, bias=False),
            ))
            #model.lm_head.weight = model.embedding.weight
            return model

        if 'FP' in cfg.model:
            from models import FPLMHeadModel
            from mamba_ssm.models.config_mamba import MambaConfig
            config = MambaConfig(
                d_model=cfg.d_model,
                n_layer=cfg.n_layers,
                vocab_size=cfg.n_vocab,
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
                             use_short_conv=cfg.use_short_conv),
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
                vocab_size=cfg.n_vocab,
                ssm_cfg=dict(layer=cfg.model, 
                             d_state=cfg.d_state),
            )
            model = MambaLMHeadModel(config)

        elif cfg.model == 'Transformers':
            from models.transformers import EncoderTokenClassifier
            model = EncoderTokenClassifier(
                d_model=2 * cfg.d_model,
                n_heads=8,
                d_ff=2048,
                dropout=0.0,
                activation='gelu',
                layer_norm_eps=cfg.layer_norm_eps,
                norm_first=False,
                n_layers=cfg.n_layers,
                weight_sharing=False,
                weight_scale=1.0,
                n_vocab=cfg.n_vocab,
                batch_first=True,
                bias=cfg.bias,
            )

        elif cfg.model == 'GatedDeltaProduct':
            from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM
            from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
            config = GatedDeltaProductConfig(hidden_size=cfg.d_model,
                                             vocab_size=cfg.n_vocab,
                                             num_heads=cfg.n_heads,
                                             head_dim=32, # from DeltaProduct Paper
                                             num_hidden_layers=cfg.n_layers,
                                             num_householder=cfg.mixer_rank,
                                             allow_neg_eigval=True,
                                             use_short_conv=False,
                                             use_forget_gate=False)
            model = GatedDeltaProductForCausalLM(config)

        elif cfg.model == 'GatedDeltaNet':
            from models import GatedDeltaNetForCausalLM
            from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
            config = GatedDeltaNetConfig(hidden_size=cfg.d_model,
                                         vocab_size=cfg.n_vocab,
                                         num_heads=cfg.n_heads,
                                         head_dim=32,
                                         num_hidden_layers=cfg.n_layers,
                                         attn_mode="chunk")
            model = GatedDeltaNetForCausalLM(config)

        elif cfg.model == 'LSTM':
            # from lstm import LSTMTokenClassifier
            from models.lstm import LSTM
            # model = LSTMTokenClassifier(d_model=cfg.d_model,
            #                             n_layers=cfg.n_layers,
            #                             n_vocab=cfg.n_vocab)
            model = LSTM(embedding_dim=cfg.d_model, vocab_size=cfg.n_vocab, num_layers=cfg.n_layers, dropout_rate=0.0)
            
        elif cfg.model == 'xLSTM':
            from xlstm import xLSTMLMModel, xLSTMLMModelConfig
            from dacite import from_dict

            config = dict()
            config['vocab_size'] = cfg.n_vocab
            config['num_blocks'] = cfg.n_layers
            config['embedding_dim'] = cfg.d_model
            if cfg.xlstm_setup == "00":
                config['mlstm_block'] = {'mlstm': {'num_heads': 4}}
                config['slstm_block'] = {'slstm': {'num_heads': 4, 'conv1d_kernel_size': 0}}
                config['slstm_at'] = [0]
            elif cfg.xlstm_setup == "01":
                config['mlstm_block'] = {'mlstm': {'num_heads': 4}}
                config['slstm_block'] = {'slstm': {'num_heads': 4, 'conv1d_kernel_size': 0}}
                config['slstm_at'] = [0, 1]
            elif cfg.xlstm_setup == "10":
                config['mlstm_block'] = {'mlstm': {'num_heads': 4}}
                config['slstm_block'] = {}
                config['slstm_at'] = []
            else:
                config['mlstm_block'] = {'mlstm': {'num_heads': 4}}
                config['slstm_block'] = {'slstm': {'num_heads': 4, 'conv1d_kernel_size': 0}}
                config['slstm_at'] = [1]
            model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, config))

    else:
        raise NotImplementedError("Upgrade to variable-cls index classifier!")
        raise NotImplementedError("SSMs only support Tagging")
    
    return model
    