"""Pretrain a Transformer on language modeling."""
import sys
from pathlib import Path

from absl import app, flags
from collections import defaultdict

if __package__ is None or __package__ == "":
  sys.path.append(str(Path(__file__).resolve().parent.parent))

from plainLM import utils
from plainLM.utils import print_master
from plainLM.torch_utils import pytorch_setup, destroy_ddp
from plainLM.data import get_dataloaders
from plainLM.checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from plainLM.construct import construct_model
from plainLM.engine import TorchEngine

import numpy as np

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
  
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  
  #if master_process:
  #  utils.maybe_make_dir(cfg, JOB_IDX)

  #if cfg.use_wandb and master_process:
  #  utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  ckpt, micro_step = maybe_load_checkpoint(cfg, device)
  
  # Dataset
  _, validloader = get_dataloaders(cfg)
  
  # Model
  model, model_cfg = construct_model(cfg)

  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  print_master(cfg)
  
  # Validation
  metrics = defaultdict(list)
  train_losses = []
  
  print('cfg.valid_seqlens:', cfg.valid_seqlens)
  for seqlen in cfg.valid_seqlens:
    print_master(f"Evaluating on validation set: seqlen={seqlen}")
    engine.seq_len = seqlen
    if hasattr(model, 'freqs_cis'):
      model.update_freq_cis(seqlen)

    if cfg.eval:
        valid_loss = engine.eval(validloader, max_iter=None)

    if master_process:
        utils.log(cfg, metrics, micro_step + 1, train_losses, valid_loss, engine.optimizer, world_size)
    
    print(metrics)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
