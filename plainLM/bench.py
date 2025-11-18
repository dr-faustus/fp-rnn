"""Pretrain a Transformer on language modeling."""
import os
import sys
from pathlib import Path

os.environ['TMPDIR'] = '/tmp'

from absl import app, flags
from collections import defaultdict
from tqdm import tqdm
from triton.testing import do_bench

if __package__ is None or __package__ == "":
  sys.path.append(str(Path(__file__).resolve().parent.parent))

from plainLM import utils
from plainLM.utils import print_master
from plainLM.torch_utils import pytorch_setup, destroy_ddp
from plainLM.data import get_dataloaders
from plainLM.checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from plainLM.construct import construct_model
from plainLM.engine import TorchEngine
from copy import deepcopy
import torch
print(torch.cuda.get_device_name())


import numpy as np

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx

  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
  cfg2 = cfg._replace(grad_accumulation_steps=-1) # bench only 
  cfg2 = deepcopy(cfg)._replace(model='bedouin_compileable')
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  
  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  #ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)
  
  # Model
  model1, model_cfg = construct_model(cfg)
  model2, model_cfg = construct_model(cfg2)
  model2.load_state_dict(deepcopy(model1.state_dict()))
  model3 = torch.compile(model2)



  engine1 = TorchEngine(model1, cfg, device, local_rank, None)
  engine2 = TorchEngine(model2, cfg2, device, local_rank, None)
  engine3 = TorchEngine(model3, cfg2, device, local_rank, None)

  micro_batch = next(iter(trainloader))
  engine1.step(micro_batch), engine2.step(micro_batch)
  
  print(f'params diff: {max((p1-p2).abs().max().item() for (p1, p2) in zip(model1.parameters(), model2.parameters())):.3}')
  print(f'gradient diff: {max((p1.grad-p2.grad).abs().max().item() for (p1, p2) in zip(model1.parameters(), model2.parameters())):.3}')

  ms1 = do_bench(lambda: engine1.step(micro_batch), grad_to_none=model1.parameters())
  print(f'no compile: {ms1/1e3:.3}s')
  ms2 = do_bench(lambda: engine2.step(micro_batch), grad_to_none=model2.parameters())
  print(f'compile step: {ms2/1e3:.3}s')
  #ms3 = do_bench(lambda: engine3.step(micro_batch), grad_to_none=model3.parameters())
  #print(f'compile full: {ms3/1e3:.3}s')
  print("done!")


if __name__ == "__main__":
  app.run(main)
