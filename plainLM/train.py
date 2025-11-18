"""Pretrain a Transformer on language modeling."""
import os
import sys
from pathlib import Path
import tempfile

# Choose a temp directory that avoids NFS churn: prefer local scratch if present, then home, else /tmp.
def _pick_tmp():
  candidates = []
  local_scratch = os.environ.get("LOCAL_SCRATCH")
  if local_scratch:
    candidates.append(os.path.join(local_scratch, "tmp"))
  user = os.environ.get("USER")
  if user:
    candidates.append(f"/fast/{user}/tmp")
  candidates.append(os.path.expanduser("~/tmp"))
  candidates.append("/tmp")
  for path in candidates:
    try:
      os.makedirs(path, exist_ok=True)
      return path
    except Exception:
      continue
  return "/tmp"

home_tmp = _pick_tmp()
os.environ.setdefault("TMPDIR", home_tmp)
os.environ.setdefault("TEMP", home_tmp)
tempfile.tempdir = home_tmp

# Keep huggingface cache off NFS too
hf_cache = os.path.join(home_tmp, "hf-cache")
os.makedirs(hf_cache, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)

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
  
  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)
  
  # Model
  model, model_cfg = construct_model(cfg)
  
  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  print_master(cfg)
  
  # Training
  print_master("=== Start Training! ===")
  metrics = defaultdict(list)
  train_losses = []
  
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start+1):
    step = micro_step // cfg.grad_accumulation_steps
    if step > cfg.steps_budget:
      break

    # Train
    train_loss = engine.step(micro_batch)
    train_losses.append(train_loss)

    # Eval
    valid_loss = None
    if step % cfg.eval_every_steps == 0:
      print_master("Evaluating on validation set")
      if cfg.eval:
        valid_loss = engine.eval(validloader)
      else:
        valid_loss = engine.eval(validloader, max_itfr=1)
    
    # Log
    if step % cfg.log_every_steps == 0:
      if master_process:
        utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size, model)
      train_losses = []
    
    # Checkpoint
    if master_process and cfg.save_intermediate_checkpoints \
        and micro_step % cfg.save_every_steps == 0:
      save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # End of training: log and save checkpoint
  print_master(f"=== Training Completed! ===")
  try:
    print_master("Evaluating on validation set - final")
    if cfg.eval:
      valid_loss = engine.eval(validloader)
    else:
      valid_loss = engine.eval(validloader, max_iter=100)
    if master_process:
      utils.log(cfg, metrics, micro_step + 1, train_losses, valid_loss, engine.optimizer, world_size)
  except:
    pass
  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
