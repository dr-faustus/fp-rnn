#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate bedouin

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Launch torch distributed run on 8 devices
# torchrun \
#   --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
#   --standalone --nnodes=1 --nproc_per_node=8 \
#   ~/codes/plainLM/train.py --config=$config --job_idx=$job_idx

# Launch torch distributed run on 4 devices
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone --nnodes=1 --nproc_per_node=4 \
  ~/codes/plainLM/train.py --config=$config --job_idx=$job_idx

# Launch torch distributed run on 2 devices
# torchrun \
#   --redirects 1:0,2:0 \
#   --standalone --nnodes=1 --nproc_per_node=2 \
#   ~/codes/plainLM/train.py --config=$config --job_idx=$job_idx