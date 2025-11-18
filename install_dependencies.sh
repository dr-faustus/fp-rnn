conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 cudatoolkit cudatoolkit-dev tqdm wandb accelerate pathlib scipy fire pyrootutils matplotlib seaborn -c pytorch -c nvidia

pip install torch torchvision torchaudio tqdm wandb accelerate pathlib scipy fire pyrootutils matplotlib seaborn transformers datasets dacite omegaconf ordered_set polars --use-deprecated=legacy-resolver
conda install cudatoolkit cudatoolkit-dev
pip install mamba-ssm causal-conv1d 
pip install --pre flash-linear-attention