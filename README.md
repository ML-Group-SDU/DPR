# DPR: 
  
This code is based on <a href="https://github.com/pickxiguapi/Clean-Offline-RLHF">[ Clean-Offline-RLHF]</a>.

## Getting Started

Install PyTorch & torchvision.
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Install extra dependencies.
```bash
pip install -r requirements.txt
```

## Usage

### Pre-train Reward Models

Here we provide the bash files:
```bash
cd rlhf
bash scripts/train_mujoco.sh
bash scripts/run_adroit.sh
```

### Train Offline RL with Pre-trained Rewards 

Here we provide the bash files:
```bash
bash scripts/run_mujoco.sh
bash scripts/run_adroit.sh
```

## Acknowledgements

This repo relies on the following existing codebases:

- The Offline RLHF framework on <a href="https://github.com/pickxiguapi/Clean-Offline-RLHF">Clean-Offline-RLHF</a>
- The diffusion model variant based on <a href="https://github.com/NVlabs/DRAIL">DiffAIL</a> and <a href="">DRAIL</a>
