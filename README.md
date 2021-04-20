# Implementation of Paper Auto-Reid 

This is a fast re-implementation of paper [Auto-ReID: Searching for a Part-aware ConvNet for Person Re-Identification](https://arxiv.org/abs/1903.09776). 

## Features & Introductions. 

+ Please note that this repo is based on Cuda 8.0+ and Cudnn related to them. 

+ This model is tested on Cuda 9.0 9.1 and 10.0 and related cudnn libs on ubuntu server 14 and 16.

+ However, please do not use a batch size smaller than 2*your_gpus, because that will cause bn failure. At the same time, check that your PyTorch > = 1.0 and slurm is installed on your server.
Also, please note that if sub batch size (batch size // world_size) is compared small, it may have a decreased final performance. A preferred setting is batch size 128-512. However, for DARTS searching structures, which may require a higher single card memory, batch_size 16-64 have been tested successfully to get the correct result. 

+ The training could be visualized through TensorboardX.
All the checkpoints and searched graph structures are automatically saved into subfolder based on the task_name.

Update August 2th: integrate DARTS genotypes into tensorboard, you could see how structure changes during training right now!


### Features to add 

+ Add apex support for distributed training.
+ Improve the paper
+ Install as a lib
  





## Dependency & Install

```bash
conda create -n auto-reid python=3.6
conda install pip
conda install -c pytorch torch torchvision
conda install pyyaml
pip install graphviz
pip install tensorboardX
pip install apex
python setup.py install
```

## Run Training and Searching 

```
bash scripts/prepare_datasets/prepare_market.py
srun -n your_node_nums --gres gpu:gpunums -p your_partition python train_baseline_search_triplet.py --distributed True --config configs/Retrieval_classification_DARTS_distributed_triplet.yaml
```

## Visualize the training



