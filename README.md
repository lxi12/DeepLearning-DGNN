## Dynamic-Graph-Neural-Networks
This repository is modified from the source code (https://github.com/IBM/EvolveGCN) of paper Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), in AAAI, 2020.

Also it has refers to the following resources: https://github.com/njuhtc/LEDG , https://github.com/xkcd1838/bench-DGNN , https://github.com/benedekrozemberczki/pytorch_geometric_temporal .

Thank the authors of EvolveGCN for well-written codes, and all others for the great job. 

## Data
- bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html

For downloaded datasets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher, install: https://pytorch.org/
  * Python 3.7 - 3.9
  * PyTorch_Geometric

## Task
Link prediction on dataset, would two traders interact bitcoin with each other in the future, a binary classification task.

## Methods
GCN (Graph Convolutional Networks) 

EGCN_O: a version of EvolveGCN, GCN + LSTM

Skip_GCN: experiment, apply a random matrix see how bad is the performance

## Usage

Download the repository to a folder, set the command prompt (or Anaconda prompt) working directory to it.  

Run below script:

```sh
python run_exp.py --config_file ./parameters_bitcoin_alpha_linkpt_meta_gcn.yaml
```

It will run the experiment of using Skip_GCN / GCN / EvolveGCNO on the bitcoinalpha dataset.

The yaml file in the folder contains the hyperparameters for model training. 

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs.

## Have fun~ 
