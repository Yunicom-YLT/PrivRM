# PrivRM: A Framework for Range Mean Estimation under Local Differential Privacy

This repository implements the paper: [PrivRM: A Framework for Range Mean Estimation under Local Differential Privacy](https://dl.acm.org/doi/abs/10.1145/3725414) accepted at SIGMOD2025.

The aim of this project is explore the problem of range mean estimation under LDP. We propose a novel framework PrivRM, which is capable of integrating all the existing numerical mechanisms. 


## Requirements

The code is implemented in Python 3.11. Please refer to `requirements.txt` to see all the required packages.

## Usage

To reproduce the experiments, run the script `exp.py` and modify the parameters: `dataname`, `espilon`, `range size`, `nvp` and `method`.
