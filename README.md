# TemproxRec (Temporal-Proximity-aware Recommendation model)

This repository provides PyTorch implementation of TemproxRec from the paper "Sequential Recommendation on Temporal Proximities with Contrastive Learning and Self-Attention"

## Environment Setup

We test code with python 3.11.3 on ubuntu with `cuda 11.4`

Install the required packages into your python environment:
```
pip install -r requirements.txt
```

## Trained Networks
The trained networks representing the best performances from our experiments for all datasets are available on [our anonymous Google Drive](https://drive.google.com/drive/folders/1xMVpgk2vs3S8k3OOpui9zDdkT-cz6X28) 

Their training scripts with optimal parameters are described in the 'run.sh'.

To reproduce the experiments, download all the folders and add them to the `./experiments/`. Then, test the trained networks.

## Experimental reproduction

### For Training
Training TemproxRec is produced by using the `train.py`. For example, to train on Amazon 'beauty'
```
python main.py --dataset_code='beauty' --clip_time=128 --interval=60 --temperature=0.05 --lamb=0.3 --mode=train
```
The running scripts for the best experiment for all datasets are described in the 'run.sh'

### For testing only
By changing the 'mode' to 'test_only' on the training code, the best model stored in the training stage can be tested.
```
python main.py --dataset_code='beauty' --clip_time=128 --interval=60 --temperature=0.05 --lamb=0.3 --mode=test_only
```

### For Data Preparation

We experimeted with four dataset : **Amazon Beauty**, **Video**, **Books**, and **Steam**.

Preprocessed data for all are stored in `./Data/preprocessed`. You can run the above training and test code with the preprocessed data.

Law data for Beauty and Video datasets are stored in `./Data/beauty` and `./Data/video`.

Due to capacity issues, the law data for Books and Steam datasets are available on [our anonymous Google Drive](https://drive.google.com/drive/folders/168xjW9GeqX1OwipPxshSAtW8GaKDEMQB). 

Download the folder "books" and "steam" from the drive and add them to the TemproxRec's code in `./Data/`.
