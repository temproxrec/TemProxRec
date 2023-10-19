# TemproxRec (Temporal-Proximity-aware Recommendation model)

This repository provide PyTorch implementation of TemproxRec from the paper "Sequential Recommendation on Temporal Proximities with Contrastive Learning and Self-Attention"

## Environment Setup

We test code with python 3.11.3 on ubuntu with `cuda 11.4`

Install the required packages into your python environment:
```
pip install -r requirements.txt
```

## Trained Networks
The trained network representing the best performances from our experimnets are availabe on [our anonymous Google Drive](https://drive.google.com/drive/folders/1xMVpgk2vs3S8k3OOpui9zDdkT-cz6X28)

## Experimental reproduction

### For Training
Training TemproxRec is produced by using the `train.py`. For example, to train on Amazon 'beauty'
```
python main.py --dataset_code='beauty' --clip_time=128 --interval=60 --temperature=0.05 --lamb=0.3 --mode=train
```
The running scripts for the best experiment of all datasets ares described in the 'run.sh'

### For testing only
By changing the 'mode' to 'test_only' on the training code, the best model stored in the training stage can be tested.
```
python main.py --dataset_code='beauty' --clip_time=128 --interval=60 --temperature=0.05 --lamb=0.3 --mode=test_only
```

### For Data Preparation

We experimeted with four dataset : **Amazon Beauty**, **Video**, **Books**, and **Steam**
The preprocessed data for all are stored in `./Data/preprocessed` 
The law data for Beauty and Video are stored in `./Data/beauty` and `./Data/video`
For law data for Books and Steam are availbalte the are availabe on [our anonymous Google Drive](https://drive.google.com/drive/folders/168xjW9GeqX1OwipPxshSAtW8GaKDEMQB).
