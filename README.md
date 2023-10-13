# TemproxRec (Temporal-Proximity-aware Recommendation model)

This repository provide PyTorch implementation of TemproxRec from the paper "Sequential Recommendation on Temporal Proximities with Contrastive Learning and Self-Attention"

## Environment Setup

We test code with python 3.11.3 on ubuntu with `cuda 11.4`

Install the required packages into your python environment:
```
pip install -r requirements.txt
```

## Experimental reproduction

Training TemproxRec is produced using the `train.py`. For example, to train on Amazon 'beauty'
```
python main.py --dataset_code='beauty' --time_unit_divide=128 --interval=7 --temperature=0.05 --w2=0.3
```
The running scripts for the best experiment of all datasets ares described in the 'run.sh'
