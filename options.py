#from templates import set_template
from datasets import DATASETS
#from dataloaders import DATALOADERS
#from models import MODELS
#from trainers import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train','test_only'])

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)


################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='beauty', choices=DATASETS.keys())
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=5, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')

################
# Dataloader
################
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)

################
# NegativeSampler
################
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=25, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
# contrastive learning called TCL #
parser.add_argument('--temperature', type=float, default=0.1, help = 'Temperature of TCL Loss')
parser.add_argument('--interval', type=int, default=50, help = 'Radius of a time window in TCL')
parser.add_argument('--lamb', type=float, default=0.1, help = 'Weight for TCL Loss')
parser.add_argument('--tcl', type=str, default='yes')

################
# Model
################
parser.add_argument('--maxlen', type=int, default=50, help='Length of sequence for bert')
parser.add_argument('--num_items', type=int, help='Number of total items')
parser.add_argument('--num_time_items', type=int, help='Number of total time categories')
parser.add_argument('--hidden_units', type=int, default=32, help='Size of hidden vectors (d_model)')
parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
parser.add_argument('--mask_prob', type=float, default=0.2, help='Probability for masking items in the training sequence')
parser.add_argument('--clip_time', type=int, default=512, help='clipping value for time interval')

###############
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='downstream')

################
args = parser.parse_args()

