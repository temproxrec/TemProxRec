import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim as optim
import numpy as np
from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

            
def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}


    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.item()

    return metrics



def setup_train(args):
    #set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root


def create_experiment_export_folder(args):
    file_name = f'{args.dataset_code}' # _{args.batch_size}batch{args.hidden_units}hidden_{args.interval}interval_{args.temperature}temperature_{args.maxlen}maxlen'
    experiment_dir =  args.experiment_dir + '/'+ file_name
    print(experiment_dir)
    experiment_description =  args.experiment_description
    
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
        
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    json_result = {k: v.item() for k, v in result.items()}
    
    with filepath.open('w') as f:
        json.dump(json_result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))


def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)




class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)