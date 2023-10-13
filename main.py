from options import args
from collections import defaultdict
import argparse

from dataloaders import dataloader_factory
from models.modules.encoder import Encoder
#from trainers.pretrain import Trainer
from trainers.trainer import Trainer

from utils import *

# start a new wandb run to track this script

# track hyperparameters and run metadata
# # wandb.config.dr
    
def train():

    fix_seed(args)
    export_root = setup_train(args)

    print("Generating Dataset and Dataloader")
    train_loader, val_loader, test_loader = dataloader_factory(args)

    print("Building Encoder model")
    encoder = Encoder(args)

    print("Creating  Trainer")
    trainer = Trainer(encoder, args, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    trainer.test()

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
