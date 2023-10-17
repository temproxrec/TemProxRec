from options import args
from dataloaders import dataloader_factory
from models.modules.encoder import Encoder
from trainers.trainer import Trainer
from utils import *

def train():

    fix_seed(args)
    export_root = setup_train(args)
    
    print("Generating Dataset and Dataloader")
    train_loader, val_loader, test_loader = dataloader_factory(args)

    print("Building Encoder model")
    encoder = Encoder(args)

    print("Creating  Trainer")
    trainer = Trainer(encoder, args, train_loader, val_loader, test_loader, export_root)
    
    if args.mode == 'test_only':
        trainer.test()
        
    elif args.mode =='train':
        trainer.train()
        trainer.test()
    

if __name__ == '__main__':
    if args.mode == 'train' or 'test_only':
        train()
    else:
        raise ValueError('Invalid mode')