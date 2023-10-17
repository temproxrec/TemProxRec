from datasets import dataset_factory
from .dataloader import Dataloader

def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = Dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()

    return train, val, test 