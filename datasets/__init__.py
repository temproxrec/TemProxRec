from .amazon import AmazonDataset, CDsDataset, MoviesDataset, VideosDataset, BooksDataset
from.steam import SteamDataset

DATASETS = {
    AmazonDataset.code(): AmazonDataset,
    MoviesDataset.code():MoviesDataset,
    CDsDataset.code():CDsDataset,
    VideosDataset.code():VideosDataset,
    SteamDataset.code():SteamDataset,
    BooksDataset.code():BooksDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
