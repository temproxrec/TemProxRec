from abc import *
from utils import *
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pathlib import Path
import os
import tempfile
import shutil
import pickle
from datetime import datetime

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = args
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    
    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    

    
    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        df = self.load_ratings_df()
        df = self.filter_triplets(df)
        df, umap, smap, tmap = self.densify_index(df)
        #df2 = df[['uid', 'sid', 'time']] #--> for sasrec data set
        #df2.to_csv('%s.txt'%self.args.dataset_code, sep='\t', header=None, index = False ) #--> for sasrec data set
        train, val, test, train_t, val_t, test_t= self.split_df(df, len(umap))
        dataset = {'train': train,
                   'train_abstime': train_t,
                   'val': val,
                   'val_abstime': val_t,
                   'test': test,
                   'test_abstime': test_t,
                   'umap': umap,
                   'smap': smap,
                   'tmap': tmap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)
    
    
    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    
    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df
    

    def densify_index(self, df):
        print('Densifying index')

        umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}
        tmap = {t: (i+1) for i, t in enumerate(np.arange(df['time'].max()+1))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        df['time'] = df['time'].map(tmap)
        return df, umap, smap, tmap

    
    
    def split_df(self, df, user_count):
        
        print('Donwnstream Splitting')
        
        user_group = df.groupby('uid')
        user2items = user_group.apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
        user2times = user_group.apply(lambda d: list(d.sort_values(by='timestamp')['time']))
        
        train, val, test = {}, {}, {}
        train_t, val_t, test_t = {}, {}, {}
        for user in range(1, user_count+1):
            # if user in trainval_idx:
            items = user2items[user]
            times = user2times[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            train_t[user], val_t[user], test_t[user] = times[:-2], times[-2:-1], times[-1:]
        
            
        return train, val, test, train_t, val_t, test_t

    def _get_rawdata_root_path(self):
        return Path('Data')

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_uc{}-min_sc{}-split{}'.format(self.code(), self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)
    

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')