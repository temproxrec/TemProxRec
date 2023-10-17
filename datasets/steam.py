from .base import AbstractDataset
import pandas as pd

class SteamDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steam' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['steam.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('steam.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid',  'timestamp'] #user_idm, movie_id, rating, time
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        years = [2014, 2015, 2016]
        df = df[df['year'].isin(years)]
        
        # year
        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df