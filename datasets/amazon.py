from .base import AbstractDataset
import pandas as pd

class AmazonDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['beauty.csv']


    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('beauty.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp'] #user_idm, movie_id, rating, time
        
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        # 2011년 ~ 2014년 데이터만 사용
        df = df[df['year'].isin([2011, 2012, 2013, 2014])]
        
        df = df[['uid', 'sid', 'rating', 'timestamp']]

        # year
        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df
    

class MoviesDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'movies' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Movies_and_TV.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Movies_and_TV.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp'] #user_idm, movie_id, rating, time
        
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        # 2011년 ~ 2014년 데이터만 사용
        df = df[df['year'].isin([2009, 2010, 2011, 2012])]
        
        df = df[['uid', 'sid', 'rating', 'timestamp']]
        
        # year
        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df



class CDsDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'cds' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['CDs_and_Vinyl.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('CDs_and_Vinyl.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp'] #user_idm, movie_id, rating, time
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        # 2004년 ~ 2007년 데이터만 사용
        df = df[df['year'].isin([2004, 2005, 2006])]
        df = df[['uid', 'sid', 'rating', 'timestamp']]

        # year
        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    
    
class VideosDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'video' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Video_Games.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Video_Games.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp'] #user_idm, movie_id, rating, time
        
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        # 2004년 ~ 2007년 데이터만 사용
        df = df[df['year'].isin([2011, 2012, 2013, 2014])]
        df = df[['uid', 'sid', 'rating', 'timestamp']]
        
        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df


class BooksDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'books' #수정 

    @classmethod
    def url(cls):
        return 
    
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Books.csv']


    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Books.csv')
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp'] #user_idm, movie_id, rating, time
        
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year

        # 2011년 ~ 2014년 데이터만 사용
        df = df[df['year'].isin([2011, 2012, 2013])]
        
        df = df[['uid', 'sid', 'rating', 'timestamp']]

        min = df.timestamp.min()
        df['time'] = df['timestamp'].apply(lambda x:(x-min).days) 
        df.reset_index(drop=True, inplace=True)
        
        return df
