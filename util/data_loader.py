import pandas as pd
import os

from util.models import Dataset


class DataLoader:
    def __init__(self, user_size:int=1000 , test_size:int=5, data_path:str="ml-10M100K"):
        self.user_size= user_size
        self.test_size = test_size
        self.data_path = data_path


    def load(self) -> (pd.DataFrame, pd.DataFrame): #다볃합한거, 영화랑 태그병합한거 리턴함
        cols = ["movie_id", "title","genre"]
        movies=pd.read_csv(os.path.join(self.data_path,"movies.dat"), encoding="latin-1", names=cols,
                           sep="::")
        movies['genre'] = movies['genre'].apply(lambda x:x.split("|"))

        cols = ['user_id', 'movie_id','tag','timestamp']
        tags = pd.read_csv(os.path.join(self.data_path, "tags.dat"), encoding='latin-1',
                           names=cols, sep="::")
        tags['tag'] = tags['tag'].str.lower()

        tags=tags.groupby("movie_id").agg({"tag":list})
        movies = movies.merge(tags,on="movie_id", how="left")

        cols = ['user_id','movie_id','rating','timestamp']
        ratings = pd.read_csv(os.path.join(self.data_path, "ratings.dat"), encoding='latin-1',
                              sep="::",names=cols)
        valid_user_ids = sorted(ratings.user_id.unique())[:self.user_size]
        #10000부터 시작해서 20000에서끝난다하면 11000까지 뽑겟지
        ratings = ratings[ratings.user_id.isin(valid_user_ids)]
        ratings = ratings.merge(movies,on="movie_id")

        return ratings,movies

    def split_data(self, data)->(pd.DataFrame,pd.DataFrame):
        #각사용자의 최신5건은 테스트 나머지는 훈련으로 할거기때문에 rating_orders를 만들고
        data['rating_order'] = data.groupby("user_id")['timestamp'].rank(ascending=False,
                                                                         method="first")
        test = data[data.rating_order<=self.test_size]
        train = data[data.rating_order>self.test_size]

        return train,test


    def main(self):
        ratings, movies = self.load()
        train, test = self.split_data(ratings)
        user_love_items = test[test.rating>=4].groupby("user_id").agg({'movie_id':list})['movie_id'].to_dict()

        return Dataset(train,test,user_love_items,movies)




