
import pandas as pd
from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate
import numpy as np
import os


class UIPairFeatures():
    def tuple_to_surprise_dataset(self, tupl):
        """
        This function convert a subset in the tuple form to a `surprise` dataset. 
        """
        ratings_dict = {
            "userID": tupl[0],
            "itemID": tupl[1],
            "rating": tupl[2],
        }

        df = pd.DataFrame(ratings_dict)

        reader = Reader(rating_scale=(1, 5))

        # The columns must correspond to user id, item id and ratings (in that order).
        dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

        return dataset

    def stack_pair_features(self, data_tuple):
        vector_len = self.P.shape[1]
        pair_features_list = list()
        for usid, itid in zip(data_tuple[0], data_tuple[1]):
            user_vector = np.ones((vector_len,))
            item_vector = np.ones((vector_len,))

            if usid in self.train_users_ids:
                user_vector = self.P[self.trainset.to_inner_uid(usid)]

            if itid in self.train_items_ids:
                item_vector = self.Q[self.trainset.to_inner_iid(itid)] 

            age = self.all_users_age_M[usid]
            gender = self.all_users_gender_M[usid]
            year = self.all_movies_year_M[itid]
            
            pair_features_list.append(np.append(user_vector, np.append(item_vector, [age, gender, year])))
        
        return np.array(pair_features_list)
    
    def ratings_to_binary(self, ratings):
        true_ratings = np.where(ratings > 4.5, 1, 0)
        return true_ratings

    ## Below we train an SVD model and get its vectors, then add user and item
    # features to create input data for a classifier
    def train_pair_features(self, train_tuple):
        self.trainset = self.tuple_to_surprise_dataset(train_tuple).build_full_trainset()
        print("SVD(n_factors=10, n_epochs=10000, random_state=42)")
        algo = SVD(n_factors=50, n_epochs=1000, random_state=42)
        algo.fit(self.trainset)

        self.Q = np.array(algo.qi)
        self.P = np.array(algo.pu)

        all_users_df = pd.read_csv(os.path.join('..', "data_movie_lens_100k/", "user_info.csv"))
        self.all_users_id = all_users_df['user_id'].values
        self.all_users_age_M = all_users_df['age'].values
        self.all_users_gender_M = all_users_df['is_male'].values

        all_movies_df = pd.read_csv(os.path.join('..', "data_movie_lens_100k/", "movie_info.csv"))
        self.all_movies_id = all_movies_df['item_id'].values
        self.all_movies_year_M = all_movies_df['release_year'].values
        
        self.train_users_ids = train_tuple[0]
        self.train_items_ids = train_tuple[1]

        pair_features = self.stack_pair_features(train_tuple)
        true_ratings = self.ratings_to_binary(train_tuple[2])

        print(pair_features.shape)
        return (pair_features, true_ratings)

    def pair_features_with_ratings(self, data_tuple):
        pair_features = self.stack_pair_features(data_tuple)
        binary_ratings = self.ratings_to_binary(data_tuple[2])
        return (pair_features, binary_ratings)
        
    def pair_features_no_ratings(self, data_tuple):
        pair_features = self.stack_pair_features(data_tuple)
        return pair_features