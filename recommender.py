"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import match_movie_title


def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """
    user = pd.DataFrame(user_rating, index=[0])
    user_t = user.T.reset_index()
    user_movie_entries = list(user_t["index"])
    movie_titles = list(movies["title"])
    intended_movies = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    
    # convert these movies to intended movies and convert them into movie ids
    recommend = movies.copy()
    recommend = recommend.reset_index()
    recommend = recommend.set_index("title")
    recommend.drop(intended_movies, inplace=True)
    random_movies = np.random.choice(list(recommend.index), replace=False, size=k)
    return random_movies  

def recommend_test_nmf(movies, user_rating, model, k=5):
    """
    return k random unseen movies for user 
    """
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
 
    data = list(user_rating.values())   # the ratings of the new user
    row_ind = [0]*len(data)       # we use just a single row 0 for this user 
    col_ind = list(user_rating.keys())  # the columns (=movieId) of the ratings
    # R.shape[1] = 168253
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, 168253))
   
    # 3. scoring
    # calculate the score with the NMF model
    scores = model.inverse_transform(model.transform(user_vec))
    # convert to a pandas series
    scores = pd.Series(scores[0])

    # 4. ranking

    # filter out movies allready seen by the user
    # give a zero score to movies the user has already seen
    scores[user_rating.keys()] = 0
    # sort the scores from high to low 
    scores = scores.sort_values(ascending=False)
    
    # return the top-k highst rated movie ids or titles
    recommendations = scores.head(k).index
    recommendations = movies.loc[recommendations]
    return list(recommendations["title"])

def recommend_with_NMF(user_item_matrix, user_rating, model, k=5):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds
    """
    
    # initialization - impute missing values    
    
    # transform user vector into hidden feature space
    
    # inverse transformation

    # build a dataframe

    # discard seen movies and sort the prediction
    
    # return a list of movie ids
    pass

def recommend_with_user_similarity(user_item_matrix, user_rating, k=5):
    pass
