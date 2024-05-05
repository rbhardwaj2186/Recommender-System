import csv
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.stats import linregress
import os
import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ratings data
ratings = tfds.load('movielens/100k-ratings', split='train')

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)

movies = tfds.load('movielens/100k-movies', split='train')

for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

#movie_titles = movies.batch(1_000)
#user_ids = ratings.batch(1_000_000).map(lambda x: x['user_id'])

#unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
##unique_movie_titles = np.unique(np.concatenate(list(movie_titles.flat_map(lambda x: x['movie_title']))))
#unique_user_ids = np.unique((np.concatenate(list(user_ids))))
# Flatten and extract movie titles
movie_titles = movies.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x['movie_title']]))
#unique_movie_titles = np.unique(np.concatenate(list(movie_titles.as_numpy_iterator())))
unique_movie_titles = np.unique(np.array(list(movie_titles.as_numpy_iterator())))


# Extract user IDs
user_ids = ratings.map(lambda x: x['user_id'])
#unique_user_ids = np.unique(np.concatenate(list(user_ids.as_numpy_iterator())))
unique_user_ids = np.unique(np.array(list(user_ids.as_numpy_iterator())))

print(unique_movie_titles[:4])



