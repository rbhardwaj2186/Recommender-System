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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pprint

# Ratings data
ratings = tfds.load('movielens/100k-ratings', split='train')

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)

# Features of all the available movies
movies = tfds.load('movielens/100k-movies', split='train')

for x in movies.take(1).as_numpy_iterator():
    pprint(x).pprint(x)

tf.random.set_seed(42)
shuffled=ratings.shuffle(100_00, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movie.batch(1_000)
user_ids = ratings.batch(1_00_000).map(lambda x: x['user_id'])