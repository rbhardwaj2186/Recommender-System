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

user_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
    # For unknown tokens, we add an additional embedding!
     tf.keras.layers.Embedding(len(unique_user_ids)+1, embedding_dimenson)

])

movie_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
    tf.keras.layers.Embedding(len(unique_movie_titles)+1, embedding_dimension)

])

metrics = tfrs.metrics.FactorizedTopK(
    candidates = movies.batch(128).map(movie_model),
    k = 100
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)

class MovielensModel(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie.model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics
        return self.task(user_embeddings, positive_movie_embeddings)

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

mode.fit(cached_train, epochs=2)

model.evaluate(cached_test, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movie dataset.
index.index(movie.batch(100).map(model.movie_model), movies)






