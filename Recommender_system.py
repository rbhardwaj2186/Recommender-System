import tensorflow as tf
import tensorflow_datasets as tfds
ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda  x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})