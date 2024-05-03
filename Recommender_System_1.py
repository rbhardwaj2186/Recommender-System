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
