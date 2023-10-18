import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
#

raw_data = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz'
)

print(raw_data.head())