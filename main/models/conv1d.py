import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, UpSampling1D

input_layer = Input(shape=(400, 7))

encoded = Conv1D(8, 5, activation = 'relu', padding = 'same')(input_layer)
encoded = MaxPool1D(4, padding='same')(encoded)
encoded = Conv1D(8, 5, activation = 'relu', padding = 'same')(encoded)
encoded = MaxPool1D(4, padding='same')(encoded)

decoded = Conv1D(8, 5, activation = 'relu', padding = 'same')(encoded)
decoded = UpSampling1D(4)(decoded)
decoded = Conv1D(8, 5, activation = 'relu', padding = 'same')(decoded)
decoded = UpSampling1D(4)(decoded)
decoded = Conv1D(7, 5, activation = 'sigmoid', padding = 'same')(decoded)

autoencoder = Model(input_layer, decoded)