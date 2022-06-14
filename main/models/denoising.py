import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Dense, Dropout

input_layer = Input(shape=(2807,))
encoded = Dropout(0.15)(input_layer)

encoded = Dense(1024, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)

decoded = Dense(1024, activation='relu')(encoded)
decoded = Dense(2807, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)