import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape=(2807,))

def basic(n_layers, latent_dim, activation='relu'):
    
    encoded = Dense(n_layers*latent_dim, activation=activation)(input_layer)
    for current_dim in range(n_layers - 1, 1, -1):
        encoded = Dense(current_dim*latent_dim, activation=activation)(encoded)
    
    decoded = Dense(2*latent_dim, activation=activation)(encoded)
    for current_dim in range(3, n_layers + 1):
        decoded = Dense(current_dim*latent_dim, activation=activation)(decoded)
    decoded = Dense(2807, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    
    return autoencoder
    
# encoded = Dense(512, activation='elu')(input_layer)
# encoded = Dense(256, activation='elu')(encoded)
# encoded = Dense(128, activation='elu')(encoded)

# decoded = Dense(128, activation='elu')(encoded)
# decoded = Dense(256, activation='elu')(decoded)
# decoded = Dense(512, activation='elu')(decoded)
# decoded = Dense(2807, activation='sigmoid')(decoded)

# autoencoder = Model(input_layer, decoded)