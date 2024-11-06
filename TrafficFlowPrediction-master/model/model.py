"""
Definition of Neural Network (NN) Models
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.layers import Dense, Dropout, Activation, InputLayer
from keras.api.layers import LSTM, SimpleRNN
from keras.api.models import Sequential

def build_simple_rnn(units_list):
    """SimpleRNN (Simple Recurrent Neural Network)
    Build SimpleRNN Model.

    # Arguments
        units_list: List(int), number of input, hidden, and output units.
    # Returns
        model: Model, neural network model.
    """
    
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(units_list[1], input_shape=(units_list[0], 1), return_sequences=True))
    rnn_model.add(SimpleRNN(units_list[2]))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(units_list[3], activation='sigmoid'))
    return rnn_model

def build_lstm(units_list):
    """LSTM (Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units_list: List(int), number of input, hidden, and output units.
    # Returns
        model: Model, neural network model.
    """

    lstm_model = Sequential()
    lstm_model.add(LSTM(units_list[1], input_shape=(units_list[0], 1), return_sequences=True))
    lstm_model.add(LSTM(units_list[2]))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units_list[3], activation='sigmoid'))

    return lstm_model


def build_extended_stacked_autoencoders(input_units, hidden_units, output_units, num_autoencoders=3, layers_per_autoencoder=1):
    """Extended Stacked Autoencoders (SAEs)
    Build Extended Stacked Autoencoders Model.

    # Arguments
        input_units: Integer, number of input units.
        hidden_units: Integer, number of hidden units.
        output_units: Integer, number of output units.
        num_autoencoders: Integer, number of autoencoders.
        layers_per_autoencoder: Integer, number of layers in the encoder and decoder of an autoencoder.
    # Returns
        model: Model, neural network model.
    """

    extended_sae_model = Sequential()
    extended_sae_model.add(InputLayer(input_shape=(input_units, )))
    units_per_layer = hidden_units // layers_per_autoencoder

    # Build and stack autoencoders
    for i in range(num_autoencoders):
        # Encoder layers
        for j in range(1, layers_per_autoencoder + 1):
            layer_size = hidden_units if j == layers_per_autoencoder else units_per_layer * j
            extended_sae_model.add(Dense(layer_size, name=f'encoder_layer_{2*layers_per_autoencoder*i + j}'))
            extended_sae_model.add(Activation('relu'))

        # Decoder layers
        for k in range(1, layers_per_autoencoder + 1):
            layer_size = max(units_per_layer * (layers_per_autoencoder - k), input_units)
            extended_sae_model.add(Dense(layer_size, name=f'decoder_layer_{2*layers_per_autoencoder*i + k + layers_per_autoencoder}'))
            extended_sae_model.add(Activation('relu'))

    extended_sae_model.add(Dropout(0.2))
    extended_sae_model.add(Dense(output_units, activation='sigmoid'))

    return extended_sae_model
