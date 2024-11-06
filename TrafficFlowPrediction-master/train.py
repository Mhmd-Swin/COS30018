import sys
import warnings
import pathlib
import numpy as np
import pandas as pd
import os
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, LSTM, Dense, Input
from data.data import process_data, split_data
from newfile import create_scats_location_map

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define Simple RNN model
def build_simple_rnn(units):
    model = Sequential()
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(SimpleRNN(units[2]))
    model.add(Dense(units[3]))
    return model

# Define LSTM model
def build_lstm(units):
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dense(units[3]))
    return model

# Define Stacked Autoencoder (SAES-EXT) model
def build_stacked_autoencoder(input_dim, hidden_dim, output_dim, num_layers, bottleneck_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(hidden_dim, activation='relu')(input_layer)
    for _ in range(num_layers - 1):
        encoded = Dense(hidden_dim, activation='relu')(encoded)
    bottleneck = Dense(bottleneck_dim, activation='relu')(encoded)
    decoded = bottleneck
    for _ in range(num_layers - 1):
        decoded = Dense(hidden_dim, activation='relu')(decoded)
    decoded = Dense(output_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

def train_single_model(nn_model, X_train, y_train, model_name, scats_id, location, config):
    nn_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    training_history = nn_model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05
    )

    # Define model and loss paths according to specified structure
    model_path = pathlib.Path(f"C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/model/CUSTOM_MODELS/{model_name}/{scats_id}_{location}.h5")
    loss_path = pathlib.Path(f"C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/model/CUSTOM_MODELS/{model_name}/{scats_id}_{location}_loss.csv")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model and training history
    nn_model.save(model_path)
    pd.DataFrame.from_dict(training_history.history).to_csv(loss_path, encoding='utf-8', index=False)

def model_training_pipeline(model_type, scats_id, location):
    lag_time = 12
    training_config = {"batch": 256, "epochs": 600}
    X_train, y_train, _, _, _ = process_data('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/train.csv', 'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/test.csv', lag_time)

    if model_type == 'srnn':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        nn_model = build_simple_rnn([12, 64, 64, 1])
    elif model_type == 'lstm':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        nn_model = build_lstm([12, 64, 64, 1])
    elif model_type == 'saes-extended':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        nn_model = build_stacked_autoencoder(12, 400, 1, 3, 1)

    train_single_model(nn_model, X_train, y_train, model_type, scats_id, location, training_config)

def automated_model_training(start, end, model_name):
    scats_dict = create_scats_location_map()

    for scats_id in scats_dict:
        if int(scats_id) in range(start, end + 1):
            for location in scats_dict[scats_id]:
                model_path = pathlib.Path(f"C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/model/CUSTOM_MODELS/{model_name}/{scats_id}_{location}.h5")
                loss_path = pathlib.Path(f"C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/model/CUSTOM_MODELS/{model_name}/{scats_id}_{location}_loss.csv")

                # Check if the model and loss file already exist to skip retraining
                if not model_path.exists() or not loss_path.exists():
                    print(f"\nTraining '{model_name}' model for SCATS '{scats_id}' at location '{location}'")

                    split_data(scats_id, location, test_ratio=0.2)
                    model_training_pipeline(model_name, scats_id, location)

                    # Remove temporary training and testing data files
                    for file_type in ("train", "test"):
                        try:
                            os.remove(f"data/{file_type}.csv")
                        except FileNotFoundError:
                            pass
                else:
                    print(f"Skipping '{model_name}' model for SCATS '{scats_id}' at location '{location}' - Model already exists.")

if __name__ == "__main__":
    automated_model_training(970, 4821, "lstm")
    automated_model_training(970, 4821, "saes-extended")
    automated_model_training(970, 4821, "srnn")
