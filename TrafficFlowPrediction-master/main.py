"""
Traffic Flow Prediction with Neural Networks (SRNN, LSTM, SAE-Extended).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.api.models import load_model
from keras.api.utils import plot_model
from keras.api.losses import MeanSquaredError
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def calculate_absolute_percentage_error(actual_values, predicted_values):
    """Absolute Percentage Error Calculation
    Compute the absolute percentage error for model assessment.

    # Arguments
        actual_values: List/ndarray, real traffic values.
        predicted_values: List/ndarray, model-predicted values.
    # Returns
        ape: Float, calculated Absolute Percentage Error.
    """
    valid_actuals = [v for v in actual_values if v > 0]
    valid_preds = [predicted_values[i] for i in range(len(actual_values)) if actual_values[i] > 0]

    total_count = len(valid_preds)
    error_accumulation = sum(abs(valid_actuals[i] - valid_preds[i]) / valid_actuals[i] for i in range(total_count))

    ape = (error_accumulation * 100) / total_count
    return ape


def performance_metrics(real_values, forecasted_values):
    """Performance Evaluation
    Calculate different regression metrics to evaluate model forecasts.

    # Arguments
        real_values: List/ndarray, actual observed values.
        forecasted_values: List/ndarray, predicted values from the model.
    """
    ape = calculate_absolute_percentage_error(real_values, forecasted_values)
    explained_variance = metrics.explained_variance_score(real_values, forecasted_values)
    mean_absolute_error = metrics.mean_absolute_error(real_values, forecasted_values)
    mean_squared_error = metrics.mean_squared_error(real_values, forecasted_values)
    r_squared = metrics.r2_score(real_values, forecasted_values)
    rmse = math.sqrt(mean_squared_error)

    print(f'Explained Variance Score: {explained_variance:.4f}')
    print(f'Absolute Percentage Error: {ape:.4f}%')
    print(f'Mean Absolute Error: {mean_absolute_error:.4f}')
    print(f'Mean Squared Error: {mean_squared_error:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R-Squared Score: {r_squared:.4f}')


def plot_forecasts(real_values, model_predictions, model_names):
    """Visualize Forecasts
    Plot real values against predictions from each model.

    # Arguments
        real_values: List/ndarray, true values.
        model_predictions: List of lists/ndarray, predictions from each model.
        model_names: List of str, names of each model.
    """
    start_timestamp = '1/10/2006 00:00'
    time_slots = pd.date_range(start_timestamp, periods=96, freq='15min')

    fig, ax = plt.subplots()
    ax.plot(time_slots, real_values, label='Actual Data', color='black')
    
    for model, predictions in zip(model_names, model_predictions):
        ax.plot(time_slots, predictions, label=model)

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Traffic Flow')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.show()


def main():
    custom_objects = {'mse': MeanSquaredError()}
    models = {
        'SRNN': load_model('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/model/srnn.h5', custom_objects),
        'LSTM': load_model('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/model/lstm.h5', custom_objects),
        'SAE-Extended': load_model('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/model/saes-extended-version.h5', custom_objects)
    }

    lag_window = 12
    train_data_file = 'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/train.csv'
    test_data_file = 'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/test.csv'
    _, _, test_features, test_labels, scaler = process_data(train_data_file, test_data_file, lag_window)
    test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1)).reshape(1, -1)[0]

    model_predictions = []
    for model_name, model in models.items():
        if model_name == 'SAE-Extended':
            test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1]))
        else:
            test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

        # Attempt to plot model, handle errors if unsuccessful
        model_diagram_file = f'images/{model_name}.png'
        try:
            plot_model(model, to_file=model_diagram_file, show_shapes=True)
        except (AttributeError, FileNotFoundError) as e:
            print(f"Could not generate model plot for {model_name}: {e}")

        predictions = model.predict(test_features)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(1, -1)[0]
        model_predictions.append(predictions[:96])

        print(f"Performance for {model_name}:")
        performance_metrics(test_labels[:96], predictions[:96])

    plot_forecasts(test_labels[:96], model_predictions, list(models.keys()))


if __name__ == '__main__':
    main()
