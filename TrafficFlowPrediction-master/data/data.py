import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def split_data(scats_number, location, test_ratio):
    """
    Split the data file with the specified SCATS number and location into train and test CSV files.

    Parameters:
    scats_number (str): SCATS number identifying the data file.
    location (str): Location information for file identification.
    test_ratio (float): Proportion of data to allocate to the test set.

    This function reads the specified CSV file, formats the 'SCATS Number' as a 4-digit string,
    splits the data into training and testing sets based on the provided test_ratio, and
    saves these sets as separate CSV files.
    """
    # Load data from the specified CSV file for the given SCATS number and location
    df = pd.read_csv(f'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/TRAFFIC_FLOW/{scats_number}_{location}.csv')
    
    # Ensure the 'SCATS Number' column is formatted as a 4-digit string for consistency
    df['SCATS Number'] = df['SCATS Number'].apply(lambda x: f'{x:04d}')

    # Calculate index for splitting data based on the specified test_ratio
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index]  # Select rows for training data
    test_df = df.iloc[split_index:]   # Select rows for testing data

    # Save the training and testing data as separate CSV files
    train_df.to_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/train.csv', index=False)
    test_df.to_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/test.csv', index=False)

def process_data(train, test, lags):
    """
    Process and prepare data for model training and testing.

    Parameters:
    train (str): Path to the training data CSV file.
    test (str): Path to the testing data CSV file.
    lags (int): Number of time lags to include for prediction.

    Returns:
    X_train, y_train, X_test, y_test, scaler: Arrays of features and labels for training and testing, and scaler instance.
    
    This function loads and scales the 'Lane 1 Flow (Veh/15 Minutes)' data, reshapes it based on
    time lags for temporal sequence analysis, and splits it into training and testing sets.
    """
    # Specify the column containing traffic flow data for prediction
    attr = 'Lane 1 Flow (Veh/15 Minutes)'
    
    # Load training and testing data, and replace any missing values with zero
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # Initialize MinMaxScaler and fit it to the training data to scale 'Lane 1 Flow' values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]  # Scale and flatten training data
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]  # Scale and flatten testing data

    # Create sequences of lagged data for model training (lags are used for temporal predictions)
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])  # Append lagged sequences for training
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])   # Append lagged sequences for testing

    # Convert sequences into NumPy arrays for model compatibility
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)  # Shuffle training data for robust model training

    # Separate features (X) and labels (y) for both training and testing datasets
    X_train = train[:, :-1]  # Features for training
    y_train = train[:, -1]   # Labels for training
    X_test = test[:, :-1]    # Features for testing
    y_test = test[:, -1]     # Labels for testing

    # Return processed data and the scaler for potential inverse transformations
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Define SCATS number and location for data file identification, and the test ratio for the data split
    scats_number = '0970'
    location = 'HIGH STREET_RD W of WARRIGAL_RD'
    
    # Perform data split, allocating 80% of data for training and 20% for testing
    split_data(scats_number, location, 0.2)
