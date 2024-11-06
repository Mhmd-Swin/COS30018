import pandas as pd
import os

# Load the source traffic flow data into a DataFrame
df = pd.read_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/Scats Data October 2006.csv')

# Specify the output directory for the processed data files
output_directory = 'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/TRAFFIC_FLOW'
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize an empty list to store the transformed data
modified_data = []

# Loop through each row in the dataset
for index, row in df.iterrows():
    scats_number = row['SCATS Number']  # Extract SCATS Number
    location = row['Location']          # Extract location
    date = row['Date']                  # Extract date

    # Loop through 96 intervals, representing 15-minute blocks in a day
    for interval in range(96):
        # Append a dictionary with the transformed row to the modified_data list
        modified_data.append({
            'SCATS Number': scats_number,
            'Location': location,
            '15 Minutes': date,                         # Append date as part of the '15 Minutes' column
            'Lane 1 Flow (Veh/15 Minutes)': row[f'V{interval:02d}'],  # Vehicle flow data for each 15-minute interval
            '# Lane Points': 1,                         # Assumed to be 1 lane point
            '% Observed': 100                           # Assumed 100% data observation
        })

# Create a DataFrame from the transformed data
df_modified = pd.DataFrame(
    modified_data, 
    columns=['SCATS Number', 'Location', '15 Minutes', 'Lane 1 Flow (Veh/15 Minutes)', '# Lane Points', '% Observed']
)

# Format the SCATS Number as a 4-digit string for consistency
df_modified['SCATS Number'] = df_modified['SCATS Number'].apply(lambda x: f'{x:04d}')

# Generate time intervals for each 15-minute block in a day
time_intervals = pd.date_range(start='00:00', end='23:45', freq='15min').strftime('%H:%M').tolist()

# Append the generated time intervals to the '15 Minutes' column to form complete datetime entries
df_modified['Time'] = time_intervals * (len(df_modified) // len(time_intervals))
df_modified['15 Minutes'] = df_modified['15 Minutes'] + ' ' + df_modified['Time']

# Clean up by removing the temporary 'Time' column used to form complete timestamps
df_modified = df_modified.drop(columns=['Time'])

# Group the transformed data by 'Location'
scats_groups = df_modified.groupby('Location')

# Save each location's data to a separate CSV file
for location, group in scats_groups:
    # Construct the file name based on SCATS number and location
    file_name = f'{group["SCATS Number"].iloc[0]}_{location}.csv'
    file_path = f'C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/TRAFFIC_FLOW/{file_name}'
    # Save the grouped data to a CSV file
    group.to_csv(file_path, index=False)

# Extract relevant columns for coordinates and remove duplicate entries
coordinates = df[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()

# Manually correct the coordinates for a specific location
coordinates.loc[df['Location'] == 'AUBURN_RD N of BURWOOD_RD', ['NB_LATITUDE', 'NB_LONGITUDE']] = [-37.8251000, 145.0434710]

# Reset the index of the coordinates DataFrame for a cleaner structure
coordinates = coordinates.reset_index(drop=True)

# Format the 'SCATS Number' as a 4-digit string for consistency
coordinates['SCATS Number'] = coordinates['SCATS Number'].apply(lambda x: f'{x:04d}')

# Adjust latitude and longitude slightly to account for manual corrections
coordinates['NB_LATITUDE'] += 0.00145
coordinates['NB_LONGITUDE'] += 0.00135

# Calculate the mean coordinates for each SCATS site by grouping based on 'SCATS Number'
mean_coordinates = coordinates.groupby('SCATS Number').agg({
    'NB_LATITUDE': 'mean',
    'NB_LONGITUDE': 'mean'
}).reset_index()

# Save the original and mean coordinates to CSV files with high precision
coordinates.to_csv('Scats Location points.csv', index=False, float_format='%.7f')
mean_coordinates.to_csv('Scats Mean points.csv', index=False, float_format='%.7f')
