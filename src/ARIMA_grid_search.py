import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Constants
FILE_PATH_TRAIN = '../data/train_data.csv'
TARGET_COLUMN_Y = 'smoothed_difference_2_4'
INTERVAL = 1 / 6  # 10 minutes

# Load and preprocess data
df = pd.read_csv(FILE_PATH_TRAIN)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split the dataset
split_point = int(len(df) * 0.762)  # Example split percentage
train_before_resample, test_before_resample = df[0:split_point], df[split_point:len(df)]
train_before_resample.rename(columns={TARGET_COLUMN_Y: 'Value'}, inplace=True)
test_before_resample.rename(columns={TARGET_COLUMN_Y: 'Value'}, inplace=True)


# Define the functions resample_data and redo_resample here as you provided them
def resample_data(df, column_name, hours):
    """
    Adds a reference column to the dataframe indicating the first value of a specified column
    at the start of each hour interval defined by n_hours.
    """
    seconds_per_n_hours = hours * 3600
    df['time'] = pd.to_datetime(
        ((df['timestamp'].astype('int64') // 1e9 // seconds_per_n_hours) * seconds_per_n_hours).astype(
            'int64') * 1e9)
    df_first_value_each_hour = df.groupby('time')[column_name].first().reset_index()
    new_df = df_first_value_each_hour[['time', column_name]]
    return new_df
# Grid Search Function
def grid_search_arima(train, test, p_values, d_values, q_values, window_sizes):
    best_score, best_cfg = float("inf"), None
    for p, d, q, window_size in itertools.product(p_values, d_values, q_values, window_sizes):
        try:
            # Prepare history
            history = train['Value'].tolist()
            history = history[-window_size:]  # Use the last window_size values

            # Predictions list
            predict_col = []

            for _, row in test.iterrows():
                model = ARIMA(history, order=(p, d, q))
                model_fit = model.fit()
                output = model_fit.forecast()
                obs = row['Value']
                predict_col.append(output[0])
                history[:-1] = history[1:]  # Shift everything one position to the left
                history[-1] = obs  # Append the new observation

            # Evaluate predictions
            test['pred'] = predict_col
            std_pred = np.std(test['Value'] - test['pred'])
            if std_pred < best_score:
                best_score, best_cfg = std_pred, (p, d, q, window_size)
            print(f'Tested ARIMA({p},{d},{q}) with window {window_size}: STD={std_pred:.5f}')
        except Exception as e:
            print(f"Failed to evaluate ARIMA({p},{d},{q}) with window {window_size}: {str(e)}")
            continue

    print(f'Best ARIMA{best_cfg} with STD={best_score:.5f}')
    return best_cfg


# Define your p, d, q ranges and window sizes to test
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
window_sizes = [100, 200, 300]

# Resample data
train = resample_data(train_before_resample, 'Value', INTERVAL)
test = resample_data(test_before_resample, 'Value', INTERVAL)

# Run Grid Search
best_cfg = grid_search_arima(train, test, p_values, d_values, q_values, window_sizes)
