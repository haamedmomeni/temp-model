import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error
import xgboost as xg

import matplotlib.pyplot as plt
from xgboost import plot_tree
import matplotlib
matplotlib.use('agg')


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop([0, 1], inplace=True)
    df.dropna(inplace=True)
    return df


def preprocess_data(df):
    # First, ensure the DataFrame is sorted by timestamp.
    df.sort_values('timestamp', inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
                # print(f"Cannot convert column {col} to float.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    df['difference_1_3'] = df.iloc[:, 3] - df.iloc[:, 1]
    df['difference_2_4'] = df.iloc[:, 4] - df.iloc[:, 2]

    df = add_reference_columns(df, 18, 'difference_1_3', 'refX')  # For 'difference_1_3' at 6 PM
    df = add_reference_columns(df, 18, 'difference_2_4', 'refY')  # For 'difference_2_4' at 6 PM

    # Drop intermediate columns if necessary
    df.drop(columns=['time'], inplace=True)

    # # Convert 'timestamp' column to datetime, ensuring it matches the format with milliseconds
    # df['timestamp2'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')
    #
    # # It might be useful to set the timestamp column as the index if you're planning on using time-based shifting
    # df.set_index('timestamp2', inplace=True)
    #
    # # Shift without using freq since we're directly manipulating the DataFrame index
    # for col in df.columns:
    #     if df[col].dtype != 'object':
    #         new_col_name = f"{col}_1hr_ago"
    #         # Shift the series by the equivalent of 1 hour in your DataFrame's frequency
    #         df[new_col_name] = df[col].shift(periods=1)  # Adjust periods based on your DataFrame's actual frequency
    #
    # # Reset index if you want to keep 'timestamp' as a column instead of index
    # df.reset_index(inplace=True)
    # # Drop intermediate columns if necessary
    # df.drop(columns=['timestamp2'], inplace=True)

    return df


def smooth_data(df, window=5):
    for column in df.select_dtypes(include=[np.number]).columns:
        df[f'smoothed_{column}'] = df[column].rolling(window=window).mean()
    return df


def split_train_test(df):
    unique_dates = np.sort(df['date'].unique())
    last_date = unique_dates[-1]
    day_before_last = unique_dates[-2]

    # Define the cutoff timestamp: 6 PM of the day before the last date
    cutoff_timestamp = pd.Timestamp(year=day_before_last.year, month=day_before_last.month,
                                    day=day_before_last.day, hour=18, minute=0, second=0)

    # Split the data based on the cutoff timestamp
    train_df = df[df['timestamp'] <= cutoff_timestamp].copy()
    test_df = df[df['timestamp'] > cutoff_timestamp].copy()

    return train_df, test_df


def get_column_list(df):
    """
    Return a list of column names from the DataFrame,
    skipping the first seven and the last one.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        col_list (list): The list of relevant column names.
    """
    col_list = df.columns.tolist()
    items_to_remove = ['timestamp', 'Reference Mirror X-Motion', 'Reference Mirror Y-Motion', 'Motorized Mirror X-Motion',
                       'Motorized Mirror Y-Motion', 'Differential X-Motion', 'Differential Y-Motion',
                       'difference_1_3', 'difference_2_4', 'date']
    col_list = [item for item in col_list if item not in items_to_remove]
    return col_list


def filter_dataframe_by_hours(df, start_hour, end_hour):
    if start_hour < end_hour:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) & (df['timestamp'].dt.hour < end_hour)]
    else:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) | (df['timestamp'].dt.hour < end_hour)]
    return df_filtered


def add_reference_columns(df, hour, col_name, new_col_name):
    # Ensure the 'date' and 'time' columns are present
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    # Convert hour to a time object
    reference_time = pd.to_datetime(f'{hour:02d}:00:00').time()

    # Filter for records at or after the specified hour
    df_after_hour = df[df['time'] >= reference_time]

    # Get the first record for each date at or after the specified hour
    df_first_after_hour = df_after_hour.groupby('date').first().reset_index()

    # Keep only the relevant column and rename it for merging
    df_first_after_hour = df_first_after_hour[['date', col_name]]
    df_first_after_hour.rename(columns={col_name: f'{col_name}_at_6'}, inplace=True)
    df_first_after_hour[f'{col_name}_at_6_yesterday'] = df_first_after_hour[f'{col_name}_at_6'].shift(1)

    # Merge the new column with the original DataFrame
    new_df = pd.merge(df, df_first_after_hour, on='date', how='left')
    # If the column is still vacant, fill it with the last available value in that column
    new_df[f'{col_name}_at_6'] = new_df[f'{col_name}_at_6'].ffill()
    new_df[f'{col_name}_at_6_yesterday'] = new_df[f'{col_name}_at_6_yesterday'].ffill()

    # Determine which reference value to use for each row
    new_df[new_col_name] = new_df.apply(
        lambda row: row[f'{col_name}_at_6_yesterday'] if row['time'] < reference_time else row[f'{col_name}_at_6'], axis=1
    )

    # Drop intermediate columns if necessary
    new_df.drop(columns=[f'{col_name}_at_6', f'{col_name}_at_6_yesterday'], inplace=True)

    return new_df


def fit_and_predict_linear_eq(toggle_value, training_df, col, options):
    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Precompute the column names you'll need, including the target column to ensure alignment
    col_names = [f'smoothed_{value_to_label[value]}' for value in toggle_value] + [col]

    # Select the relevant columns from 'training_df' and drop rows with any NaN values to ensure alignment
    df_selected = training_df[col_names].dropna()

    # Separate X and y after ensuring they are aligned
    X = df_selected[col_names[:-1]].to_numpy()
    y = df_selected[col].values.reshape(-1, 1)

    # Select the model
    mode_type = "XG"  # "LR"
    if mode_type == "XG":
        model = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)
    else:
        model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # if mode_type == "XG":
    #     # Plot the fifth tree
    #     fig, ax = plt.subplots(figsize=(15, 15))
    #     xg.plot_tree(model, num_trees=4, ax=ax)
    #     plt.savefig('foo.png', dpi=300)

    # Predict the values of y (target variable)
    y_pred = model.predict(X).ravel()  # Flatten the array

    # Calculate RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Calculate the maximum error
    max_err = max_error(y, y_pred)

    if mode_type == "XG":
        return y_pred, rmse, max_err, 0, 0
    else:
        return y_pred, rmse, max_err, model.coef_, model.intercept_
