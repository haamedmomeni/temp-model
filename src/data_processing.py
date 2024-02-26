import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, max_error


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop([0, 1], inplace=True)
    df.dropna(inplace=True)
    df = df.iloc[::5, :]
    return df


def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
                # Log or handle columns that cannot be converted to float
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert the 'timestamp' column to datetime

    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    df['difference_1_3'] = df.iloc[:, 3] - df.iloc[:, 1]
    df['difference_2_4'] = df.iloc[:, 4] - df.iloc[:, 2]
    # drop rows that the date is January 26th
    df = df[df['date'] != pd.to_datetime('2024-01-26').date()]
    # df = add_reference_column_at_hour_start(df, 'difference_1_3', 'refX')  # For 'difference_1_3' at 6 PM
    # df = add_reference_column_at_hour_start(df, 'difference_2_4', 'refY')  # For 'difference_2_4' at 6 PM
    df = add_reference_columns(df, 18, 'difference_1_3', 'refX')  # For 'difference_1_3' at 6 PM
    df = add_reference_columns(df, 18, 'difference_2_4', 'refY')  # For 'difference_2_4' at 6 PM

    return df


def smooth_data(df, window=1):
    for column in df.select_dtypes(include=[np.number]).columns:
        df[f'smoothed_{column}'] = df[column].rolling(window=window).mean()

    # Keep only columns that start with 'smoothed_'
    # But exclude 'date', 'time', and 'timestamp' columns
    # This includes creating a list of columns to drop that don't start with 'smoothed_'
    cols_to_drop = [col for col in df.columns if not col.startswith('smoothed_')
                    and col not in ['date', 'time', 'timestamp']]
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df


def split_train_test(df, test_date_str):
    # Convert test_date_str to a datetime.date object
    test_date = pd.to_datetime(test_date_str).date()

    # Calculate the start timestamp (midnight at the beginning of test_date)
    start_timestamp = pd.Timestamp(test_date) + pd.Timedelta(days=0, hours=18)

    # Calculate the end timestamp (6 PM the day after test_date)
    end_timestamp = pd.Timestamp(test_date) + pd.Timedelta(days=1, hours=18)

    # Split the data based on the start and end timestamps
    # Train data: before the test period
    train_df = df[(df['timestamp'] <= start_timestamp) | (df['timestamp'] >= end_timestamp)].copy()
    # train_df = df[(df['timestamp'] <= start_timestamp)].copy()

    # Test data: within the test period
    test_df = df[(df['timestamp'] > start_timestamp) & (df['timestamp'] < end_timestamp)].copy()
    # test_df = df[(df['timestamp'] > start_timestamp)].copy()

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
    items_to_remove = ['timestamp', 'Reference Mirror X-Motion', 'Reference Mirror Y-Motion',
                       'Motorized Mirror X-Motion', 'Motorized Mirror Y-Motion',
                       'Differential X-Motion', 'Differential Y-Motion',
                       'difference_1_3', 'difference_2_4', 'date', 'time']
    # col_list = [item for item in col_list if item not in items_to_remove] # and not item.startswith('smoothed')]
    col_list = [item.replace('smoothed_', '') for item in col_list
                if item.replace('smoothed_', '') not in items_to_remove]

    return col_list


def filter_dataframe_by_hours(df, start_hour, end_hour):
    if start_hour < end_hour:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) & (df['timestamp'].dt.hour < end_hour)]
    else:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) | (df['timestamp'].dt.hour < end_hour)]
    return df_filtered


def add_reference_column_at_hour_start(df, col_name, new_col_name):
    n_hours = 24
    # Calculate the total number of seconds since a fixed point (e.g., 1970-01-01)
    # Then divide by the number of seconds in n hours, floor the result, and multiply back
    seconds_per_n_hours = n_hours * 3600
    df['hour_start'] = pd.to_datetime(
        ((df['timestamp'].astype('int64') // 1e9 // seconds_per_n_hours) * seconds_per_n_hours).astype('int64') * 1e9)

    # Create a dataframe with the first value of the specified column at the start of each hour
    df_first_value_each_hour = df.groupby('hour_start')[col_name].first().reset_index()
    df_first_value_each_hour.rename(columns={col_name: new_col_name}, inplace=True)
    # Merge this dataframe back to the original dataframe on 'hour_start'
    new_df = pd.merge(df, df_first_value_each_hour, on='hour_start', how='left')
    # Drop the 'hour_start' column if it's no longer needed
    new_df.drop(columns=['hour_start'], inplace=True)
    return new_df


def add_reference_columns(df, hour, col_name, new_col_name):

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
    df_first_after_hour.iloc[-1, 1:2] = df_first_after_hour.iloc[-2, 1:2]

    # Merge the new column with the original DataFrame
    new_df = pd.merge(df, df_first_after_hour, on='date', how='left')
    # If the column is still vacant, fill it with the last available value in that column
    new_df[f'{col_name}_at_6'] = new_df[f'{col_name}_at_6'].ffill()
    new_df[f'{col_name}_at_6_yesterday'] = new_df[f'{col_name}_at_6_yesterday'].ffill()

    # Determine which reference value to use for each row
    new_df[new_col_name] = new_df.apply(
        lambda row: row[f'{col_name}_at_6_yesterday']
        if row['time'] < reference_time else row[f'{col_name}_at_6'], axis=1
    )

    # Drop intermediate columns if necessary
    new_df.drop(columns=[f'{col_name}_at_6', f'{col_name}_at_6_yesterday'], inplace=True)

    return new_df


def fit_and_predict_training_data(model_type, toggle_value, training_df, col, options):
    reshape = False
    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Precompute the column names you'll need, including the target column to ensure alignment
    col_names = [f'smoothed_{value_to_label[value]}' for value in toggle_value] + [col]

    # Select the relevant columns from 'training_df' and drop rows with any NaN values to ensure alignment
    df_selected = training_df[col_names].dropna()

    # Separate X and y after ensuring they are aligned
    X = df_selected[col_names[:-1]].to_numpy()
    y = df_selected[col].values.reshape(-1, 1)

    # Select the model ["LR", "KNN", "XGB", "RNN"]
    if model_type == "LR":
        model = LinearRegression()

    elif model_type == "KNN":
        model = KNeighborsRegressor(n_neighbors=1)

    elif model_type == "XGB":
        model = xg.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,  # Increased number of trees
            # learning_rate=0.1,  # Lower learning rate
            max_depth=3,  # Limiting tree depth
            # min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            seed=123
        )

    elif model_type == "RF":
        model = RandomForestRegressor(
            n_estimators=100,  # Total number of trees to generate (can be adjusted)
            max_depth=5,  # Adjust this as necessary to suit your data/problem
            random_state=42  # For reproducibility
            # , min_samples_split=2,  # Minimum samples needed to split a node
            # , min_samples_leaf=1,  # Minimum samples needed to be at a leaf node
            # , bootstrap=True  # Method of re-sampling for each new tree
            # , oob_score=False  # If out-of-bag score to use
            # , n_jobs=-1  # Number of cores to be used in parallel, -1 will use all available
        )

    # elif model_type == "RNN":
    #     n_features = X.shape[1]
    #     reshape = True
    #     X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
    #     model = Sequential([
    #         SimpleRNN(20, activation='relu', input_shape=(5, n_features)),  # 20 units, input shape = (time steps, features)
    #         Dense(1)  # Output layer with 1 unit for regression task
    #     ])
    #
    #     # Compile the model
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #
    #     # Print model summary
    #     model.summary()

    elif model_type == "SVM":
        model = SVR(
            kernel='rbf',  # Radial Basis Function kernel; consider 'linear', 'poly', for others.
            C=0.01,
            # Regularization parameter: tradeoff between smooth decision boundary
            # and classifying all training points correctly
            epsilon=0.1,  # Specifies the epsilon-tube within which no penalty is associated in the training loss
            gamma='scale'
            # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If 'gamma'='scale', 1/(n_features * X.var())
        )

    if reshape:
        # Train the model
        model.fit(X_reshaped, y, epochs=10, batch_size=32)
    if model_type in ["SVM", "RF"]:
        model.fit(X, y.ravel())
    else:
        # Fit the model
        model.fit(X, y)

    # Predict the values of y (target variable)
    y_pred = model.predict(X).ravel()  # Flatten the array

    # Calculate RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Calculate the maximum error
    max_err = max_error(y, y_pred)

    if model_type == "LR":
        return model, y_pred, rmse, max_err, model.coef_, model.intercept_
    else:
        return model, y_pred, rmse, max_err, 0, 0


def predict_test_data(model, toggle_value, test_df, col, options):
    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Precompute the column names you'll need, including the target column to ensure alignment
    col_names = [f'smoothed_{value_to_label[value]}' for value in toggle_value] + [col]

    # Select the relevant columns from 'test_df' and drop rows with any NaN values to ensure alignment
    df_selected = test_df[col_names]  # .dropna()

    # Separate X and y after ensuring they are aligned
    X = df_selected[col_names[:-1]].to_numpy()
    y = df_selected[col].values.reshape(-1, 1)

    # Predict the values of y (target variable)
    y_pred = model.predict(X).ravel()  # Flatten the array

    # y_pred = rolling_mean_with_padding(y_pred, 50)

    # Calculate RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Calculate the maximum error
    max_err = max_error(y, y_pred)

    return y_pred, rmse, max_err

def rolling_mean_with_padding(arr, window):
    """Calculate the rolling mean of a numpy array, with padding."""
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    rolling_mean = ret[window - 1:] / window
    # Pad with NaNs to keep the original array size
    padding = np.empty(window-1)
    padding.fill(np.nan)
    return np.concatenate((padding, rolling_mean))
