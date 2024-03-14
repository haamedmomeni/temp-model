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
    df = df.ffill()
    df.dropna(inplace=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    # df = df.iloc[::5, :]
    return df


def preprocess_data(df, interval):
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
    if interval != 0:
        df = add_reference_column_at_periodic_interval_optimized(df, 'difference_1_3', 'refX', interval)
        df = add_reference_column_at_periodic_interval_optimized(df, 'difference_2_4', 'refY', interval)
        df['difference_1_3'] = df['difference_1_3'] - df['refX']
        # df.drop('refX', axis=1, inplace=True)
        df['difference_2_4'] = df['difference_2_4'] - df['refY']
        # df.drop('refY', axis=1, inplace=True)

    return df


def smooth_data(df, window=1):
    new_df = df.copy()
    if window == 1:
        for column in new_df.select_dtypes(include=[np.number]).columns:
            new_df[f'smoothed_{column}'] = new_df[column]
    else:
        for column in new_df.select_dtypes(include=[np.number]).columns:
            new_df[f'smoothed_{column}'] = new_df[column].rolling(window=window).mean()
    # Keep only columns that start with 'smoothed_'
    # But exclude 'date', 'time', and 'timestamp' columns
    # This includes creating a list of columns to drop that don't start with 'smoothed_'
    cols_to_drop = [col for col in new_df.columns if not col.startswith('smoothed_')
                    and col not in ['date', 'time', 'timestamp']]
    new_df.drop(cols_to_drop, axis=1, inplace=True)

    return new_df


def split_train_test(df, test_date_str, train_date_start=None, train_date_end=None):
    # Convert test_date_str to a datetime.date object
    test_date = pd.to_datetime(test_date_str).date()

    # Calculate the start timestamp (midnight at the beginning of test_date)
    start_timestamp = pd.Timestamp(test_date) + pd.Timedelta(days=0, hours=18)

    # Calculate the end timestamp (6 PM the day after test_date)
    end_timestamp = pd.Timestamp(test_date) + pd.Timedelta(days=1, hours=18)

    # Split the data based on the start and end timestamps
    # Train data: before the test period
    train_df = df[(df['timestamp'] <= start_timestamp) | (df['timestamp'] >= end_timestamp)].copy()
    # Test data: within the test period
    test_df = df[(df['timestamp'] > start_timestamp) & (df['timestamp'] < end_timestamp)].copy()

    if train_date_start and train_date_end:
        train_date_start = (pd.Timestamp(pd.to_datetime(train_date_start).date())
                            + pd.Timedelta(days=0, hours=18))
        train_date_end = (pd.Timestamp(pd.to_datetime(train_date_end).date())
                          + pd.Timedelta(days=0, hours=18))
        # train_date_start = pd.to_datetime(train_date_start).date()
        # train_date_end = pd.to_datetime(train_date_end).date()
        train_df = train_df[(train_df['timestamp'] >= train_date_start) &
                            (train_df['timestamp'] <= train_date_end)]

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


# this function add a reference column to the dataframe indicating the value of each column repeating every n minutes
def add_reference_column_at_periodic_interval_optimized(df, col_name, new_col_name, interval):
    new_df = df.copy()

    # Calculate the interval condition
    if interval <= 60:
        condition = (df['timestamp'].dt.minute % interval) == 0
    else:
        interval = interval // 60
        condition = ((df['timestamp'].dt.minute == 0) &
                     (df['timestamp'].dt.hour % interval == 20 % interval))
    # Initialize the new column with NaNs
    new_df[new_col_name] = pd.NA
    # Set the value in the new column where the condition is true
    new_df.loc[condition, new_col_name] = new_df.loc[condition, col_name]
    # Forward fill the new column to propagate the last valid observation
    new_df[new_col_name] = new_df[new_col_name].ffill()

    new_df.dropna(subset=[new_col_name], inplace=True)
    new_df[new_col_name] = new_df[new_col_name].astype(float)

    return new_df


def fit_and_predict_training_data(model_type, toggle_value, training_df, col, options):
    # if no option is selected, the reference value will be used as the prediction
    if len(toggle_value) == 0:
        y = training_df[col].values.reshape(-1, 1)
        y_pred = np.zeros(training_df.shape[0])

        # Calculate RMSE
        rmse = mean_squared_error(y, y_pred, squared=False)

        # Calculate the maximum error
        max_err = max_error(y, y_pred)

        return 'ref', y_pred, rmse, max_err, 0, 0

    reshape = False
    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Precompute the column names you'll need, including the target column to ensure alignment
    col_names = [f'smoothed_{value_to_label[value]}' for value in toggle_value] + [col]
    if "smoothed_refX" in col_names:
        col_names.remove('smoothed_refX')
    if "smoothed_refY" in col_names:
        col_names.remove('smoothed_refY')
    # Separate X and y after ensuring they are aligned
    X = training_df[col_names[:-1]].to_numpy()
    y = training_df[col].values.reshape(-1, 1)

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
    if model == 'ref':
        y = test_df[col].values.reshape(-1, 1)
        y_pred = np.zeros(test_df.shape[0])

        # Calculate RMSE
        rmse = mean_squared_error(y, y_pred, squared=False)

        # Calculate the maximum error
        max_err = max_error(y, y_pred)

        return y_pred, rmse, max_err

    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Precompute the column names you'll need, including the target column to ensure alignment
    col_names = [f'smoothed_{value_to_label[value]}' for value in toggle_value] + [col]
    if "smoothed_refX" in col_names:
        col_names.remove('smoothed_refX')
    if "smoothed_refY" in col_names:
        col_names.remove('smoothed_refY')


    # Separate X and y after ensuring they are aligned
    X = test_df[col_names[:-1]].to_numpy()
    y = test_df[col].values.reshape(-1, 1)

    # Predict the values of y (target variable)
    y_pred = model.predict(X).ravel()  # Flatten the array

    # Calculate RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Calculate the maximum error
    max_err = max_error(y, y_pred)

    return y_pred, rmse, max_err


# def rolling_mean_with_padding(arr, window):
#     """Calculate the rolling mean of a numpy array, with padding."""
#     ret = np.cumsum(arr, dtype=float)
#     ret[window:] = ret[window:] - ret[:-window]
#     rolling_mean = ret[window - 1:] / window
#     # Pad with NaNs to keep the original array size
#     padding = np.empty(window-1)
#     padding.fill(np.nan)
#     return np.concatenate((padding, rolling_mean))
