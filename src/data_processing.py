import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, max_error
from sklearn.linear_model import Lasso


def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=False)
    df.drop([0, 1], inplace=True)
    df = df.ffill()
    df.dropna(inplace=True)
    # df = df.iloc[::5, :]
    return df


def preprocess_data(df, interval, reference_hour=20):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
                # Log or handle columns that cannot be converted to float

    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert the 'timestamp' column to datetime
    df['date'] = df['timestamp'].dt.date
    ########################################################################################
    df = add_missing_values(df)
    df = add_missing_values(df)
    df = add_missing_values(df)
    ########################################################################################
    df['diffX'], df['diffY'] = df.iloc[:, 3] - df.iloc[:, 1], df.iloc[:, 4] - df.iloc[:, 2]
    if interval != 0:
        df = add_reference_column_at_periodic_interval_optimized(df, 'diffX', 'refX',
                                                                 interval, reference_hour)
        df = add_reference_column_at_periodic_interval_optimized(df, 'diffY', 'refY',
                                                                 interval, reference_hour)

        keywords = ['time', 'date', 'centroid', 'diff', 'ref']
        lst = [col for col in df.columns if all(keyword not in col for keyword in keywords)]

        for c in lst:
            df = add_reference_column_at_periodic_interval_optimized(df, c, f'ref_{c}',
                                                                     interval, reference_hour)
            df[c] = df[c] - df[f'ref_{c}']

    return df


def add_missing_values(df):
    df['diff'] = df['timestamp'].diff()
    mask = (df['diff'] > pd.Timedelta(minutes=1)) & (df['diff'] < pd.Timedelta(minutes=5))
    # Filter rows with gaps and calculate mid points
    new_rows = []
    for i in df[mask].index:
        current_row = df.loc[i]
        mid_timestamp = current_row['timestamp'] - pd.Timedelta(minutes=1)
        new_row = current_row.copy()
        new_row['timestamp'] = mid_timestamp
        new_rows.append(new_row)
    # Create a DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)
    # Append the new rows to the original DataFrame
    df = pd.concat([df, new_df], ignore_index=True)
    # Drop the 'diff' column and sort again
    df.drop(columns='diff', inplace=True)
    df.sort_values('timestamp', inplace=True)
    # Reset index for clean DataFrame
    df.reset_index(drop=True, inplace=True)
    return df


def smooth_data(df, window=1):
    # Ensure we're working with a copy to avoid modifying the original dataframe
    new_df = pd.DataFrame()

    # Process numeric columns for smoothing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        # Apply smoothing or copy directly based on the window size
        # if column_name is strat with 'smoothed_', it will be skipped
        if column.startswith('smoothed_'):
            column_name = column
        else:
            column_name = f'smoothed_{column}'
        if window > 1:
            new_df[column_name] = df[column].rolling(window=window, min_periods=1).mean()
            # remove the rows are the first window-1 rows in each day
            new_df[column_name] = new_df[column_name].mask(df['timestamp'].dt.hour < window - 1)
        else:
            new_df[column_name] = df[column]

    # add date and timestamp columns
    new_df[['date', 'timestamp']] = df[['date', 'timestamp']]
    new_df.dropna(inplace=True)
    return new_df


def split_train_test(df, test_date_str, train_date_start=None, train_date_end=None):
    start_timestamp = pd.to_datetime(test_date_str).normalize() + pd.Timedelta(hours=18)
    end_timestamp = start_timestamp + pd.Timedelta(days=1)

    # Split the data based on the start and end timestamps
    condition = (start_timestamp < df['timestamp']) & (df['timestamp'] <= end_timestamp)
    train_df, test_df = df[~condition].copy(), df[condition].copy()
    if train_date_start and train_date_end:
        train_df = filter_train_data_by_date(train_df, train_date_start, train_date_end)

    return train_df, test_df


def filter_train_data_by_date(df, start_date, end_date):
    train_date_start = pd.to_datetime(start_date).normalize() + pd.Timedelta(hours=18)
    train_date_end = pd.to_datetime(end_date).normalize() + pd.Timedelta(hours=18)

    train_df = df[(train_date_start <= df['timestamp']) & (df['timestamp'] <= train_date_end)]

    return train_df


def get_column_list(df):
    col_list = df.columns.tolist()
    keywords = ['time', 'centroid', 'diff']
    col_list = [col for col in col_list if all(keyword not in col for keyword in keywords)]
    items_to_exclude = ['timestamp', 'Reference Mirror X-Motion', 'Reference Mirror Y-Motion',
                        'Motorized Mirror X-Motion', 'Motorized Mirror Y-Motion',
                        'Differential X-Motion', 'Differential Y-Motion',
                        'diffX', 'diffY', 'date']
    col_list = [item.replace('smoothed_', '') for item in col_list
                if item.replace('smoothed_', '') not in items_to_exclude]

    return col_list


def filter_dataframe_by_hours(df, start_hour, end_hour):
    if start_hour < end_hour:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) & (df['timestamp'].dt.hour < end_hour)]
    else:
        df_filtered = df[(df['timestamp'].dt.hour >= start_hour) | (df['timestamp'].dt.hour < end_hour)]
    return df_filtered


# this function add a reference column to the dataframe indicating the value of each column repeating every n minutes
def add_reference_column_at_periodic_interval_optimized(df, col_name, new_col_name, interval, reference_hour):
    # Copy the DataFrame to avoid modifying the original
    new_df = df.copy()

    # Convert interval to hours if greater than 60 minutes, otherwise work with minutes
    if interval > 60:
        # Calculate the condition based on hours for intervals longer than 60 minutes
        condition = (((new_df['timestamp'].dt.hour % (interval // 60)) == reference_hour % (interval // 60))
                     & (new_df['timestamp'].dt.minute == 0))
    else:
        # Calculate the condition based on minutes for intervals up to 60 minutes
        condition = (new_df['timestamp'].dt.minute % interval) == 0

    # Use condition to directly set values in the new column and forward fill to propagate non-NA values
    new_df[new_col_name] = pd.NA
    new_df.loc[condition, new_col_name] = new_df.loc[condition, col_name]
    new_df[new_col_name].ffill(inplace=True)

    # Optionally, filter out rows where the new column is NA (if any exist after forward fill)
    new_df.dropna(subset=[new_col_name], inplace=True)

    # Ensure the new column is of float type
    new_df[new_col_name] = new_df[new_col_name].astype(float)

    return new_df


def fit_and_predict_training_data(model_type, toggle_value, training_df, col, options, alpha=0.5):
    coef, intercept = 0, 0
    col_ref = col.replace('diff', 'ref')
    # if no option is selected, the reference value will be used as the prediction
    if len(toggle_value) == 0:
        y = training_df[col]
        y_pred = training_df[col_ref]
        rmse, max_err = calc_rmse_maxerr(y, y_pred)
        return 'ref', y_pred, rmse, max_err, 0, 0

    reshape = False

    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Build column names list for X based on toggle_value, avoiding specific ref columns
    col_names = [f'smoothed_{value_to_label[val]}' for val in toggle_value if
                 f'smoothed_{value_to_label[val]}' not in ['smoothed_refX', 'smoothed_refY']] + [col]

    # Extract X and y from test_df using col_names
    X, y = training_df[col_names[:-1]].values, training_df[col].values.reshape(-1, 1)
    y_ref = training_df[col_ref].values.reshape(-1, 1)
    y = np.array(y) - np.array(y_ref)

    # Select the model ["LR", "KNN", "XGB", "RNN"]
    if model_type == "LR":
        model = LinearRegression(fit_intercept=False)

    elif model_type == "Lasso":
        model = Lasso(alpha=10 ** alpha, fit_intercept=False)
        # model = make_pipeline(StandardScaler(), Lasso(alpha=10 ** alpha, fit_intercept=False))

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

    # Fit the model
    if reshape:
        model.fit(X_reshaped, y, epochs=10, batch_size=32)
    if model_type in ["SVM", "RF"]:
        model.fit(X, y.ravel())
    else:
        model.fit(X, y)

    # Predict the values of y (target variable)
    y_pred = np.array(model.predict(X)).reshape(-1, 1)
    rmse, max_err = calc_rmse_maxerr(y, y_pred)
    y_pred += np.array(y_ref)
    y_pred = list(y_pred.ravel())

    if model_type in ["LR", "Lasso"] and len(toggle_value) > 0:
        coef, intercept = model.coef_, model.intercept_

    # if model_type == "Lasso":
    #     print(model.named_steps)
    #     model0 = model.named_steps['lasso']
    #     print(model0.coef_)
    #     coef, intercept = model0.coef_, model0.intercept_

    return model, y_pred, rmse, max_err, coef, intercept


def predict_test_data(model, toggle_value, test_df, col, options):
    col_ref = col.replace('diff', 'ref')
    if model == 'ref':
        y = test_df[col].values
        y_pred = test_df[col_ref].values
        rmse, max_err = calc_rmse_maxerr(y, y_pred)
        return y_pred, rmse, max_err

    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Build column names list for X based on toggle_value, avoiding specific ref columns
    col_names = [f'smoothed_{value_to_label[val]}' for val in toggle_value if
                 f'smoothed_{value_to_label[val]}' not in ['smoothed_refX', 'smoothed_refY']] + [col]

    # Extract X and y from test_df using col_names
    X, y = test_df[col_names[:-1]].values, test_df[col].values.reshape(-1, 1)
    y_ref = test_df[col_ref].values.reshape(-1, 1)
    y = np.array(y) - np.array(y_ref)
    # Predict the values of y (target variable)
    y_pred = np.array(model.predict(X)).reshape(-1, 1)
    rmse, max_err = calc_rmse_maxerr(y, y_pred)
    y_pred += np.array(y_ref)
    y_pred = list(y_pred.ravel())

    return y_pred, rmse, max_err


# def find_best_inputs(desired_num_features, toggle_value, training_df, col, options):
#     col_ref = col.replace('diff', 'ref')
#
#     # Create a mapping dictionary from value to label
#     value_to_label = {option['value']: option['label'] for option in options}
#
#     # Build column names list for X based on toggle_value, avoiding specific ref columns
#     col_names = [f'smoothed_{value_to_label[val]}' for val in toggle_value if
#                  f'smoothed_{value_to_label[val]}' not in ['smoothed_refX', 'smoothed_refY']] + [col]
#
#     # Extract X and y from test_df using col_names
#     X, y = training_df[col_names[:-1]].values, training_df[col].values.reshape(-1, 1)
#     y_ref = training_df[col_ref].values.reshape(-1, 1)
#     y = np.array(y) - np.array(y_ref)
#
#     alpha = 0.01
#     lasso_model = Lasso(alpha=alpha, fit_intercept=False)
#     lasso_model.fit(X, y)
#     coefficients = lasso_model.coef_
#     num_non_zero_coeffs = np.count_nonzero(coefficients)
#     while num_non_zero_coeffs > desired_num_features and alpha < 1.0:
#         alpha += 0.01
#         lasso_model = Lasso(alpha=alpha, fit_intercept=False)
#         lasso_model.fit(X, y)
#         coefficients = lasso_model.coef_
#         num_non_zero_coeffs = np.count_nonzero(coefficients)
#     for c, t in zip(coefficients, col_names[:-1]):
#         if c != 0:
#             print(f'{t}: {c}')


def calc_rmse_maxerr(y, y_pred):
    # Calculate RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Calculate the maximum error
    max_err = max_error(y, y_pred)

    return rmse, max_err
