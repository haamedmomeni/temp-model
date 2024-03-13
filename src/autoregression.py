import itertools
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ProcessPoolExecutor

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.interpolate import CubicSpline, UnivariateSpline

# Ignore the ConvergenceWarning
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals", category=ConvergenceWarning)


# Constants
FILE_PATH_TRAIN = 'train_data.csv'
TARGET_COLUMN_Y = 'smoothed_difference_2_4'
INTERVAL = 1/6  # 10 minutes
# p, d, q = 0, 0, 0  # Example ARIMA model parameters
# WINDOW_SIZE = 500
p_values = [0]
d_values = [0]
q_values = [0]
window_sizes = [4]
# coeff_values = [1]
coeff_values = [0.1*i for i in range(11)]
c1_values = [0.01*i for i in range(51)]
c2_values = [0.01*i for i in range(51)]
c3_values = [0.01*i for i in range(51)]
# c1_values = [0.3]
# c2_values = [0.1]
# c3_values = [0]
min_val = 1

# Functions
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


def redo_resample(df):
    # Create a list to hold the new rows
    rows = []

    # iterate over rows of dataframe
    for i, row in df.iterrows():
        time, value, pred = row['time'], row['Value'], row['pred']
        # if i < df.index[-1]:  # Ensure we don't go out of bounds
        #     # Access the 'pred' value of the next row
        #     next_pred = df.iloc[i + 1 - df.index[0], pred_col_index]
        if i < df.index[-1]:
            pred = df['pred'].iloc[i + 1 - df.index[0] ]
        # Extend the rows list with new rows for each minute
        rows.extend([{'time': time + pd.Timedelta(minutes=i),
                      'Value': value,
                      'pred': value + (pred - value) * (i/10)} for i in range(10)])

    # Create a new DataFrame from the list of dictionaries
    new_df = pd.DataFrame(rows, columns=['time', 'Value', 'pred'])

    return new_df


# Load and preprocess data
df = pd.read_csv(FILE_PATH_TRAIN)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split the dataset
split_point = int(len(df) * 0.762)  # 80% for training, 20% for testing
train_before_resample, test_before_resample = df[0:split_point], df[split_point:len(df)]

train_before_resample.rename(columns={TARGET_COLUMN_Y: 'Value'}, inplace=True)
test_before_resample.rename(columns={TARGET_COLUMN_Y: 'Value'}, inplace=True)
TARGET_COLUMN_Y = 'Value'

train = resample_data(train_before_resample, TARGET_COLUMN_Y, INTERVAL)
test = resample_data(test_before_resample, TARGET_COLUMN_Y, INTERVAL)
pp = []
with ProcessPoolExecutor(max_workers=16) as executor:
    # for p, d, q, WINDOW_SIZE in itertools.product(p_values, d_values, q_values, window_sizes):
    for c1, c2, c3 in itertools.product(c1_values, c2_values, c3_values):
        WINDOW_SIZE = 4
        # Modeling
        history = train['Value'].tolist()
        history = history[-WINDOW_SIZE:]  # Limit the history to the last WINDOW_SIZE values
        predict_col = []

        # iterate over rows of dataframe
        for _, row in test.iterrows():
            # model = ARIMA(history, order=(p, d, q))
            # model_fit = model.fit()
            # output = model_fit.forecast()
            time, obs = row['time'], row['Value']
            # print(history[-2:], history[-1] + (history[-1] - history[-2]), output[0])
            if time.hour == 20 and time.minute == 10 and time.second == 0\
                    or time.hour == 20 and time.minute == 20 and time.second == 0\
                    or time.hour == 20 and time.minute == 30 and time.second == 0\
                    or time.hour == 5 and time.minute == 40 and time.second == 0\
                    or time.hour == 5 and time.minute == 50 and time.second == 0:
                predict_col.append(obs)
            else:
                # x = np.array([0, 1, 2, 3])
                # y = np.array(history[-4:])
                # print(x, y)
                # spline = CubicSpline(x, y)
                # spline = UnivariateSpline(x, y, s=0.1)
                # x_next = 4
                # y_next = spline(x_next)

                # model = LinearRegression()
                # model.fit(x.reshape(-1, 1), y)
                # y_next = model.predict(np.array([[4]]))
                # predict_col.append(y_next)

                # x_next = []
                # for i in range(10):
                #     if time + pd.Timedelta(minutes=i) in test_before_resample['timestamp'].values:
                #         # print('found')
                #         x_next.append(4+i)
                #
                # x_next_array = np.array(x_next).reshape(-1, 1)
                # y_next = model.predict(x_next_array)
                # y_next = spline(x_next)
                # pp.extend(y_next)
                predict_col.append(history[-1] + c1 * (history[-1] - history[-2])
                                               + c2 * (history[-1] - history[-2])
                                               + c3 * (history[-1] - history[-3]))
                # predict_col.append(history[-1] + coeff/2 * (history[-1] - history[-2])
                #                                 + coeff/2 * (history[-2] - history[-3]))
                # predict_col.append(output[0])
            # Update the array by rolling and appending the new observation
            history[:-1] = history[1:]  # Shift everything one position to the left
            history[-1] = obs  # Set the last element as the new observation

        test['pred'] = predict_col

        new_test = redo_resample(test)

        # rename column of test_before_resample from 'Value' to "original"
        test_before_resample.rename(columns={'Value': 'original'}, inplace=True)
        # Merge the DataFrames on the specified columns
        merged_df = pd.merge(left=test_before_resample[['timestamp', 'original']], right=new_test[['time', 'Value', 'pred']],
                             left_on='timestamp', right_on='time')

        std_val = np.std(merged_df['original'] - merged_df['Value'])
        std_pred = np.std(merged_df['original'] - merged_df['pred'])
        if std_pred < min_val:
            min_val = std_pred
        # print(f'for p; {p}, d: {d}, q: {q}, window: {WINDOW_SIZE} Std of ARIMA: {std_pred:.3f}')
            print(f'for c1; {c1}, c2: {c2}, c3: {c3} Std of realignment: {std_pred:.3f}')
        # print(f'Std of ARIMA: {std_pred:.3f}')

        # print(f'Std of realignment: {std_val:.3f}')

# Visualization
fig = plt.figure(figsize=(10, 4))

# plot original data
plt.scatter(merged_df['time'], merged_df['original'], color='black', label='Actual', marker='.')

# plot original data with resampling and same length
plt.scatter(merged_df['time'], merged_df['Value'], color='red', label='Realignment', marker='.')

# plot predictions
plt.scatter(merged_df['time'], merged_df['pred'], color='blue', label='ARIMA', marker='.')
n = len(pp)
print(len(pp))
print(len(merged_df['time']))

# plot predictions
# plt.scatter(merged_df['time'][:n], pp, color='green', label='ARIMA', marker='.')
plt.legend()
plt.grid(True)
plt.show()
