import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, max_error
from math import sqrt

# Constants
FILE_PATH_TRAIN = 'train_data.csv'
NUM_COLORS = 8  # Adjust based on the number of conditions you have
# COLOR_PALETTE = sns.color_palette("flare", NUM_COLORS)
TARGET_COLUMN_X = 'smoothed_difference_1_3'
TARGET_COLUMN_Y = 'smoothed_difference_2_4'
INTERVAL = 1/6  # 10 minutes


def make_series(df, column_name, hours=1):
    """
    Adds a reference column to the dataframe indicating the first value of a specified column
    at the start of each hour interval defined by n_hours.
    """
    seconds_per_n_hours = hours * 3600
    df['hour_start'] = pd.to_datetime(
        ((df['timestamp'].astype('int64') // 1e9 // seconds_per_n_hours) * seconds_per_n_hours).astype(
            'int64') * 1e9)
    df_first_value_each_hour = df.groupby('hour_start')[column_name].first().reset_index()
    # df_first_value_each_hour.rename(columns={column_name: new_column_name}, inplace=True)
    new_df = df_first_value_each_hour[['hour_start', column_name]]
    # make timestamp the index
    # new_df.set_index('timestamp', inplace=True)
    # new_df = pd.merge(df, df_first_value_each_hour, on='hour_start', how='left')
    # new_df.drop(columns=['hour_start'], inplace=True)
    return new_df


# Load and preprocess data
df_train = pd.read_csv(FILE_PATH_TRAIN)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
# series = make_series(df_train, TARGET_COLUMN_X, 1/6)

# # fit model
# model = ARIMA(series, order=(5,1,0))
# model_fit = model.fit()
#
# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# # density plot of residuals
# residuals.plot(kind='kde', xlim=[-10, 10])
# pyplot.show()
# # summary stats of residuals
# print(residuals.describe())
#
# y = series
#
# model = ARIMA(y, order=(5,1,0))
# model_fit = model.fit()
# output = model_fit.forecast()
# print(output)
# yhat = output[0]
# # plot forecasts against actual outcomes
# pyplot.plot(y, color='blue')
# pyplot.plot(yhat, color='red')
# pyplot.show()
#
# ax = autocorrelation_plot(series)
# ax.set_xlim([0, 24 / INTERVAL])
# pyplot.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from math import sqrt

# Load the dataset
# This example assumes a CSV file with a 'Date' column and a 'Value' column
# Replace 'your_dataset.csv' and 'Value' with your actual dataset details
# df = pd.read_csv('your_dataset.csv', parse_dates=['Date'], index_col='Date')

# Create a time series
df = make_series(df_train, TARGET_COLUMN_Y, 1/6)
df.rename(columns={TARGET_COLUMN_Y: 'Value'}, inplace=True)

# # Visualize the dataset
# df.plot()
# plt.show()

# Split the dataset into train and test sets
split_point = int(len(df) * 0.8)  # 80% for training, 20% for testing
train, test = df[0:split_point], df[split_point:len(df)]
history = [x for x in train['Value']]
predictions = list()

# Define the ARIMA model
# Replace p, d, q values with your model's parameters
p, d, q = 5, 1, 0  # Example values
model = ARIMA(history, order=(p, d, q))

# Fit the model
model_fit = model.fit()

# Make predictions
for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test['Value'].iloc[t]
    history.append(obs)

# Calculate and print RMSE
rmse = sqrt(mean_squared_error(test['Value'], predictions))
max_err = max_error(test['Value'], predictions)
print('Test RMSE: %.3f' % rmse)
print('Test Max Error: %.3f' % max_err)

# Plot predictions vs actual values
plt.scatter(train['hour_start'], train['Value'].values, label='Actual')
plt.scatter(test['hour_start'], test['Value'].values, label='Actual')
plt.scatter(test['hour_start'], predictions, color='red', label='Predicted')
plt.legend()
plt.show()
