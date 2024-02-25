# Loading training data
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
import seaborn as sns
from sklearn.metrics import mean_squared_error, max_error


def add_reference_column_at_hour_start(df, col_name, new_col_name, n_hours=1):
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


file_path_train = 'train_data.csv'
df_train = pd.read_csv(file_path_train)
y_train = df_train.iloc[:, -1].values.reshape(-1, 1)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])  # Convert the 'timestamp' column to datetime

# df_train = add_reference_column_at_hour_start(df_train,
#                                               'smoothed_difference_2_4',
#                                               'refYY',
#                                               1/2)  # For 'difference_2_4' at 6 PM
# y_pred_train = df_train['refYY'].values.reshape(-1, 1)
#
# rmse = mean_squared_error(y_train, y_pred_train, squared=False)
# max_err = max_error(y_train, y_pred_train)
# print(f"RMSE: {rmse:.2f}")
# print(f"Max Error: {max_err:.2f}")


n_hours_range = np.arange(1/6, 5, 1/6)  # For example, from 0.5 to 6 hours, in half-hour increments
rmse_values = []
max_error_values = []

# Generate a color palette with seaborn
num_colors = 8  # Adjust based on the number of conditions you have
colors = sns.color_palette("rainbow", num_colors)

plt.figure(figsize=(10, 6))
legend_labels = []  # List to hold the legend labels
j = 0
for i, n_hours in enumerate(n_hours_range):
    # Apply the function with the current n_hours interval
    temp_df = add_reference_column_at_hour_start(df_train.copy(), 'smoothed_difference_2_4', 'refYY', n_hours)
    y_pred_train = temp_df['refYY'].values.reshape(-1, 1)

    # Calculate metrics
    rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    max_err = max_error(y_train, y_pred_train)

    rmse_values.append(rmse)
    max_error_values.append(max_err)

    # Plot for specific n_hours intervals
    if n_hours < 4/6 or 0.99 < n_hours < 1.01 or 1.99 < n_hours < 2.01 or 3.99 < n_hours < 4.01:
        # Assuming y_train and y_pred_train are already defined and have the same shape
        errors = y_train.reshape(-1) - y_pred_train.reshape(-1)
        std = np.std(errors)

        # Select color from the palette
        color = colors[i % num_colors]
        sns.kdeplot(errors, color=color, label=f'minutes={60*n_hours:.0f}, std={std:.2f}', linewidth=2)

# Adding the legend outside the plot to the right
plt.legend(title='Interval', loc='best')
plt.title('PDF of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Density')
plt.xlim(-2, 2)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect to make room for the legend
plt.show()

n_minutes_range = n_hours_range * 60

plt.figure(figsize=(10, 5))

# Create the first plot with RMSE values
ax1 = plt.gca()  # Get the current Axes instance
line1 = ax1.plot(n_minutes_range, rmse_values, label='RMSE', marker='o', color='blue', linestyle='-')
ax1.set_xlabel('Minutes')
ax1.set_ylabel('RMSE', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a twin Axes sharing the x-axis for Max Error
ax2 = ax1.twinx()
line2 = ax2.plot(n_minutes_range, max_error_values, label='Max Error', marker='x', color='red', linestyle='--')
ax2.set_ylabel('Max Error', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine the legends for both Axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc='upper left')

plt.title('RMSE and Max Error over Different n_minutes Intervals')
plt.grid(True)

plt.show()