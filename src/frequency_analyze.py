# this code is not part of dashboard
# analysis of alignment frequency

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Constants
FILE_PATH_TRAIN = 'train_data.csv'
NUM_COLORS = 8  # Adjust based on the number of conditions you have
COLOR_PALETTE = sns.color_palette("flare", NUM_COLORS)
TARGET_COLUMN_X = 'smoothed_difference_1_3'
TARGET_COLUMN_Y = 'smoothed_difference_2_4'
PLOT_NAME = 'alignment_frequency'
ERROR_TYPE = 'euclidean'  # 'euclidean', 'x', 'y'


def calculate_misalignment_error(error_type, x, y, x_pred, y_pred):
    if error_type == 'euclidean':
        return ((x_pred - x) ** 2 + (y_pred - y) ** 2) ** 0.5
    elif error_type == 'x':
        return x_pred - x
    elif error_type == 'y':
        return y_pred - y
    else:
        raise ValueError('Invalid type')


def add_reference_column_at_hour_start(df, column_name, new_column_name, hours=1):
    """
    Adds a reference column to the dataframe indicating the first value of a specified column
    at the start of each hour interval defined by n_hours.
    """
    seconds_per_n_hours = hours * 3600
    df['hour_start'] = pd.to_datetime(
        ((df['timestamp'].astype('int64') // 1e9 // seconds_per_n_hours) * seconds_per_n_hours).astype('int64') * 1e9)
    df_first_value_each_hour = df.groupby('hour_start')[column_name].first().reset_index()
    df_first_value_each_hour.rename(columns={column_name: new_column_name}, inplace=True)
    new_df = pd.merge(df, df_first_value_each_hour, on='hour_start', how='left')
    new_df.drop(columns=['hour_start'], inplace=True)
    return new_df


# Load and preprocess data
df_train = pd.read_csv(FILE_PATH_TRAIN)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
x = df_train[TARGET_COLUMN_X].values.reshape(-1, 1)
y = df_train[TARGET_COLUMN_Y].values.reshape(-1, 1)

# Analysis parameters
n_hours_range = np.arange(1/6, 5, 1/6)  # From 0.5 to 6 hours, in half-hour increments
std_values, max_error_values = [], []

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# First subplot for error distributions
for i, n_hours in enumerate(n_hours_range):
    temp_df = add_reference_column_at_hour_start(df_train.copy(), TARGET_COLUMN_Y, 'ref_y', n_hours)
    temp_df = add_reference_column_at_hour_start(temp_df, TARGET_COLUMN_X, 'ref_x', n_hours)
    x_pred = temp_df['ref_x'].values.reshape(-1, 1)
    y_pred = temp_df['ref_y'].values.reshape(-1, 1)

    errors = calculate_misalignment_error(ERROR_TYPE, x, y, x_pred, y_pred)
    std_val = np.std(errors)
    max_val = np.max(np.abs(errors))
    std_values.append(std_val)
    max_error_values.append(max_val)

    # Plot for specific intervals
    if any(np.isclose(n_hours, target_hour, atol=1e-2) for target_hour in [1/6, 2/6, 3/6, 1, 2, 4]):
        sns.histplot(errors.flatten(), binwidth=0.025, element='step', fill=False, color=COLOR_PALETTE[i % NUM_COLORS], label=f'{60 * n_hours:.0f} min, std={std_val:.3f}', linewidth=2, ax=axs[0])

axs[0].legend(title='Frequency of alignment each ', loc='best')
axs[0].set_title('PDF of misalignment errors')
axs[0].set_xlabel('Error (pixels, ~1 mas)')
axs[0].set_ylabel('Density')
if ERROR_TYPE == 'euclidean':
    axs[0].set_xlim(-.1, 2)
else:
    axs[0].set_xlim(-1, 1)

# Second subplot for Std and Max Error over time
n_minutes_range = n_hours_range * 60
line1 = axs[1].plot(n_minutes_range, std_values, label='Std', marker='o', color='blue', linestyle='-')
axs[1].set_xlabel('Frequency of realignment (each x minutes)')
axs[1].set_ylabel('Std (pixels, ~1 mas)', color='blue')
axs[1].tick_params(axis='y', labelcolor='blue')

ax2 = axs[1].twinx()
line2 = ax2.plot(n_minutes_range, max_error_values, label='Max Error', marker='x', color='red', linestyle='--')
ax2.set_ylabel('Max error (pixels, ~1 mas)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines = line1 + [line2[0]]
labels = [l.get_label() for l in lines]
axs[1].legend(lines, labels, loc='upper left')
axs[1].set_title('Misalignment error over different realignment frequencies')
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"{PLOT_NAME}_analyze.jpg")
plt.show()

figure = plt.figure(figsize=(12, 4))
n_hours = 1/6

y_train = df_train[TARGET_COLUMN_Y].values.reshape(-1, 1)
y_pred_train = temp_df['ref_y'].values.reshape(-1, 1)
errors = y_train - y_pred_train
plt.scatter(df_train['timestamp'], y_pred_train - y_train, marker='.', color='blue', linestyle='-')

y_train = df_train[TARGET_COLUMN_X].values.reshape(-1, 1)
y_pred_train = temp_df['ref_x'].values.reshape(-1, 1)
errors = y_train - y_pred_train
plt.scatter(df_train['timestamp'], y_pred_train - y_train, marker='.', color='red', linestyle='-')

y_train = (df_train[TARGET_COLUMN_Y].values.reshape(-1, 1) ** 2 + df_train[TARGET_COLUMN_X].values.reshape(-1, 1) ** 2) ** 0.5
y_pred_train = (temp_df['ref_x'].values.reshape(-1, 1) ** 2 + temp_df['ref_y'].values.reshape(-1, 1) ** 2) ** 0.5
errors = y_train - y_pred_train
plt.scatter(df_train['timestamp'], y_pred_train - y_train, marker='.', color='green', linestyle='-')

plt.show()
