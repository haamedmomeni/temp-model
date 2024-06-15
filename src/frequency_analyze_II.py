# this code is not part of dashboard
# analysis of alignment frequency

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Constants
FILE_PATH_TRAIN = '../data/train_data.csv'
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
df = pd.read_csv(FILE_PATH_TRAIN)
df['timestamp'] = pd.to_datetime(df['timestamp'])

timestamp0 = pd.to_datetime('2024-02-03 18:00:00')
condition = df['timestamp'] > timestamp0
df_train, df_test = df[~condition].copy(), df[condition].copy()
df_test = df

col_list = df.columns.tolist()
keywords = ['time', 'ref', 'diff']
col_list = [col for col in col_list if all(keyword not in col for keyword in keywords)]

x = df_test[TARGET_COLUMN_X].values.reshape(-1, 1)
y = df_test[TARGET_COLUMN_Y].values.reshape(-1, 1)

# Analysis parameters
# n_hours_range = np.arange(1/6, 5, 1/6)  # From 0.5 to 6 hours, in half-hour increments
n_hours_range = [1/6, 2/6, 3/6, 1, 2, 4, 12]  # From 0.5 to 6 hours, in half-hour increments
print(n_hours_range*60)
std_values, max_error_values = [], []

# Create a figure
fig = plt.plot(figsize=(12, 4))

# First subplot for error distributions
for i, n_hours in enumerate(n_hours_range):
    temp_df = add_reference_column_at_hour_start(df_test.copy(), TARGET_COLUMN_Y, 'ref_y', n_hours)
    temp_df = add_reference_column_at_hour_start(temp_df, TARGET_COLUMN_X, 'ref_x', n_hours)
    x_periodic = temp_df['ref_x'].values.reshape(-1, 1)
    y_periodic = temp_df['ref_y'].values.reshape(-1, 1)

    xx = df_test[TARGET_COLUMN_X].values.reshape(-1, 1) - temp_df['ref_x'].values.reshape(-1, 1)
    regressors = temp_df[col_list].values
    model = LinearRegression(fit_intercept=False)
    model.fit(regressors, xx)
    # x_periodic = model.predict(regressors) + temp_df['ref_x'].values.reshape(-1, 1)

    errors = calculate_misalignment_error(ERROR_TYPE, x, y, x_periodic, y_periodic)
    std_val = np.std(errors)
    max_val = np.max(np.abs(errors))
    std_values.append(std_val)
    max_error_values.append(max_val)

    # Plot for specific intervals
    # if any(np.isclose(n_hours, target_hour, atol=1e-2) for target_hour in [1/6, 2/6, 3/6, 1, 2, 4]):
    # sns.histplot(errors.flatten(), binwidth=0.025, element='step', fill=False, color=COLOR_PALETTE[i % NUM_COLORS], label=f'{60 * n_hours:.0f} min, std={std_val:.3f}', linewidth=2)
    sns.ecdfplot(errors.flatten(), color=COLOR_PALETTE[i % NUM_COLORS], label=f'{60 * n_hours:.0f} min', linewidth=2) # , std={std_val:.3f}

ax = plt.gca()
# Add a vertical line at x=0.2
ax.axvline(x=0.2, color='black', linestyle='--', linewidth=1)


# Annotation for the vertical line
x_val = 0.2
y_val = 0.0
ax.annotate('Error Budget = 0.2 arcsec', xy=(x_val, y_val), xytext=(x_val + 0.1, y_val + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=3, headlength=4, linewidth=0.5),
                fontsize=9, color='black')

ax.legend(title='Frequency of alignment', loc='best')
ax.set_title('CDF of misalignment errors')
ax.set_xlabel('Error (pixels, ~1 arcsec)')
ax.set_ylabel('Probability')
ax.grid(True, which='both', axis='y')
if ERROR_TYPE == 'euclidean':
    ax.set_xlim(0, 3)
else:
    ax.set_xlim(-1, 1)

plt.savefig(f'../data/{PLOT_NAME}.png')
plt.show()
