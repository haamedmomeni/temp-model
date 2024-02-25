import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, max_error


def create_lagged_features(X, max_lags):
    n_features = X.shape[1]
    rows = X.shape[0]
    lagged_features_list = []
    for lag in range(1, max_lags + 1, 10):
        lagged_feature = np.roll(X, lag, axis=0)[max_lags:, :]
        lagged_features_list.append(lagged_feature)
    lagged_features = np.hstack(lagged_features_list)
    trimmed_X = X[max_lags:, :]
    combined_features = np.hstack([trimmed_X, lagged_features])
    return combined_features

n_lags = 360

# Loading training data
file_path_train = 'train_data.csv'
df_train = pd.read_csv(file_path_train)
y_train = df_train.iloc[n_lags:, -1].values.reshape(-1, 1)  # Adjust y to match the reduced number of X rows
X_train = df_train.iloc[:, 1:-2].to_numpy()  # Assuming the last column is y and the first is an index or identifier

# Creating lagged features for training data
X_train = create_lagged_features(X_train, n_lags)

# print(X_train.shape)

# Loading testing data
file_path_test = 'test_data.csv'
df_test = pd.read_csv(file_path_test)
y_test = df_test.iloc[n_lags:, -1].values.reshape(-1, 1)  # Similarly adjust y_test
X_test = df_test.iloc[:, 1:-2].to_numpy()

# Creating lagged features for testing data
X_test = create_lagged_features(X_test, n_lags)

# Define a range of alpha values
alpha_range = np.linspace(0.01, 1, 20)

# Initialize lists to store the RMSE and Max Error for each alpha
rmse_values = []
max_error_values = []

# Iterate over alpha values
for alpha in alpha_range:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    max_err = max_error(y_test, y_pred)

    # Append metrics to their respective lists
    rmse_values.append(rmse)
    max_error_values.append(max_err)

# Plotting the metrics
plt.figure(figsize=(10, 5))

# Plot RMSE
plt.subplot(1, 2, 1)
plt.plot(alpha_range, rmse_values, marker='o', linestyle='-', color='blue')
plt.title('RMSE over Alpha')
plt.xlabel('Alpha')
plt.ylabel('RMSE')

# Plot Max Error
plt.subplot(1, 2, 2)
plt.plot(alpha_range, max_error_values, marker='o', linestyle='-', color='red')
plt.title('Max Error over Alpha')
plt.xlabel('Alpha')
plt.ylabel('Max Error')

plt.tight_layout()
plt.show()

# # Training the model
# model = Lasso(alpha=0.2)
# # model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Making predictions
# y_pred = model.predict(X_test)
#
# # Calculating metrics
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# max_err = max_error(y_test, y_pred)
#
# print(f'RMSE: {rmse:.2f}', f'Max Error: {max_err:.2f}')
#
# # Display non-zero features
# non_zero_weights = np.where(model.coef_ != 0)[0]
# print("Indices of non-zero features: ", non_zero_weights)
#
# lst = df_train.columns[1:-2].to_list()
# k = len(lst)
# # Identify non-zero features
# non_zero_indices = np.where(model.coef_ != 0)[0]
# non_zero_feature_names = [f'{lst[i%k].replace("smoothed_", "")} at -{i//k}' for i in non_zero_indices]
# print("Non-zero feature names: ", non_zero_feature_names)