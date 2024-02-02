# from sklearn.linear_model import LinearRegression
# import numpy as np
# from dash import dcc, html, Dash, Input, Output
# import plotly.graph_objs as go
# import pandas as pd
#
#
# # === Data Loading and Preprocessing ===
# def load_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     df.drop([0, 1], inplace=True)
#     df.dropna(inplace=True)
#
#     # Loop through each column and convert to float if the dtype is object
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             try:
#                 df[col] = df[col].astype(float)
#             except ValueError:
#                 print(f"Cannot convert column {col} to float.")
#
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['date'] = df['timestamp'].dt.date
#     unique_dates = np.sort(df['date'].unique())
#     day_before_last = unique_dates[-2]
#     train_df = df[df['date'] < day_before_last].copy()
#     test_df = df[df['date'] == day_before_last].copy()
#     col_list = df.columns[7:-1]
#     return train_df, test_df, col_list
#
#
# # === Feature Engineering ===
# def calculate_differences(train_df, test_df):
#     for df in [train_df, test_df]:
#         df['difference_1_3'] = df.iloc[:, 3] - df.iloc[:, 1]
#         df['difference_2_4'] = df.iloc[:, 4] - df.iloc[:, 2]
#
#
# def smooth_data(df, window=5):
#     for column in df.select_dtypes(include=[np.number]).columns:
#         df[f'smoothed_{column}'] = df[column].rolling(window=window).mean()
#
#
# # === Plotting Function ===
# def generate_scatter_plot(x_array, y_array, color):
#     return go.Scatter(
#         x=x_array,
#         y=y_array,
#         mode='markers',
#         marker=dict(size=5, color=color)
#     )
#
#
# # === Dash App Initialization ===
# app = Dash(__name__)
# train_df, test_df, col_list = load_and_preprocess_data('2024-01-10.csv')
# calculate_differences(train_df, test_df)
# smooth_data(train_df)
# smooth_data(test_df)
#
# # Assume col_list is a list of column names from your DataFrame
# # Generate the options dynamically based on column names
# options = [{'label': col, 'value': f'SHOW_{col.upper().replace(" ", "_")}'} for col in col_list]
#
#
# # === Dash Component Generating Function ===
# def create_graph_div(title, graph_id, data, x_axis_title='Timestamp', y_axis_title=''):
#     return html.Div([
#         html.H2(title),
#         dcc.Graph(
#             id=graph_id,
#             figure={
#                 'data': [data],
#                 'layout': go.Layout(
#                     title=title,
#                     xaxis={'title': x_axis_title},
#                     yaxis={'title': y_axis_title}
#                 )
#             }
#         )
#     ], style={'width': '48%', 'display': 'inline-block'})
#
#
# # === Dash Layout ===
# app.layout = html.Div([
#     html.H1('Temp Data Dashboard'),
#     html.Div([  # First row
#         dcc.Checklist(
#             id='toggle-data',
#             options=options,
#             value=[]
#         ),
#
#         html.Div(id='equation-display-1'),
#
#         create_graph_div('Training Data (Difference 1-3)', 'train-diff-1-3',
#                          generate_scatter_plot(train_df['timestamp'].values, train_df['smoothed_difference_1_3'].values,
#                                                'blue'),
#                          y_axis_title='Smoothed Difference 1-3'),
#         create_graph_div('Testing Data (Difference 1-3)', 'test-diff-1-3',
#                          generate_scatter_plot(test_df['timestamp'].values, test_df['smoothed_difference_1_3'].values,
#                                                'red'),
#                          y_axis_title='Smoothed Difference 1-3')
#     ]),
#     html.Div([  # Second row
#         html.Div(id='equation-display-2'),
#         create_graph_div('Training Data (Difference 2-4)', 'train-diff-2-4',
#                          generate_scatter_plot(train_df['timestamp'].values, train_df['smoothed_difference_2_4'].values,
#                                                'green'),
#                          y_axis_title='Smoothed Difference 2-4'),
#         create_graph_div('Testing Data (Difference 2-4)', 'test-diff-2-4',
#                          generate_scatter_plot(test_df['timestamp'].values, test_df['smoothed_difference_2_4'].values,
#                                                'purple'),
#                          y_axis_title='Smoothed Difference 2-4')
#     ])
# ])
#
#
# # Function to create a figure dictionary
# def create_figure(data_list, title):
#     return {
#         'data': data_list,
#         'layout': go.Layout(
#             title=title,
#             xaxis={'title': 'Timestamp'},
#             yaxis={'title': 'Smoothed Difference 1-3'}
#         )
#     }
#
#
# # Function to generate equation string
# def generate_equation_str(coef, intercept):
#     coef = coef.flatten().tolist()
#     equation_terms = [f"{c:.2f}*x{i + 1}" for i, c in enumerate(coef)]
#     return f"y = {intercept[0]:.2f} + {' + '.join(equation_terms)}"
#
# @app.callback(
#     [Output('train-diff-1-3', 'figure'),
#      Output('test-diff-1-3', 'figure'),
#      Output('equation-display-1', 'children')],
#     [Input('toggle-data', 'value')]
# )
# def update_diff_1_3_graphs(toggle_value):
#     # Initialize data list with original scatter plots
#     train_data_list = [generate_scatter_plot(train_df['timestamp'].values, train_df['smoothed_difference_1_3'].values, 'blue')]
#     test_data_list = [generate_scatter_plot(test_df['timestamp'].values, test_df['smoothed_difference_1_3'].values, 'green')]
#
#     equation_str = "No equation to display"
#
#     # If no features selected, return early with default figures and equation
#     if not toggle_value:
#         return create_figure(train_data_list, 'Training Data'), create_figure(test_data_list, 'Test Data'), equation_str
#
#     # Generate predictions and update data list
#     y_pred_train, coef, intercept = fit_and_predict_linear_eq(toggle_value, train_df)
#     y_pred_test, _, _ = fit_and_predict_linear_eq(toggle_value, test_df)
#     train_data_list.append(generate_scatter_plot(train_df['timestamp'].values, y_pred_train, 'red'))
#     test_data_list.append(generate_scatter_plot(test_df['timestamp'].values, y_pred_test, 'orange'))
#
#     # Update equation string
#     equation_str = generate_equation_str(coef, intercept)
#
#     return create_figure(train_data_list, 'Training Data'), create_figure(test_data_list, 'Test Data'), equation_str
#
# @app.callback(
#     [Output('train-diff-2-4', 'figure'),
#      Output('test-diff-2-4', 'figure'),
#      Output('equation-display-2', 'children')],
#     [Input('toggle-data', 'value')]
# )
# def update_diff_2_4_graphs(toggle_value):
#     # Initialize data list with original scatter plots
#     train_data_list = [generate_scatter_plot(train_df['timestamp'].values, train_df['smoothed_difference_2_4'].values, 'blue')]
#     test_data_list = [generate_scatter_plot(test_df['timestamp'].values, test_df['smoothed_difference_2_4'].values, 'green')]
#
#     equation_str = "No equation to display"
#
#     # If no features selected, return early with default figures and equation
#     if not toggle_value:
#         return create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 4 - Column 2)'), create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 4 - Column 2)'), equation_str
#
#     # Generate predictions and update data list
#     y_pred_train, coef, intercept = fit_and_predict_linear_eq(toggle_value, train_df)
#     y_pred_test, _, _ = fit_and_predict_linear_eq(toggle_value, test_df)
#     train_data_list.append(generate_scatter_plot(train_df['timestamp'].values, y_pred_train, 'red'))
#     test_data_list.append(generate_scatter_plot(test_df['timestamp'].values, y_pred_test, 'orange'))
#
#     # Update equation string
#     equation_str = generate_equation_str(coef, intercept)
#
#     return create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 4 - Column 2)'), create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 4 - Column 2)'), equation_str
#
#
# def fit_and_predict_linear_eq(toggle_value, train_df):
#     # Always consider 'Difference 1-3' as the target variable
#     y = train_df['smoothed_difference_1_3'].dropna().values.reshape(-1, 1)
#
#     # Create a mapping dictionary from value to label
#     value_to_label = {option['value']: option['label'] for option in options}
#
#     # Initialize feature matrix X
#     X = np.empty((len(y), 0))
#
#     # Loop through the toggle values
#     for value in toggle_value:
#         # Extract the original column name using the mapping dictionary
#         # if value in value_to_label:
#         col_name = value_to_label[value]
#         # Now add 'smoothed_' prefix before checking in DataFrame
#         col_data = train_df[f'smoothed_{col_name}'].dropna().values.reshape(-1, 1)
#         # Append as a new column to feature matrix X
#         X = np.c_[X, col_data]
#
#     # Fit the linear equation
#     model = LinearRegression()
#     model.fit(X, y)
#
#     # Predict the values of y (Difference 1-3)
#     y_pred = model.predict(X)
#     y_pred = y_pred.ravel()
#
#     return y_pred, model.coef_, model.intercept_
#
#
# # === Run the App ===
# if __name__ == '__main__':
#     app.run_server(debug=True)
