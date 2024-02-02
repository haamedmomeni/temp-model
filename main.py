import numpy as np
from dash import dcc, html, Dash, Input, Output
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data_processing import load_csv, preprocess_data, split_train_test, get_column_list, smooth_data, filter_dataframe_by_hours
from plotting import generate_scatter_plot, create_figure
from get_ip import get_wired_interface_ip

# Load data
FILENAME = '2024-01-10.csv'
raw_df = load_csv(FILENAME)
# Preprocess data
processed_df = preprocess_data(raw_df)
processed_df.to_csv('processed_df.csv')
# Split data into training and test datasets
train_df, test_df = split_train_test(processed_df)
train_df = smooth_data(train_df)
test_df = smooth_data(test_df)

# === Dash App Initialization ===
app = Dash(__name__)

# Assume col_list is a list of column names from your DataFrame
col_list = get_column_list(processed_df)

# Generate the options dynamically based on column names
options = [{'label': col, 'value': f'SHOW_{col.upper().replace(" ", "_")}'} for col in col_list]


# === Dash Component Generating Function ===
def create_graph_div(title, graph_id, data, x_axis_title='Timestamp', y_axis_title=''):
    return html.Div([
        html.H2(title),
        dcc.Graph(
            id=graph_id,
            figure={
                'data': [data],
                'layout': go.Layout(
                    title=title,
                    xaxis={'title': x_axis_title},
                    yaxis={'title': y_axis_title}
                )
            }
        )
    ], style={'width': '48%', 'display': 'inline-block'})


# # === Dash Layout ===
app.layout = html.Div([
    html.H1('Predictive Model Dashboard'),
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap'}, children=[  # Use flexbox for layout
        # First column
        html.Div([
            html.Label('Temperatures:',
                       style={'display': 'inline-block', 'margin-right': '10px', 'fontWeight': 'bold'}),
            dcc.Checklist(id='toggle-data', options=options, value=[]),
            html.Br(),
            html.Div([
                html.Label('Select Starting Hour:', style={'display': 'inline-block', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='start-hour-input',
                    options=[{'label': f'{i} AM' if i < 12 else f'{i} PM', 'value': i} for i in range(24)],
                    value=20,  # Default value
                    clearable=False,
                    style={'width': '150px', 'display': 'inline-block'}
                ),
            ]),
            html.Br(),
            html.Div([
                html.Label('Select Ending Hour:', style={'display': 'inline-block', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='end-hour-input',
                    options=[{'label': f'{i} AM' if i < 12 else f'{i} PM', 'value': i} for i in range(24)],
                    value=6,  # Default value
                    clearable=False,
                    style={'width': '150px', 'display': 'inline-block'}
                ),
            ]),
            html.Br(),
        ], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Second column (Combining Second and Third rows)
        html.Div([
            html.Div(id='equation-display-2'),
            create_graph_div('Training Data (Differential Y motion)', 'train-diff-2-4',
                             generate_scatter_plot(train_df['timestamp'].values,
                                                   train_df['smoothed_difference_2_4'].values,
                                                   'green', 'observation'),
                             y_axis_title='Smoothed Difference 2-4'),
            create_graph_div('Testing Data (Differential Y motion)', 'test-diff-2-4',
                             generate_scatter_plot(test_df['timestamp'].values,
                                                   test_df['smoothed_difference_2_4'].values,
                                                   'purple', 'observation'),
                             y_axis_title='Smoothed Difference 2-4'),
            html.Div(id='equation-display-1'),
            create_graph_div('Training Data (Differential X motion)', 'train-diff-1-3',
                             generate_scatter_plot(train_df['timestamp'].values,
                                                   train_df['smoothed_difference_1_3'].values,
                                                   'blue', 'observation'),
                             y_axis_title='Smoothed Difference 1-3'),
            create_graph_div('Testing Data (Differential X motion)', 'test-diff-1-3',
                             generate_scatter_plot(test_df['timestamp'].values,
                                                   test_df['smoothed_difference_1_3'].values,
                                                   'red', 'observation'),
                             y_axis_title='Smoothed Difference 1-3')
        ], style={'width': '90%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]),
])


def fit_and_predict_linear_eq(toggle_value, train_df, col):
    # Always consider 'Difference 1-3' as the target variable
    y = train_df[col].dropna().values.reshape(-1, 1)

    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}

    # Initialize feature matrix X
    X = np.empty((len(y), 0))

    # Loop through the toggle values
    for value in toggle_value:
        # Extract the original column name using the mapping dictionary
        col_name = value_to_label[value]
        # Now add 'smoothed_' prefix before checking in DataFrame
        col_data = train_df[f'smoothed_{col_name}'].dropna().values.reshape(-1, 1)
        # Append as a new column to feature matrix X
        X = np.c_[X, col_data]

    # Fit the linear equation
    model = LinearRegression()
    model.fit(X, y)

    # Predict the values of y (Difference 1-3)
    y_pred = model.predict(X)
    y_pred = y_pred.ravel()
    rmse = mean_squared_error(y, y_pred, squared=False)

    return y_pred, rmse, model.coef_, model.intercept_


# Function to generate equation string
def generate_equation_str(coef, intercept):
    coef = coef.flatten().tolist()
    equation_terms = [f"{c:.2f}*x{i + 1}" for i, c in enumerate(coef)]
    return f"y = {intercept[0]:.2f} + {' + '.join(equation_terms)}"


@app.callback(
    [Output('train-diff-1-3', 'figure'),
     Output('test-diff-1-3', 'figure'),
     Output('equation-display-1', 'children')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value')]
)
def update_diff_1_3_graphs(toggle_value, start_hour, end_hour):

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    # Initialize data list with original scatter plots
    train_data_list = [generate_scatter_plot(trained_df_filtered['timestamp'].values,
                                             trained_df_filtered['smoothed_difference_1_3'].values,
                                             'blue', 'observation')]
    test_data_list = [generate_scatter_plot(test_df_filtered['timestamp'].values,
                                            test_df_filtered['smoothed_difference_1_3'].values,
                                            'green', 'observation')]

    equation_str = "No equation to display"

    # If no features selected, return early with default figures and equation
    if not toggle_value:
        return (create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 3 - Column 1)'),
                create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 3 - Column 1)'),
                equation_str)

    # Generate predictions and update data list
    y_pred_train, rmse_train, coef, intercept = fit_and_predict_linear_eq(toggle_value, trained_df_filtered,
                                                            col='smoothed_difference_1_3')
    y_pred_test, rmse_test, _, _ = fit_and_predict_linear_eq(toggle_value, test_df_filtered,
                                                            col='smoothed_difference_1_3')
    train_data_list.append(generate_scatter_plot(trained_df_filtered['timestamp'].values, y_pred_train,
                                                 'red', 'prediction'+f", rmse: {rmse_train:.2f}"))
    test_data_list.append(generate_scatter_plot(test_df_filtered['timestamp'].values, y_pred_test,
                                                'orange', 'prediction'+f", rmse: {rmse_test:.2f}"))

    # Update equation string
    equation_str = generate_equation_str(coef, intercept)

    return (create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 3 - Column 1)'),
            create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 3 - Column 1)'),
            equation_str)

@app.callback(
    [Output('train-diff-2-4', 'figure'),
     Output('test-diff-2-4', 'figure'),
     Output('equation-display-2', 'children')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value')]
)
def update_diff_2_4_graphs(toggle_value, start_hour, end_hour):

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    # Initialize data list with original scatter plots
    train_data_list = [generate_scatter_plot(trained_df_filtered['timestamp'].values,
                                             trained_df_filtered['smoothed_difference_2_4'].values,
                                             'blue', 'observation')]
    test_data_list = [generate_scatter_plot(test_df_filtered['timestamp'].values,
                                            test_df_filtered['smoothed_difference_2_4'].values,
                                            'green', 'observation')]

    equation_str = "No equation to display"

    # If no features selected, return early with default figures and equation
    if not toggle_value:
        return (create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 4 - Column 2)'),
                create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 4 - Column 2)'),
                equation_str)

    # Generate predictions and update data list
    y_pred_train, rmse_train, coef, intercept = fit_and_predict_linear_eq(toggle_value, trained_df_filtered,
                                                            col='smoothed_difference_2_4')
    y_pred_test, rmse_test, _, _ = fit_and_predict_linear_eq(toggle_value, test_df_filtered,
                                                            col='smoothed_difference_2_4')
    train_data_list.append(generate_scatter_plot(trained_df_filtered['timestamp'].values, y_pred_train,
                                                 'red', 'prediction'+f", rmse: {rmse_train:.2f}"))
    test_data_list.append(generate_scatter_plot(test_df_filtered['timestamp'].values, y_pred_test,
                                                'orange', 'prediction'+f", rmse: {rmse_test:.2f}"))

    # Update equation string
    equation_str = generate_equation_str(coef, intercept)

    return (create_figure(train_data_list, 'Training Data: Smoothed Difference (Column 4 - Column 2)'),
            create_figure(test_data_list, 'Test Data: Smoothed Difference (Column 4 - Column 2)'),
            equation_str)


# === Run the App ===
if __name__ == '__main__':
    ip_address = get_wired_interface_ip()
    app.run_server(debug=True, host=f"{ip_address}",
                   port=8050)
    # app.run_server(debug=True)
