from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from io import BytesIO
from zipfile import ZipFile

from data_processing import (load_csv, preprocess_data, split_train_test,
                             get_column_list, smooth_data, filter_dataframe_by_hours)
from plotting import generate_scatter_plot, create_graph_div, update_graphs_and_predictions
from get_ip import get_wired_interface_ip
import numpy as np
# import pandas as pd

# df1 = load_csv('src/2024-01-10.csv')
# df2 = load_csv('src/2024-01-13.csv')
#
# processed_df1 = preprocess_data(df1)
# processed_df2 = preprocess_data(df2)
#
# processed_df1[['difference_1_3', 'difference_2_4']] = -processed_df1[['difference_1_3', 'difference_2_4']]
#
# processed_df = pd.concat([processed_df1, processed_df2], ignore_index=True)

# Load data
FILENAME = 'src/2024-01-13.csv'
raw_df = load_csv(FILENAME)

# Preprocess data
processed_df = preprocess_data(raw_df)
processed_df = smooth_data(processed_df)

# List all unique dates in the dataframe
unique_dates = np.sort(processed_df['date'].unique())[:-1]
# Format dates as strings
formatted_dates = [date.strftime('%Y/%m/%d') for date in unique_dates]
# Create dropdown options
dropdown_options = [{'label': date, 'value': date} for date in formatted_dates]

# Split data into training and test datasets
train_df, test_df = split_train_test(processed_df, formatted_dates[-1])
# train_df = smooth_data(train_df)
# test_df = smooth_data(test_df)

# Assume col_list is a list of column names from your DataFrame
col_list = get_column_list(processed_df)
# Generate the options dynamically based on column names
options = [{'label': col, 'value': f'SHOW_{col.upper().replace(" ", "_")}'} for col in col_list]

model_type_options = [
    {'label': 'Linear Regression', 'value': 'LR'},
    {'label': 'XGBoost', 'value': 'XGB'},
]


# === Dash App Initialization ===
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# # === Dash Layout ===
app.layout = html.Div([
    html.H3('Predictive Model Dashboard'),

    html.Button('Download Data as ZIP', id='download-zip-btn',
                style={'display': 'block', 'margin': '10px', 'fontWeight': 'bold', "color": "white",
                       "background-color": "#3B3B3B", "padding": "5px", "border-radius": "5px"}),
    dcc.Download(id='download-zip'),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}, children=[
        # First column
        html.Div([

            html.Div([

                html.Label('Choose Model Type:',
                           style={'display': 'inline-block', 'margin-right': '10px', 'fontWeight': 'bold',
                                  "color": "white"}),

                dcc.RadioItems(
                    id='model-type-toggle',
                    options=model_type_options,
                    value='LR',  # Default value
                    labelStyle={'display': 'block', "color": "white", "background-color": "#3B3B3B", "padding": "5px",
                                "border-radius": "5px", "margin": "5px"},
                    style={'fontWeight': 'bold'}
                )
            ],
                style={'padding': '10px', 'background-color': '#2D2D2D', 'border-radius': '5px'}),

            html.Label('Temperatures:',
                       style={'display': 'inline-block', 'margin-right': '10px', 'fontWeight': 'bold'}),

            dcc.Checklist(
                id='toggle-data',
                options=options,
                value=[],
                style={'display': 'block', 'margin': '5px', 'fontWeight': 'bold'},
                inputStyle={"margin-right": "5px", "cursor": "pointer"},
                labelStyle={"display": "block", "margin": "5px", "color": "white",
                            "background-color": "#3B3B3B", "padding": "5px", "border-radius": "5px"}
            ),

            html.Br(),

            html.Div([
                html.Label('Select Starting Hour:',
                           style={'display': 'inline-block', 'margin-right': '10px',  'width': '150px'}),
                dcc.Dropdown(
                    id='start-hour-input',
                    options=[{'label': f'{i} AM' if i < 12 else f'{i} PM', 'value': i} for i in range(24)],
                    value=20,  # Default value
                    clearable=False,
                    style={'width': '80px', 'display': 'inline-block', 'color': 'black'}
                ),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '10px'}),

            html.Br(),

            html.Div([

                html.Label('Select Ending Hour:',
                           style={'display': 'inline-block', 'margin-right': '10px',  'width': '150px'}),
                dcc.Dropdown(
                    id='end-hour-input',
                    options=[{'label': f'{i} AM' if i < 12 else f'{i} PM', 'value': i} for i in range(24)],
                    value=6,  # Default value
                    clearable=False,
                    style={'width': '80px', 'display': 'inline-block', 'color': 'black'}
                ),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '10px'}),

            html.Br(),

            html.Div([
                html.Label('Select Test Date:',
                           style={'display': 'inline-block', 'margin-right': '10px', 'width': '120px'}),
                dcc.Dropdown(
                    id='test-date-dropdown',
                    options=dropdown_options,
                    value=formatted_dates[-1],  # Default value
                    clearable=False,
                    style={'width': '110px', 'display': 'inline-block', 'color': 'black'}
                ),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '10px'}),

        ], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Second column
        html.Div([
            html.Div(id='equation-display-2', style={'color': '#FF5349'}),
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
            html.Div(id='equation-display-1', style={'color': '#FF5349'}),
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
        ], style={'width': '90%', 'display': 'inline-block'}),
    ]),
])


@app.callback(
    [Output('train-diff-1-3', 'figure'),
     Output('test-diff-1-3', 'figure'),
     Output('equation-display-1', 'children')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value'),
     Input('model-type-toggle', 'value'),
     Input('test-date-dropdown', 'value')
     ]
)
def update_diff_1_3_graphs(toggle_value, start_hour, end_hour, model_type, test_date_str):
    return update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, processed_df, test_date_str,
                                         'smoothed_difference_1_3',
                                         'Training Data: Smoothed Difference (Column 3 - Column 1)',
                                         'Test Data: Smoothed Difference (Column 3 - Column 1)',
                                         options)


@app.callback(
    [Output('train-diff-2-4', 'figure'),
     Output('test-diff-2-4', 'figure'),
     Output('equation-display-2', 'children')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value'),
     Input('model-type-toggle', 'value'),
     Input('test-date-dropdown', 'value')]
)
def update_diff_2_4_graphs(toggle_value, start_hour, end_hour, model_type, test_date_str):
    return update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, processed_df, test_date_str,
                                         'smoothed_difference_2_4',
                                         'Training Data: Smoothed Difference (Column 4 - Column 2)',
                                         'Test Data: Smoothed Difference (Column 4 - Column 2)',
                                         options)


@app.callback(
    Output('download-zip', 'data'),
    Input('download-zip-btn', 'n_clicks'),
    State('toggle-data', 'value'),
    State('start-hour-input', 'value'),
    State('end-hour-input', 'value'),
    prevent_initial_call=True
)
def generate_and_download_zip(n_clicks, toggle_value, start_hour, end_hour):
    if n_clicks > 0:

        trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
        test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

        # Create a mapping from value to label
        value_to_label = {option['value']: option['label'] for option in options}

        # Filter columns based on selected options
        selected_columns = (['timestamp'] +
                            [value_to_label[value] for value in toggle_value] +
                            ['smoothed_difference_1_3', 'smoothed_difference_2_4'])

        # Create in-memory ZIP file
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            # Convert train_df and test_df to CSV strings without index
            train_csv = trained_df_filtered[selected_columns].dropna().to_csv(index=False)
            test_csv = test_df_filtered[selected_columns].dropna().to_csv(index=False)

            # Write CSV strings to the ZIP file
            zip_file.writestr('train_data.csv', train_csv)
            zip_file.writestr('test_data.csv', test_csv)

        # Prepare the ZIP file to be sent to the client
        zip_buffer.seek(0)
        return dcc.send_bytes(zip_buffer.getvalue(), filename="data.zip")


# === Run the App ===
if __name__ == '__main__':
    # ip_address = get_wired_interface_ip()
    # app.run_server(debug=True, host=f"{ip_address}",
    #                port=8050)
    app.run_server(debug=True)
