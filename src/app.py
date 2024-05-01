import io
import zipfile

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from io import BytesIO
from zipfile import ZipFile
import base64

from data_processing import (load_csv, preprocess_data, split_train_test,
                             get_column_list, smooth_data, filter_dataframe_by_hours, filter_train_data_by_date)
from plotting import generate_scatter_plot, create_graph_div, update_graphs_and_predictions, update_graphs_raw
from get_ip import get_wired_interface_ip
import numpy as np


def generate_dropdown(label, id, options, value, style, label_style, div_style):
    """Generates a div containing a label and a dropdown."""
    return html.Div([
        html.Label(label, style=label_style),
        dcc.Dropdown(id=id, options=options, value=value, clearable=False, style=style),
    ], style=div_style)


# Load data
# FILENAME = '2024-01-10_v2.csv'
FILENAME = '../data/2024-03-30_04-18.csv'
raw_df = load_csv(FILENAME)

# Preprocess data
processed_df = preprocess_data(raw_df, 0)
processed_df = smooth_data(processed_df, 5)


# List all unique dates in the dataframe
unique_dates = np.sort(processed_df['date'].unique())[:-1]
# Format dates as strings
formatted_dates = [date.strftime('%Y/%m/%d') for date in unique_dates]
# Create dropdown options
dropdown_options = [{'label': date, 'value': date} for date in formatted_dates]


# Split data into training and test datasets
train_df, test_df = split_train_test(processed_df, formatted_dates[-1], formatted_dates[-2], formatted_dates[-2])

# Assume col_list is a list of column names from your DataFrame
col_list = get_column_list(processed_df)
# Generate the options dynamically based on column names
options = [{'label': col, 'value': f'SHOW_{col.upper().replace(" ", "_")}'} for col in col_list]
options = sorted(options, key=lambda x: x['label'])

model_type_options = [
    {'label': 'Linear Regression', 'value': 'LR'},
    {'label': 'K-Nearest Neighbors', 'value': 'KNN'},
    {'label': 'XGBoost', 'value': 'XGB'},
    {'label': 'Random Forest', 'value': 'RF'},
    {'label': 'Support Vector Machine', 'value': 'SVM'},
]

interval_options = [
    {'label': '10 minutes', 'value': 10},
    {'label': '20 minutes', 'value': 20},
    {'label': '30 minutes', 'value': 30},
    {'label': '1 hour', 'value': 60},
    {'label': '2 hours', 'value': 120},
    {'label': '3 hours', 'value': 180},
    {'label': '4 hours', 'value': 240},
    {'label': '6 hours', 'value': 360},
    {'label': '8 hours', 'value': 480},
    {'label': '12 hours', 'value': 720},
    # {'label': 'None', 'value': 0}
]

button_style = {'display': 'block', 'margin': '10px', 'fontWeight': 'bold', "color": "white",
                "background-color": "#3B3B3B", "padding": "5px", "border-radius": "5px"}

label_style = {'display': 'inline-block', 'margin-right': '10px', 'fontWeight': 'bold', "color": "white"}

div_style = {'display': 'flex', 'align-items': 'center', 'margin-right': '10px', 'justifyContent': 'space-between'}

dropdown_style = {'width': '110px', 'display': 'inline-block', 'color': 'black'}

hour_options = [{'label': f'{i} AM' if i < 12 else f'{i} PM', 'value': i} for i in range(24)]

# === Dash App Initialization ===
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# # === Dash Layout ===
app.layout = html.Div([
    html.H3('Predictive Model Dashboard'),

    html.Button('Download Data as .Zip File', id='download-zip-btn', style=button_style),
    dcc.Download(id='download-zip'),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}, children=[
        # First column
        html.Div([

            html.Div([

                html.Label('Choose Model Type:', style=label_style),

                dcc.RadioItems(
                    id='model-type-toggle',
                    options=model_type_options,
                    value='LR',
                    labelStyle={'display': 'block', "color": "white", "background-color": "#3B3B3B", "padding": "5px",
                                "border-radius": "5px", "margin": "5px"},
                    style={'fontWeight': 'bold'}
                )
            ],
                style={'padding': '10px', 'background-color': '#2D2D2D', 'border-radius': '5px'}),

            html.Label('Temperatures:', style=label_style),
            html.Br(),
            html.Div(
                children=[
                    html.Button('Select All', id='select-all', n_clicks=0, style=button_style),
                    html.Button('Deselect All', id='deselect-all', n_clicks=0, style=button_style)
                ],
                style={'display': 'flex', 'flexDirection': 'row'}),
            dcc.Checklist(
                id='toggle-data',
                options=options,
                value=[],
                style={'display': 'block', 'margin': '5px', 'fontWeight': 'bold'},
                inputStyle={"margin-right": "5px", "cursor": "pointer"},
                labelStyle={"display": "block", "margin": "5px", "color": "white",
                            "background-color": "#3B3B3B", "padding": "5px", "border-radius": "5px"}
            ),
        ], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Second column
        html.Div([
            # Interval, Starting Hour, Ending Hour, and Date Selections
            generate_dropdown('Select Realignment Interval (refX, refY):', 'interval-selection-dropdown',
                              interval_options, 720, dropdown_style, label_style, div_style),
            generate_dropdown('Select Starting Hour:', 'start-hour-input',
                              hour_options, 20, dropdown_style, label_style, div_style),
            generate_dropdown('Select Ending Hour:', 'end-hour-input',
                              hour_options, 6, dropdown_style, label_style, div_style),
            generate_dropdown('Select Starting Date for Train:', 'train-date-start-dropdown',
                              dropdown_options, formatted_dates[-6], dropdown_style, label_style, div_style),
            generate_dropdown('Select Ending Date for Train:', 'train-date-end-dropdown',
                              dropdown_options, formatted_dates[-1], dropdown_style, label_style, div_style),
            generate_dropdown('Select Test Date:', 'test-date-dropdown',
                              dropdown_options, formatted_dates[-1], dropdown_style, label_style, div_style),
        ], style={'width': '14%', 'display': 'inline-block'}),

        # Third column
        html.Div([
            html.Div(id='equation-display-2', style={'color': '#FF5349'}),
            create_graph_div('Training Data (Differential Y motion)', 'train-diff-2-4',
                             generate_scatter_plot(train_df['timestamp'].values,
                                                   train_df['smoothed_diffY'].values,
                                                   'green', 'Observation'),),
            create_graph_div('Testing Data (Differential Y motion)', 'test-diff-2-4',
                             generate_scatter_plot(test_df['timestamp'].values,
                                                   test_df['smoothed_diffY'].values,
                                                   'purple', 'Observation'),),
            html.Div(id='equation-display-1', style={'color': '#FF5349'}),
            create_graph_div('Training Data (Differential X motion)', 'train-diff-1-3',
                             generate_scatter_plot(train_df['timestamp'].values,
                                                   train_df['smoothed_diffX'].values,
                                                   'blue', 'Observation'),),
            create_graph_div('Testing Data (Differential X motion)', 'test-diff-1-3',
                             generate_scatter_plot(test_df['timestamp'].values,
                                                   test_df['smoothed_diffX'].values,
                                                   'red', 'Observation'),),
            create_graph_div('Training Data (Differential X motion)', 'train-raw',
                             generate_scatter_plot(train_df['timestamp'].values,
                                                   train_df['smoothed_diffX'].values,
                                                   'blue', 'Observation'),),
            create_graph_div('Testing Data (Differential X motion)', 'test-raw',
                             generate_scatter_plot(test_df['timestamp'].values,
                                                   test_df['smoothed_diffX'].values,
                                                   'red', 'Observation'),),
        ], style={'width': '75%', 'display': 'inline-block'}),
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
     Input('test-date-dropdown', 'value'),
     Input('train-date-start-dropdown', 'value'),
     Input('train-date-end-dropdown', 'value'),
     Input('interval-selection-dropdown', 'value')
     ])
def update_diff_1_3_graphs(toggle_value, start_hour, end_hour, model_type,
                           test_date_str, train_date_start_str, train_date_end_str, interval):
    processed_df = update_processed_data(interval)
    return update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, processed_df,
                                         test_date_str, train_date_start_str, train_date_end_str,
                                         'smoothed_diffX',
                                         'Training Data (Differential X motion)',
                                         'Testing Data (Differential X motion)',
                                         options)


@app.callback(
    [Output('train-diff-2-4', 'figure'),
     Output('test-diff-2-4', 'figure'),
     Output('equation-display-2', 'children')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value'),
     Input('model-type-toggle', 'value'),
     Input('test-date-dropdown', 'value'),
     Input('train-date-start-dropdown', 'value'),
     Input('train-date-end-dropdown', 'value'),
     Input('interval-selection-dropdown', 'value')
     ])
def update_diff_2_4_graphs(toggle_value, start_hour, end_hour, model_type,
                           test_date_str, train_date_start_str, train_date_end_str, interval):
    processed_df = update_processed_data(interval)
    return update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, processed_df,
                                         test_date_str, train_date_start_str, train_date_end_str,
                                         'smoothed_diffY',
                                         'Training Data (Differential Y motion)',
                                         'Testing Data (Differential Y motion)',
                                         options)


@app.callback(
    [Output('train-raw', 'figure'),
     Output('test-raw', 'figure')],
    [Input('toggle-data', 'value'),
     Input('start-hour-input', 'value'),
     Input('end-hour-input', 'value'),
     Input('test-date-dropdown', 'value'),
     Input('train-date-start-dropdown', 'value'),
     Input('train-date-end-dropdown', 'value'),
     Input('interval-selection-dropdown', 'value')
     ])
def update_raw_graphs(toggle_value, start_hour, end_hour,
                           test_date_str, train_date_start_str, train_date_end_str, interval):
    processed_df = update_processed_data(interval)

    return update_graphs_raw(toggle_value, start_hour, end_hour, processed_df,
                                         test_date_str, train_date_start_str, train_date_end_str,
                                         'Training Data (Temperatures)',
                                         'Testing Data (Temperatures)',
                                         options)

@app.callback(
    Output("download-zip", "data"),
    Input("download-zip-btn", "n_clicks"),
    State('toggle-data', 'value'),
    State('start-hour-input', 'value'),
    State('end-hour-input', 'value'),
    State('test-date-dropdown', 'value'),
    State('train-date-start-dropdown', 'value'),
    State('train-date-end-dropdown', 'value'),
    State('interval-selection-dropdown', 'value'),
    prevent_initial_call=True,
)
def func(n_clicks, toggle_value, start_hour, end_hour, test_date_str, train_date_start_str, train_date_end_str, interval):
    processed_df = update_processed_data(interval)
    train_df, test_df = split_train_test(processed_df, test_date_str, train_date_start_str, train_date_end_str)
    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)
    value_to_label = {option['value']: option['label'] for option in options}
    selected_columns = (['timestamp'] +
                        ['smoothed_'+value_to_label[value] for value in toggle_value] +
                        ['smoothed_refX', 'smoothed_refY'] +
                        ['smoothed_diffX', 'smoothed_diffY'])

    test_csv = test_df_filtered[selected_columns].to_csv(index=False)
    train_csv = trained_df_filtered[selected_columns].to_csv(index=False)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('test_data.csv', test_csv)
        zf.writestr('train_data.csv', train_csv)
    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.getvalue(), filename="data.zip")


def update_processed_data(interval):
    # Re-load raw data if needed, or use it if already available in memory
    raw_df = load_csv(FILENAME)  # Consider optimizing this to avoid reloading

    # Update the call to your preprocess function with the selected interval
    processed_df = preprocess_data(raw_df, interval)  # Adjust function signature as needed
    processed_df = smooth_data(processed_df, 1)

    return processed_df


@app.callback(
    Output('toggle-data', 'value'),
    [Input('select-all', 'n_clicks'),
     Input('deselect-all', 'n_clicks')],
    [State('toggle-data', 'options')])
def update_checklist(select_all_clicks, deselect_all_clicks, options):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'select-all':
        return [option['value'] for option in options]
    elif button_id == 'deselect-all':
        return []

    return dash.no_update


# === Run the App ===
if __name__ == '__main__':
    ip_address = get_wired_interface_ip()
    app.run_server(debug=True, host=f"{ip_address}",
                   port=8050)
    # app.run_server(debug=True)
