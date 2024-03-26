import random

import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from data_processing import filter_dataframe_by_hours, fit_and_predict_training_data, split_train_test, \
    predict_test_data, smooth_data
from datetime import datetime


def generate_scatter_plot(x_array, y_array, color, name):
    return go.Scatter(
        x=x_array,
        y=y_array,
        mode='markers',
        marker=dict(size=5, color=color),
        name=name
    )


# Function to create a figure dictionary
def create_figure(data_list, title, y_axis_title='Motion (arcsec)'):
    return {
        'data': data_list,
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Motion (arcsec)'},
            uirevision='constant'
        )
    }


# Function to generate equation string
def generate_equation_str(coef, intercept, toggle_value):
    coef_list = coef.flatten().tolist()
    equation_terms = [f"{'+' if c > 0 else ''}{c:.2f} &theta;<sub>{toggle_value[i][5:].lower()}</sub>" for i, c in enumerate(coef_list)]
    return dcc.Markdown(f"y - y<sub>ref</sub> = &Sigma;(&theta;<sub>i</sub> - &theta;<sub>i, ref</sub>) = "
                        f" {' '.join(equation_terms)}",
                        dangerously_allow_html=True)
    # if intercept:
    #     intercept_val = intercept[0]
    #     f"Equation: {intercept_val:.2f}  {' '.join(equation_terms)}"
    # else:
    #     return f"Equation: y-y_ref = \sigma t-t_ref {' '.join(equation_terms)}"


def create_graph_div(title, graph_id, data):
    return html.Div([
        dcc.Graph(
            id=graph_id,
            figure={
                'data': [data],
                'layout': go.Layout(
                    title=title,
                    paper_bgcolor='#2D2D2D',
                    plot_bgcolor='#2D2D2D'
                )
            }
        )
    ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '10px'})


def update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, df, test_date_str,
                                  train_date_start_str, train_date_end_str, diff_col,
                                  train_fig_title, test_fig_title, options):
    test_date = datetime.strptime(test_date_str, '%Y/%m/%d')
    train_df, test_df = split_train_test(df, test_date, train_date_start_str, train_date_end_str)

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    trained_df_filtered = smooth_data(trained_df_filtered)
    test_df_filtered = smooth_data(test_df_filtered)


    t_train = trained_df_filtered['timestamp'].values
    t_test = test_df_filtered['timestamp'].values

    y_train = trained_df_filtered[diff_col].values
    y_test = test_df_filtered[diff_col].values

    y_train_std = np.std(y_train)
    y_test_std = np.std(y_test)

    model, y_pred_train, rmse_train, max_err_train, coef, intercept = fit_and_predict_training_data(
        model_type, toggle_value, trained_df_filtered, diff_col, options)
    y_pred_test, rmse_test, max_err_test = predict_test_data(
        model, toggle_value, test_df_filtered, diff_col, options)

    train_data_list = [generate_scatter_plot(t_train, y_train, 'blue', f'<b>Observation</b><br>std: {y_train_std:.3f}')]
    test_data_list = [generate_scatter_plot(t_test, y_test, 'green', f'<b>Observation</b><br>std: {y_test_std:.3f}')]

    train_data_list.append(generate_scatter_plot(t_train, y_pred_train, 'red',
                                                 f"<b>Prediction</b><br>rmse: {rmse_train:.3f}<br>max err: {max_err_train:.2f}"))
    test_data_list.append(generate_scatter_plot(t_test, y_pred_test, 'orange',
                                                f"<b>Prediction</b><br>rmse: {rmse_test:.3f}<br>max err: {max_err_test:.2f}"))

    equation_str = "No equation to display"
    if model_type == 'LR' and len(toggle_value) > 0:
        equation_str = generate_equation_str(coef, intercept, toggle_value)

    return create_figure(train_data_list, train_fig_title), create_figure(test_data_list, test_fig_title), equation_str


def update_graphs_raw(toggle_value, start_hour, end_hour, df, test_date_str,
                                  train_date_start_str, train_date_end_str,
                                  train_fig_title, test_fig_title, options):
    test_date = datetime.strptime(test_date_str, '%Y/%m/%d')
    train_df, test_df = split_train_test(df, test_date, train_date_start_str, train_date_end_str)

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    t_train = trained_df_filtered['timestamp'].values
    t_test = test_df_filtered['timestamp'].values
    # Create a mapping dictionary from value to label
    value_to_label = {option['value']: option['label'] for option in options}
    # Build column names list for X based on toggle_value, avoiding specific ref columns
    col_names = [f'smoothed_{value_to_label[val]}' for val in toggle_value]
    train_data_list = []
    test_data_list = []
    y_label = 'Temperature (°C)'
    for col in col_names:
        # make a random color
        random_color = '#' + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        col_name = col.replace('smoothed_', '')
        col_ref = 'smoothed_' + col.replace('smoothed', 'ref')
        y_train = trained_df_filtered[col].values
        y_ref_train = trained_df_filtered[col_ref].values
        y_train = y_train + y_ref_train
        train_data_list.append(generate_scatter_plot(t_train, y_train, random_color, f'<b>{col_name}</b>'))

        y_test = test_df_filtered[col].values
        y_ref_test = test_df_filtered[col_ref].values
        y_test = y_test + y_ref_test
        test_data_list.append(generate_scatter_plot(t_test, y_test, random_color, f'<b>{col_name}</b>'))

    return create_figure(train_data_list, train_fig_title, y_label),\
            create_figure(test_data_list, test_fig_title, y_label)
