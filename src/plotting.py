import plotly.graph_objs as go
from dash import dcc, html
from data_processing import filter_dataframe_by_hours, fit_and_predict_training_data, split_train_test, \
    predict_test_data
from datetime import datetime
import pandas as pd


def generate_scatter_plot(x_array, y_array, color, name):
    return go.Scatter(
        x=x_array,
        y=y_array,
        mode='markers',
        marker=dict(size=5, color=color),
        name=name
    )


# Function to create a figure dictionary
def create_figure(data_list, title):
    return {
        'data': data_list,
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Timestamp'},
            yaxis={'title': 'Smoothed Difference 1-3'}
        )
    }


# Function to generate equation string
def generate_equation_str(coef, intercept, toggle_value):
    coef = coef.flatten().tolist()
    equation_terms = [f"{'+' if c > 0 else ''}{c:.2f} {toggle_value[i][5:].lower()}" for i, c in enumerate(coef)]
    return f"Equation: {intercept[0]:.2f}  {' '.join(equation_terms)}"


def create_graph_div(title, graph_id, data, x_axis_title='Timestamp', y_axis_title=''):
    return html.Div([
        html.H4(title),
        dcc.Graph(
            id=graph_id,
            figure={
                'data': [data],
                'layout': go.Layout(
                    title=title,
                    xaxis={'title': x_axis_title},
                    yaxis={'title': y_axis_title},
                    paper_bgcolor='#2D2D2D',
                    plot_bgcolor='#2D2D2D',
                )
            }
        )
    ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '10px'})


def update_graphs_and_predictions(model_type, toggle_value, start_hour, end_hour, df, test_date_str, diff_col,
                                  train_fig_title, test_fig_title, options):
    # Convert test_date_str to datetime
    test_date = datetime.strptime(test_date_str, '%Y/%m/%d')

    # Split data based on selected test date
    train_df, test_df = split_train_test(df, test_date)
    # train_df, test_df = smooth_data(train_df), smooth_data(test_df)

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    train_data_list = [generate_scatter_plot(trained_df_filtered['timestamp'].values,
                                             trained_df_filtered[diff_col].values,
                                             'blue', 'observation')]
    test_data_list = [generate_scatter_plot(test_df_filtered['timestamp'].values,
                                            test_df_filtered[diff_col].values,
                                            'green', 'observation')]

    equation_str = "No equation to display"

    if toggle_value:
        model, y_pred_train, rmse_train, max_err_train, coef, intercept = (
            fit_and_predict_training_data(model_type, toggle_value, trained_df_filtered, diff_col, options))
        y_pred_test, rmse_test, max_err_test = (
            predict_test_data(model,toggle_value, test_df_filtered, diff_col, options))
            # fit_and_predict_training_data(model_type, toggle_value, test_df_filtered, diff_col, options))

        train_data_list.append(generate_scatter_plot(
            trained_df_filtered['timestamp'].values, y_pred_train, 'red',
            f"prediction<br>rmse: {rmse_train:.3f}<br>max err: {max_err_train:.2f}"))
        test_data_list.append(generate_scatter_plot(
            test_df_filtered['timestamp'].values, y_pred_test, 'orange',
            f"prediction<br>rmse: {rmse_test:.3f}<br>max err: {max_err_test:.2f}"))

        if model_type == 'LR':
            equation_str = generate_equation_str(coef, intercept, toggle_value)

    return (create_figure(train_data_list, train_fig_title),
            create_figure(test_data_list, test_fig_title),
            equation_str)


# TODO: Implement this function
def update_error_graphs_list(model_type, toggle_value, start_hour, end_hour, df, test_date_str, options):
    diff_col = 'smoothed_diff_1_3'
    diff_col2 = 'smoothed_diff_1_3'

    # Convert test_date_str to datetime
    test_date = datetime.strptime(test_date_str, '%Y/%m/%d')

    # Split data based on selected test date
    train_df, test_df = split_train_test(df, test_date)

    trained_df_filtered = filter_dataframe_by_hours(train_df, start_hour, end_hour)
    test_df_filtered = filter_dataframe_by_hours(test_df, start_hour, end_hour)

    train_data_list = [generate_scatter_plot(trained_df_filtered['timestamp'].values,
                                             trained_df_filtered[diff_col].values,
                                             'blue', 'observation')]
    test_data_list = [generate_scatter_plot(test_df_filtered['timestamp'].values,
                                            test_df_filtered[diff_col].values,
                                            'green', 'observation')]

    if toggle_value:
        model, y_pred_train, rmse_train, max_err_train, coef, intercept = (
            fit_and_predict_training_data(model_type, toggle_value, trained_df_filtered, diff_col, options))
        y_pred_test, rmse_test, max_err_test = (
            predict_test_data(model,toggle_value, test_df_filtered, diff_col, options))
            # fit_and_predict_training_data(model_type, toggle_value, test_df_filtered, diff_col, options))

        train_data_list.append(generate_scatter_plot(
            trained_df_filtered['timestamp'].values, y_pred_train, 'red',
            f"prediction<br>rmse: {rmse_train:.3f}<br>max err: {max_err_train:.2f}"))
        test_data_list.append(generate_scatter_plot(
            test_df_filtered['timestamp'].values, y_pred_test, 'orange',
            f"prediction<br>rmse: {rmse_test:.3f}<br>max err: {max_err_test:.2f}"))


    return (create_figure(train_data_list, 'Title1'),
            create_figure(test_data_list, 'Title2')
            )
