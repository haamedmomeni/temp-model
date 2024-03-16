import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from data_processing import filter_dataframe_by_hours, fit_and_predict_training_data, split_train_test, \
    predict_test_data
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
def create_figure(data_list, title):
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
    coef = coef.flatten().tolist()
    equation_terms = [f"{'+' if c > 0 else ''}{c:.2f} {toggle_value[i][5:].lower()}" for i, c in enumerate(coef)]
    return f"Equation: {intercept[0]:.2f}  {' '.join(equation_terms)}"


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

    t_train = trained_df_filtered['timestamp'].values
    t_test = test_df_filtered['timestamp'].values

    col_ref = 'smoothed_refY' if diff_col == 'smoothed_difference_2_4' else 'smoothed_refX'

    y_train = trained_df_filtered[diff_col].values + trained_df_filtered[col_ref].values
    y_test = test_df_filtered[diff_col].values + test_df_filtered[col_ref].values

    y_train_std = np.std(y_train)
    y_test_std = np.std(y_test)

    train_data_list = [generate_scatter_plot(t_train, y_train, 'blue', f'<b>Observation</b><br>std: {y_train_std:.3f}')]
    test_data_list = [generate_scatter_plot(t_test, y_test, 'green', f'<b>Observation</b><br>std: {y_test_std:.3f}')]

    model, y_pred_train, rmse_train, max_err_train, coef, intercept = fit_and_predict_training_data(
        model_type, toggle_value, trained_df_filtered, diff_col, options)
    y_pred_test, rmse_test, max_err_test = predict_test_data(
        model, toggle_value, test_df_filtered, diff_col, options)

    y_pred_train += trained_df_filtered[col_ref].values
    y_pred_test += test_df_filtered[col_ref].values

    train_data_list.append(generate_scatter_plot(t_train, y_pred_train, 'red',
                                                 f"<b>Prediction</b><br>rmse: {rmse_train:.3f}<br>max err: {max_err_train:.2f}"))
    test_data_list.append(generate_scatter_plot(t_test, y_pred_test, 'orange',
                                                f"<b>Prediction</b><br>rmse: {rmse_test:.3f}<br>max err: {max_err_test:.2f}"))

    equation_str = "No equation to display"
    if model_type == 'LR' and len(toggle_value) > 1:
        equation_str = generate_equation_str(coef, intercept, toggle_value)

    return create_figure(train_data_list, train_fig_title), create_figure(test_data_list, test_fig_title), equation_str
