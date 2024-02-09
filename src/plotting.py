import plotly.graph_objs as go
from dash import dcc, html
from data_processing import filter_dataframe_by_hours, fit_and_predict_linear_eq


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
    # equation_terms = [f"{c:.2f}*x{i + 1}" for i, c in enumerate(coef)]
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


def update_graphs_and_predictions(toggle_value, start_hour, end_hour, train_df, test_df, diff_col, train_fig_title, test_fig_title, options):
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
        y_pred_train, rmse_train, max_err_train, coef, intercept = fit_and_predict_linear_eq(toggle_value, trained_df_filtered, diff_col, options)
        y_pred_test, rmse_test, max_err_test, _, _ = fit_and_predict_linear_eq(toggle_value, test_df_filtered, diff_col, options)

        train_data_list.append(generate_scatter_plot(trained_df_filtered['timestamp'].values, y_pred_train, 'red',
                                                     f"prediction<br>rmse: {rmse_train:.2f}<br>max err: {max_err_train:.2f}"))
        test_data_list.append(generate_scatter_plot(test_df_filtered['timestamp'].values, y_pred_test, 'orange',
                                                    f"prediction<br>rmse: {rmse_test:.2f}<br>max err: {max_err_test:.2f}"))

        equation_str = generate_equation_str(coef, intercept, toggle_value)

    return (create_figure(train_data_list, train_fig_title),
            create_figure(test_data_list, test_fig_title),
            equation_str)