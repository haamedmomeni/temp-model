import plotly.graph_objs as go

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
