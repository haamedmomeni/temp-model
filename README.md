# Predictive Model Dashboard

## Overview

This repository hosts the code for a dynamic HTML dashboard designed to visualize and evaluate the performance of predictive models. The dashboard is interactive and publicly accessible at [temp-model-dash.onrender.com](https://temp-model-dash.onrender.com/).

### Screenshot
![Predictive Model Dashboard Screenshot](/src/screenshot.png "Dashboard Overview")

## Key Features

- **Interactive Charts**: Explore predictive model accuracy with charts that compare observed data against model predictions.
- **Custom Data Ranges**: Select specific data ranges for detailed analysis by adjusting the start and end hours.
- **Real-time Data Processing**: View the latest results with data dynamically updated to reflect current metrics.
- **Responsive Design**: Access the dashboard on any device, optimized for both desktop and mobile viewing.

### Graphical Representations Include:
- Training Data (Differential Y motion)
- Training Data (Differential X motion)
- Testing Data (Differential Y motion)
- Testing Data (Differential X motion)

Each chart allows users to visualize the smooth difference between various data columns over a timestamped range.

### Data Selection
Users can select a range of data they wish to analyze by specifying the starting and ending hour, which dynamically updates the charts.

## Local Setup
To set up this dashboard locally, clone the repository to your machine and follow the instructions below:

```bash
$ git clone https://github.com/haamedmomeni/temp-model.git
$ cd temp-model
$ pip install -r requirements.txt
$ cd src
$ python3 app.py
```

## How It Works

The Temp-Model Dashboard retrieves data from various sources and processes it to present in an easily digestible format. It utilizes web technologies such as HTML, CSS, and JavaScript to create a dynamic and responsive user interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
