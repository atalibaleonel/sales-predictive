import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

# Generate the fictional dataset
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31')
sales = 200 + (np.sin(np.linspace(0, 3.14 * 2, len(dates))) * 50) + np.random.normal(0, 20, len(dates))

data = pd.DataFrame({'date': dates, 'sales': sales})
data['sales'] = data['sales'].astype(int)

# Create features based on date
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['year'] = data['date'].dt.year

# Select features and target
features = ['month', 'quarter', 'year']
target = 'sales'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression and random forest models
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Prepare data for graphs
data['predicted_sales_lr'] = model_lr.predict(X)
data['predicted_sales_rf'] = model_rf.predict(X)
data_test = data.iloc[X_test.index]

# Initialize the Dash app with the Cyborg Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Define the layout of the dashboard
app.layout = dbc.Container(fluid=True, children=[
    dbc.NavbarSimple(
        brand="Sales Predictive Analysis Dashboard",
        color="dark",
        dark=True,
        className="mb-4"
    ),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Settings", className="card-title"),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': str(year), 'value': year} for year in sorted(data['year'].unique())],
                        value=data['year'].min(),
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.RangeSlider(
                        id='date-slider',
                        min=0,
                        max=len(data) - 1,
                        value=[0, len(data) - 1],
                        marks={i: str(date.date()) for i, date in enumerate(data['date'][::30])},
                        className="mb-3",
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dcc.RadioItems(
                        id='model-radio',
                        options=[
                            {'label': 'Linear Regression', 'value': 'LR'},
                            {'label': 'Random Forest', 'value': 'RF'}
                        ],
                        value='LR',
                        className="mb-3"
                    )
                ])
            ])
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dcc.Graph(id='sales-graph'),
            dcc.Graph(id='error-histogram'),
        ], width=12, lg=9)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='model-evaluation', className="mt-4"))
    ])
])

# Define callback to update the graphs and model evaluation
@app.callback(
    [Output('sales-graph', 'figure'),
     Output('error-histogram', 'figure'),
     Output('model-evaluation', 'children')],
    [Input('year-dropdown', 'value'),
     Input('date-slider', 'value'),
     Input('model-radio', 'value')]
)
def update_graph(selected_year, date_range, selected_model):
    filtered_data = data[(data['year'] == selected_year) & (data.index >= date_range[0]) & (data.index <= date_range[1])]
    filtered_test_data = data_test[(data_test['year'] == selected_year) & (data_test.index >= date_range[0]) & (data_test.index <= date_range[1])]
    
    # Select model based on user choice
    if selected_model == 'LR':
        model_name = 'Linear Regression'
        y_pred = filtered_test_data['predicted_sales_lr']
    else:
        model_name = 'Random Forest'
        y_pred = filtered_test_data['predicted_sales_rf']
    
    # Sales graph
    sales_graph = {
        'data': [
            go.Scatter(
                x=filtered_data['date'],
                y=filtered_data['sales'],
                mode='lines',
                name='Actual Sales'
            ),
            go.Scatter(
                x=filtered_test_data['date'],
                y=y_pred,
                mode='lines',
                name=f'{model_name} Predictions',
                line={'dash': 'dash'}
            )
        ],
        'layout': {
            'title': 'Sales Prediction',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Sales'},
            'template': 'plotly_dark'
        }
    }
    
    # Error histogram
    errors = filtered_test_data['sales'] - y_pred
    
    error_histogram = {
        'data': [
            go.Histogram(
                x=errors,
                name=f'{model_name} Errors',
                opacity=0.75
            )
        ],
        'layout': {
            'title': 'Prediction Error Distribution',
            'xaxis': {'title': 'Error'},
            'barmode': 'overlay',
            'template': 'plotly_dark'
        }
    }
    
    # Model evaluation
    mse = mean_squared_error(filtered_test_data['sales'], y_pred)
    r2 = r2_score(filtered_test_data['sales'], y_pred)
    
    model_evaluation = [
        html.H5(f'Model Evaluation: {model_name}', className="mt-4"),
        html.P(f'Mean Squared Error: {mse:.2f}'),
        html.P(f'R² Score: {r2:.2f}')
    ]
    
    return sales_graph, error_histogram, model_evaluation

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
