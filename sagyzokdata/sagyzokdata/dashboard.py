import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Загрузка результатов или KPI
data = pd.read_csv('data/processed_data.csv')

fig = px.line(data, x='date', y='future_spend', title='Прогноз будущих расходов')

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Финансовая аналитика"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)