import dash
from dash import html, dcc
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

# Global variable to store dashboard data
global_store = None

def create_dashboard(server):
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ]
    )

    dash_app.layout = html.Div([
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='precision-recall-f1', style={'height': '500px'})
            ], className='col-md-6'),
            html.Div([
                dcc.Graph(id='support-accuracy', style={'height': '500px'})
            ], className='col-md-6')
        ], className='row mb-4'),
        html.Div([
            html.Div([
                dcc.Graph(id='confusion-matrix', style={'height': '500px'})
            ], className='col-md-6'),
            html.Div([
                dcc.Graph(id='metrics-comparison', style={'height': '500px'})
            ], className='col-md-6')
        ], className='row')
    ], className='container-fluid p-4', style={'minHeight': '1000px'})

    @dash_app.callback(
        [Output('precision-recall-f1', 'figure'),
         Output('support-accuracy', 'figure'),
         Output('confusion-matrix', 'figure'),
         Output('metrics-comparison', 'figure')],
        Input('interval-component', 'n_intervals')
    )
    def update_graphs(n):
        global global_store
        data = global_store
        
        if not data:
            return [{} for _ in range(4)]

        metrics_df = pd.DataFrame(data['class_metrics'])
        
        # Precision, Recall, F1 Score by Category
        fig1 = px.bar(metrics_df, 
                     x='category',
                     y=['precision', 'recall', 'f1-score'],
                     title='Precision, Recall, and F1-Score by Attack Category',
                     barmode='group',
                     color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c'])
        fig1.update_layout(
            xaxis_tickangle=-45,
            legend_title_text='Metric',
            height=500
        )
        
        # Support vs Accuracy
        fig2 = px.bar(metrics_df,
                     x='category',
                     y=['support'],
                     title='Support Count and Accuracy by Attack Category',
                     color_discrete_sequence=['#34495e'])
        fig2.add_scatter(x=metrics_df['category'],
                        y=metrics_df['accuracy'],
                        name='Accuracy',
                        yaxis='y2',
                        line=dict(color='#e74c3c', width=3))
        fig2.update_layout(
            xaxis_tickangle=-45,
            yaxis2=dict(
                title='Accuracy',
                overlaying='y',
                side='right',
                tickformat='.0%'
            ),
            yaxis_title='Support Count',
            height=500
        )
        
        # Confusion Matrix Heatmap
        fig3 = px.imshow(data['confusion_matrix'],
                        labels=dict(x='Predicted', y='True'),
                        x=metrics_df['category'],
                        y=metrics_df['category'],
                        title='Confusion Matrix',
                        color_continuous_scale='RdYlBu_r',
                        aspect='auto')
        fig3.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        # Macro vs Weighted Metrics
        macro_weighted = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Macro Avg': [data['macro_avg']['precision'],
                         data['macro_avg']['recall'],
                         data['macro_avg']['f1-score']],
            'Weighted Avg': [data['weighted_avg']['precision'],
                           data['weighted_avg']['recall'],
                           data['weighted_avg']['f1-score']]
        })
        fig4 = px.bar(macro_weighted,
                     x='Metric',
                     y=['Macro Avg', 'Weighted Avg'],
                     title='Macro vs Weighted Average Metrics',
                     barmode='group',
                     color_discrete_sequence=['#3498db', '#e74c3c'])
        fig4.update_layout(
            legend_title_text='Average Type',
            yaxis_title='Score',
            height=500
        )

        return fig1, fig2, fig3, fig4

    return dash_app 