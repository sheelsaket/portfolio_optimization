#Import the python libraries
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pickle
import copy
from datetime import date
import pathlib
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State #ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import random
import plotly

app = dash.Dash(__name__)
server = app.server

app.title = "Portfolio Optimization | Sheel Saket"

total_clicks = 0

# layout = dict(
#     autosize=True,
#     automargin=True,
#     margin=dict(l=30, r=30, b=20, t=40),
#     hovermode="closest",
#     plot_bgcolor="#F9F9F9",
#     paper_bgcolor="#F9F9F9",
#     legend=dict(font=dict(size=10), orientation="h"),
#     title="Satellite Overview",
# #     mapbox=dict(
# #         accesstoken=mapbox_access_token,
# #         style="light",
# #         center=dict(lon=-78.05, lat=42.54),
# #         zoom=7,
# #     ),
# )

app.layout = html.Div(
    [
        #Header designing:
        
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [ html.A(
                        html.Img(
                            src=app.get_asset_url("artificialcodernew.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        ), href="https://www.youtube.com/channel/UCAzQQ9z1v-5JZAhBVmHTh_A/featured")
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Portfolio Optimization",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Based on Historical Data Optimize your investments", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Sheel Saket", id="sheel-linkedin"),
                            href="https://www.linkedin.com/in/sheelsaket/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        
        #Input box
        html.Div(
            [
                html.Div(
                    [
                        
                        html.P(
                            "Chose the Time Period to base the optimization",
                            className="date-range",
                        ),
                        dcc.DatePickerRange(
                            id='my-date-picker-range',
                            start_date = date.today().replace(year = date.today().year -1),
                            end_date=date.today(),
                            display_format='MMM Do, YY',
                            start_date_placeholder_text='MMM Do, YY'
                        ),
                        
                        html.P(
                            "\n\n\n\nEnter the Ticker Symbols of Stocks you want to purchase:",
                            className="ticker-labels",
                        ),
                        dcc.Textarea(
                        id='textarea-state-example',
                        value='AMD, NFLX, AAPL, GOOG, SPY',
                        style={'width': '100%', 'height': 100},
                        ),
                        
                        html.P(
                            "\n\nEnter the total Amount you want to invest:\n\n",
                            className="invest-amount",
                        ),
                        
                        html.Div([
                        dcc.Textarea(
                        id='investment',
                        value='$15000',
                        style={'width': '100%', 'height': 25},
                        ),
                            html.Button('Submit', id='textarea-state-example-button', n_clicks=0),
                            html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre-line'})
                        ]),
                        
                    
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                
                #output tiles on top
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="well_text"), html.P("Expected Annual Returns")],
                                    id="wells",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="gasText"), html.P("Annual volatility")],
                                    id="gas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="oilText"), html.P("Sharpe Ratio")],
                                    id="oil",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="waterText"), html.P("Sentiment")],
                                    id="water",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="hor_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        
        
        #Output Graphs
        
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="individual_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
#         html.Div(
#             [
#                 html.Div(
#                     [dcc.Graph(id="pie_graph")],
#                     className="pretty_container seven columns",
#                 ),
#                 html.Div(
#                     [dcc.Graph(id="aggregate_graph")],
#                     className="pretty_container five columns",
#                 ),
#             ],
#             className="row flex-display",
#         ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


#helper functions

def remove_negative_cleaned_wts(cleaned_weights):
    for key in cleaned_weights.keys():
        if cleaned_weights[key]<0:
            cleaned_weights[key] =0
    return cleaned_weights

def run_optimization_engine(assets, stockStartDate, stockEndDate, investment_amount):
    weights = np.array([0.25]*len(assets.split(", ")))
    # Create a dataframe to store the adjusted close price of the stocks
    df = pd.DataFrame()

    # Store the adjusted close price of the sock into the df
    for stock in assets.split(', '):
        df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate,end=stockEndDate)['Adj Close']

    returns = df.pct_change()
    # Create and show the annualized covariance matrix
    cov_matrix_annual = returns.cov()*252
    # Calculate the portfolio variance
    port_variance = np.dot(weights.T,np.dot(cov_matrix_annual,weights))
    # Calculate the portfolio volatility aka standard deviation
    port_volatility = np.sqrt(port_variance)
    # Calculate the annual portfolio return
    portfolio_simple_annual_return = np.sum(returns.mean()*weights)*252

    # Portfolio Optimization

    # Calculate the expected returns and the annualized sample covariance matrix of asset returns
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximum sharpe ratio
    ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
    ef.add_constraint(lambda w: sum(w) == 1)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights() 
    
    ef.portfolio_performance(verbose=True)

    cleaned_weights = remove_negative_cleaned_wts(cleaned_weights)
#     print(cleaned_weights)
    
    # Get the discrete allocation of each share per stock
    
    latest_prices = get_latest_prices(df)
#     print(latest_prices)
    weights = cleaned_weights
    da = DiscreteAllocation(weights,latest_prices,total_portfolio_value = investment_amount)

    allocation,leftover = da.lp_portfolio()
#     print('Discrete allocation: ',allocation)
#     print('Funds remaining: ${:.2f}'.format(leftover))
    
    return df, cleaned_weights, ef.portfolio_performance(verbose=True), allocation, leftover, latest_prices


def get_sentiments(ann_ret):
    if ann_ret >=100:
        return "Excellent"
    elif (ann_ret >=50) & (ann_ret <100):
        return "Very Good"
    elif (ann_ret >=30) & ((ann_ret <50)):
        return "Good"
    elif (ann_ret >=10) & ((ann_ret <30)):
        return "Average"
    elif (ann_ret >=0) & ((ann_ret <10)):
        return "Below Average"
    elif (ann_ret <0):
        return "poor"




#design callbacks

@app.callback(
    [Output("main_graph", "figure"),
     Output("hor_graph", "figure"),
     Output("individual_graph", "figure"),
     Output("well_text", "children"),
     Output("gasText", "children"),
     Output("oilText", "children"),
     Output("waterText", "children")
    ],
    [
        Input("textarea-state-example", "value"),
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date'),
        Input("investment", "value"),
        Input('textarea-state-example-button', 'n_clicks')
    ]
#     ,
    
#     [State("lock_selector", "value"), State("main_graph", "relayoutData")],
)


def generate_outputs(
    assets, stockStartDate, stockEndDate, investment_amount, n_clicks
):
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS
    i = 0
    colors = {}
    for asset in assets.split(", "):
        colors.update({asset:cols[i]})
        i = i+ 1

    
    global total_clicks
    print(n_clicks, total_clicks)
    if n_clicks is None:
        raise PreventUpdate
    
    elif n_clicks >= total_clicks:
        try:
            investment_amount = int(investment_amount.replace("$", ""))
            print(assets, investment_amount, stockStartDate, stockEndDate)
            df_func, cleaned_weights, portfolio_performance, allocation, leftover, latest_prices = run_optimization_engine(assets, stockStartDate, stockEndDate, investment_amount)
            print('return successful')
#             pd.options.plotting.backend = "plotly"
#             print(df_func)
    #         df = pd.DataFrame(dict(a=[1,3,2], b=[3,2,1]))
#             fig = df_func.plot(title="Performance of the Selected Stocks", template="simple_white", 
#                           labels=dict(index="time", value="Price", variable="option"))
#             fig.update_yaxes(tickprefix="$")
# #             fig.show()
            
            fig = go.Figure()
            for key in df_func.columns:
                fig.add_trace(go.Scatter(x=df_func.index, y=df_func[key],
                    mode='lines',
                    name=key, marker=dict(
                        color=str(colors[key]),
            #                         line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                    )),
                    )
            fig.update_layout(title="Performance of the Selected Stocks")
            # fig.show()
            print('Fig successful')
            
            invested = {}
            for key in allocation.keys():
                invested.update({key: round(allocation[key]*latest_prices[key],2)})

            fig2 = go.Figure()
            for key in invested.keys():
                fig2.add_trace(go.Bar(
                    y=[''],
                    x=[invested[key]],
                    name=str(key),
                    orientation='h',
                    marker=dict(
                        color=str(colors[key]),
#                         line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                    )
                ))
            fig2.update_layout(barmode='stack', title='This is how you should split your investment')
            # fig2.show()
            print('Fig2 successful')

            allo_df = pd.Series(allocation).reset_index().rename(columns = {'index': 'stocks', 0:'shares_num'})
            
            
            fig3 = go.Figure()
            for index, key in enumerate(allo_df.stocks):

                fig3.add_trace(go.Bar(x=[key], y=[allo_df['shares_num'][index]],
                    name=key, marker=dict(
                        color=str(colors[key]),
            #                         line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                    )),
                    )
            fig3.update_layout(title="No of Shares to Purchase")
            # fig3.show()
            print('Fig3 successful')

            total_clicks = total_clicks + 1
            return fig, fig2, fig3, str(round(portfolio_performance[0]*100,2)) + "%", str(round(portfolio_performance[1]*100,2)) + "%", str(round(portfolio_performance[2],2)), get_sentiments(portfolio_performance[0]*100)

        except:
            return False
      

if __name__ == '__main__':
    app.run_server(debug=False)