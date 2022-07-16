import base64
import dash
from dash import dcc#import dash_core_components as dcc
from dash import html#import dash_html_components as html
import pandas as pd
from plots import animation,table,heatmap,conflicts
from complexity_indicators import Complexity,temporal_granular_complexity,graph_from_data
import plotly.graph_objects as go
import networkx as nx
import plotly.figure_factory as ff
import numpy as np
import io
from dash.dependencies import Input, Output, State
import traceback
from dash_table import DataTable


global uploaded_T, uploaded_graphs, axis_range, final_communities, comps

def generate_random_graph():
    T = [0,10,20,30,40,50]
    graphs = []
    pos = {}

    #t0
    G = nx.Graph()
    G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    G.add_edge(2,3,weight=0.6)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(0,1),'lon':np.random.uniform(0,1)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)

    #t10
    G = nx.Graph()
    G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    G.add_edge(2,3,weight=0.6)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(1,2),'lon':np.random.uniform(1,2)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)

    #t20
    G = nx.Graph()
    G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    G.add_edge(2,3,weight=0.6)
    G.add_edge(1,5,weight=0.4)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(2,3),'lon':np.random.uniform(2,3)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)

    #t30
    G = nx.Graph()
    #G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    #G.add_edge(2,3,weight=0.6)
    G.add_edge(1,5,weight=0.4)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(3,4),'lon':np.random.uniform(3,4)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)

    #t40
    G = nx.Graph()
    #G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    #G.add_edge(2,3,weight=0.6)
    G.add_edge(1,5,weight=0.4)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(4,5),'lon':np.random.uniform(4,5)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)


    #t50
    G = nx.Graph()
    G.add_edge(1,2,weight=0.5)
    G.add_edge(1,3,weight=0.4)
    G.add_edge(1,4,weight=0.6)
    G.add_edge(2,3,weight=0.6)
    G.add_edge(1,5,weight=0.4)
    G.add_edge(3,4,weight=0.5)
    G.add_edge(7,8,weight=0.4)
    pos = {}
    for idx,node in enumerate(list(G.nodes)):
        pos[node] = {'lat':np.random.uniform(5,6),'lon':np.random.uniform(5,6)}
    nx.set_node_attributes(G, pos)
    graphs.append(G)
    return T,graphs

T, graphs = generate_random_graph()

final_communities, comps = temporal_granular_complexity(T,graphs)









meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1",
        }
    ]


external_stylesheets = [
    {
        "href":'https://codepen.io/chriddyp/pen/bWLwgP.css',
        #"href": "https://fonts.googleapis.com/css2?"
        #"family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
    
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Granular Airspace Complexity"
server = app.server
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="✈️", className="header-emoji"),
                html.H1(
                    children="Complexity", className="header-title"
                ),
                html.P(
                    children="Visualize granular complexity",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        dcc.Upload(id="upload-data",children=[html.Button('Upload file',className='button-three')] ),
        dcc.Input(id="min", type="number", placeholder="Min. default 5NM",debounce=True),
        dcc.Input(
            id="max", type="number",
            debounce=True, placeholder="Max. default 50NM"
        ),
        dcc.Input(
            id="cthresh", type="number",
            debounce=True, placeholder="Complexity threshold (%)"
            ),
         #html.Img(src='assets/ripple.gif'),
        #dcc.Download(id="download-data",children=[html.Button('Download Summary',className='button-three')] ),
        html.Div(
    [
        html.Button("Download Summary", id="btn_csv",className='button-three'),
        dcc.Download(id="download-dataframe-csv"),
    ]),

        html.Hr(),


        html.Div(
            id = "container",
            
            children=[
                dcc.Loading(
                    type="dot",
                    fullscreen=True,
                    children=[
                html.Div(
                    children=dcc.Graph(
                        id="animation",
                        config={"displayModeBar": False},
                        figure=animation(T,graphs,comps)
                    ),
                    className="card",
                ),

                html.Div(
                    children=dcc.Graph(
                        id="heatmap",
                        config={"displayModeBar":False},
                        figure=heatmap(T,final_communities)
                    ),
                    className='card',
                ),

                html.Div(
                    children=table(final_communities),id='summary',
                    #children=dcc.Graph(
                    #    id="summary",
                    #    config={"displayModeBar": False},
                    #    figure=table(final_communities)
                    #),
                    className="card",
                ),
                html.Div(
                    children = dcc.Graph(
                        id="conflicts",
                        config={"displayModeBar": False},
                        figure=conflicts(T,graphs,comps)
                    ),
                    className="card",
                )
            ],
                )
            ],
            className="wrapper",
        ),

        
    ]
)

def parse_contents(contents,filename,date,minimum,maximum,cthresh):
    global uploaded_T, uploaded_graphs, axis_range, final_communities, comps
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    #print(contents)
    #print(filename)
    #print(io.StringIO(decoded.decode('latin-1')))
    if contents is not  None: 
        
            
    #print(decoded)

        try:
            minimum = minimum if minimum is not None else 5
            maximum = maximum if maximum is not None else 50
            cthresh = cthresh/100 if cthresh is not None else 0.4
            #print("KETU")
            #print(f"MINIMUM {minimum} MAXIMUM {maximum}")
            #print("FILE AND VALUES")
            uploaded_T, uploaded_graphs, axis_range = graph_from_data(io.StringIO(decoded.decode('utf-8')),minimum,maximum)
            #print(uploaded_T)
            final_communities, comps = temporal_granular_complexity(uploaded_T,uploaded_graphs,cthresh)
        except Exception as e:
            #print(e)
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(traceback_str)
            return html.Div(['There was an error processing this file'])
        T, graphs = generate_random_graph()
        fig1 = animation(uploaded_T,uploaded_graphs, comps,axis_range)
        fig2 = table(final_communities)
        fig3 = heatmap(uploaded_T, final_communities)
        fig4 = conflicts(uploaded_T,uploaded_graphs,comps,axis_range)
        
        #print(fig1,fig2)
        return fig1, fig2, fig3,fig4,False,False,False
    else:
        if uploaded_T is not None and uploaded_graphs is not None:
            try:
                minimum = minimum if minimum is not None else 5
                maximum = maximum if maximum is not None else 50
                cthresh = cthresh/100 if cthresh is not None else 0.4
                print("KETU")
                print(f"MINIMUM {minimum} MAXIMUM {maximum}")
                print(" NO FILE AND VALUES")
                uploaded_T, uploaded_graphs, axis_range = graph_from_data(io.StringIO(decoded.decode('utf-8')),minimum,maximum)
                final_communities, comps = temporal_granular_complexity(uploaded_T,uploaded_graphs,cthresh)
            except Exception as e:
                print(e)
                return html.Div(['There was an error processing this file'])
            T, graphs = generate_random_graph()
            fig1 = animation(uploaded_T,uploaded_graphs, comps,axis_range)
            fig2 = table(final_communities)
            fig3 = heatmap(uploaded_T,final_communities)
            fig4 = conflicts(uploaded_T,uploaded_graphs,comps,axis_range)
            #print(fig1,fig2)
            
            return fig1, fig2, fig3, fig4,False,False,False


@app.callback(Output('animation','figure'), Output('summary','children'),Output('heatmap','figure'),
              Output('conflicts','figure'),Output("min","disabled"),Output("max","disabled"),Output("cthresh","disabled"),
              Input('upload-data', 'contents'),
              Input('min','value'),
              Input('max','value'),
              Input('cthresh','value'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content,minimum,maximum,cthresh,filename,date):
    #print(filename)
    if dash.callback_context.triggered[0]["prop_id"] == ".":
        return dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update
    print("UPDATING")
    #print(f"MINIMUM IS {minimum} MAXIMUM IS {maximum}")
    return parse_contents(content,filename,date,minimum, maximum,cthresh)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    Input('max','value'),
    Input('cthresh','value'),
    prevent_initial_call=True,
)
def download(n_clicks,max_val, cthresh):
    print(f"CALLER ID {dash.callback_context.triggered[0]}")
    if dash.callback_context.triggered[0]["prop_id"] != "btn_csv.n_clicks":
        return dash.no_update
    if n_clicks is not None:
        members = []
        duration = []
        contributions = []
        #print(final_communities)
        for community in final_communities:
            
            
            all_members = list(set(community.added.keys()) | set(community.removed.keys()))
            members.append(len(all_members))
            
            duration.append(community.end_time - community.start_time)
            contributions.append(np.mean(community.percentage))
        ac_contributions = []
        for comp in comps:
            node_contributions = []
            
            #print(comp.individual_contributions)
            for ac, contribution in comp.individual_contributions.items():
                node_contributions.append(contribution)
            ac_contributions.append(np.mean(node_contributions))


        print(f"LEN COMPS {len(comps)}")
        print(f" LEN AC CONT {len(ac_contributions)}")
        #print(contribution)
        df = pd.DataFrame({"Complexity_threshold":cthresh,"Interdependency_threshold":max_val,"No_communities": len(final_communities), "Members": [members], "Duration": [duration], "Contribution":[contributions], "Ac_contribution":[ac_contributions]})

        return dcc.send_data_frame(df.to_csv, f"nodelay_{max_val}_{cthresh}.csv")


if __name__ == "__main__":
    app.run_server(debug=True,port=9999)