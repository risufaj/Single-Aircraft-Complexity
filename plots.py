from time import time
#from turtle import width
import plotly.graph_objects as go
import networkx as nx
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from dash_table import DataTable
import pickle

'''
VISUALIZE ONLY CRIDA SECTOR LECMDP
'''
df = pd.read_csv("sector_boundary.csv")
lats = df['lat'].to_numpy()
lons = df['lng'].to_numpy()
#with open('sectors.pickle', 'rb') as pickle_file:
#    content = pickle.load(pickle_file)
#lats,lons = content['LECMPAU'][0].exterior.coords.xy
bounding_box = [(min(lats),max(lats)),(min(lons),max(lons))]

def coord_to_path(lats,lons):
    assert len(lats) == len(lons)
    path = ""
    for i, _ in enumerate(lats):
        if i == 0:
            path += f"M {lats[i]},{lons[i]} "
        elif i == len(lats) - 1:
            path += f"L{lats[i]},{lons[i]}, Z"
        else:
            path += f"L{lats[i]},{lons[i]} "
    return path
 


def conflicts(T,graphs,comps,axis_range=None):
    if axis_range is None:
        axis_range = [0,6,0,6]
    G = graphs[0]
    comp = comps[0]
    comp.individual_strength()
    frames = []
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False,"showticklabels":False}
    fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False,"showticklabels":False}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["showlegend"] = False
    fig_dict["layout"]["hovermode"] = 'closest'
    fig_dict["layout"]["title"] = {"text":"Conflict Information", "x":0.05, "xanchor":"left"}
    fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["margin"] = {"b":20, "l":5, "r":5,"t":40}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2000, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 1000,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    fig_dict['layout']['shapes'] = [
        {
            "type":"path",
            "path":coord_to_path(lons,lats),
            "line_color":"LightSeaGreen"

        }
    ]
    

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Timestep:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }



    #MAKE INITIAL DATA
    edge_dict = {}
    node_dict = {}

    edge_x = []
    edge_y = []
    

    for edge in G.edges():
        
        x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_dict = {
            "x": edge_x,
            "y": edge_y,
            "mode": "lines",
            "line": {"width":0.5,"color":"#888"}
        }
    fig_dict["data"].append(edge_dict)

    

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['lon'], G.nodes[node]['lat']#G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_dict = {
            "x": node_x,
            "y": node_y,
            "mode": "markers",
            "hoverinfo": "text",
            "marker": {"showscale":True, "reversescale":True,"color":[],"size":10,"colorscale":"Agsunset","cmin":0,
                    "colorbar":{"thickness":15,"title":"Strength","xanchor":"left","titleside":"right"}
                    },
            "line_width":2
    }

    

    node_contributions = []
    node_text = []
    #print(comp.individual_contributions)
    for ac, contribution in comp.conf_information.items():
        node_contributions.append(contribution)
        node_text.append(f"{ac} strength {round(contribution,2)*100}%")
    #print(node_contributions, comp.individual_contributions.items())
    node_dict["marker"]["color"] = node_contributions
    node_dict["text"] = node_text
    fig_dict["data"].append(node_dict)



    for idx, t in enumerate(T):
        G = graphs[idx]
        comp = comps[idx]
        comp.individual_strength()
        frame = {"data":[], "name": str(t)}
        
        edge_dict = {}
        node_dict = {}

        edge_x = []
        edge_y = []
        
        #print(comp.individual_contributions)
        for edge in G.edges():
            
           
            x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        edge_dict = {
                "x": edge_x,
                "y": edge_y,
                "mode": "lines",
                "line": {"width":0.5,"color":"#888"}
            }
        frame["data"].append(edge_dict)
        
        
        
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['lon'], G.nodes[node]['lat']#G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_dict = {
                "x": node_x,
                "y": node_y,
                "mode": "markers",
                "hoverinfo": "text",
                "marker": {"showscale":True, "reversescale":True,"color":[],"size":10,"colorscale":"Agsunset","cmin":0,
                        "colorbar":{"thickness":15,"title":"Strength","xanchor":"left","titleside":"right"}
                        },
                "line_width":2
        }

        
        
        node_contributions = []
        node_text = []
        for ac, contribution in comp.conf_information.items():
            node_contributions.append(contribution)
            node_text.append(f"{ac} strength {round(contribution,2)*100}%")
        
        node_dict["marker"]["color"] = node_contributions
        node_dict["text"] = node_text
        
        
        
        frame["data"].append(node_dict)
        
        fig_dict["frames"].append(frame)
        
        slider_step = {"args": [
            [t],
            {"frame": {"duration": 3000, "redraw": True},
            "mode": "immediate",
            "transition": {"duration": 3000}}
        
        
        ],
            "label": t,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig.update_yaxes(range=[bounding_box[0][0], bounding_box[0][1]])
    fig.update_xaxes(range=[bounding_box[1][0], bounding_box[1][1]])
    #fig.update_xaxes(range=[axis_range[0], axis_range[1]])
    #fig.update_yaxes(range=[axis_range[2], axis_range[3]])
    #fig.show()
    return fig


def animation(T,graphs,comps, axis_range=None):
    if axis_range is None:
        axis_range = [0,6,0,6]
    G = graphs[0]
    comp = comps[0]
    comp.individual_strength()
    frames = []
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False,"showticklabels":False}
    fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False,"showticklabels":False}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["showlegend"] = False
    fig_dict["layout"]["hovermode"] = 'closest'
    fig_dict["layout"]["title"] = {"text":"Complexity Information", "x":0.05, "xanchor":"left"}
    fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["margin"] = {"b":20, "l":5, "r":5,"t":40}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2000, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 1000,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    fig_dict['layout']['shapes'] = [
        {
            "type":"path",
            "path":coord_to_path(lons,lats),
            "line_color":"LightSeaGreen"

        }
    ]
    

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Timestep:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }



    #MAKE INITIAL DATA
    edge_dict = {}
    node_dict = {}

    edge_x = []
    edge_y = []
    edge_x_c = []
    edge_y_c = []

    for edge in G.edges():
        if comp.is_complex[edge[0]] or comp.is_complex[edge[1]]:
            x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']

            edge_x_c.append(x0)
            edge_x_c.append(x1)
            edge_x_c.append(None)
            edge_y_c.append(y0)
            edge_y_c.append(y1)
            edge_y_c.append(None)
        else:
            x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']

            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
    if len(edge_x) > 0:
        edge_dict = {
                "x": edge_x,
                "y": edge_y,
                "mode": "lines",
                "line": {"width":0.5,"color":"#888"}
            }
        
    fig_dict["data"].append(edge_dict)
    edge_dict_c = {}
    if len(edge_x_c) > 0:
        edge_dict_c = {
                "x": edge_x_c,
                "y": edge_y_c,
                "mode": "lines",
                "line":{"width":0.8,"color":"red"}
            }
    fig_dict['data'].append(edge_dict_c)

    

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['lon'], G.nodes[node]['lat']#G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_dict = {
            "x": node_x,
            "y": node_y,
            "mode": "markers",
            "hoverinfo": "text",
            "marker": {"showscale":True, "reversescale":True,"color":[],"size":10,"colorscale":"Agsunset","cmin":0,
                    "colorbar":{"thickness":15,"title":"Complexity","xanchor":"left","titleside":"right"}
                    },
            "line_width":2
    }

    

    node_contributions = []
    node_text = []
    #print(comp.individual_contributions)
    for ac, contribution in comp.individual_contributions.items():
        node_contributions.append(contribution)
        node_text.append(f"{ac} contributes {round(contribution,2)*100}%")
    #print(node_contributions, comp.individual_contributions.items())
    node_dict["marker"]["color"] = node_contributions
    node_dict["text"] = node_text
    fig_dict["data"].append(node_dict)



    for idx, t in enumerate(T):
        G = graphs[idx]
        comp = comps[idx]
        comp.individual_strength()
        frame = {"data":[], "name": str(t)}
        
        edge_dict = {}
        node_dict = {}
        edge_dict_c = {}

        edge_x = []
        edge_y = []
        edge_x_c = []
        edge_y_c = []
        #print(comp.individual_contributions)
        for edge in G.edges():
            if comp.is_complex[edge[0]] or comp.is_complex[edge[1]]:
                x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
                x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']
                edge_x_c.append(x0)
                edge_x_c.append(x1)
                edge_x_c.append(None)
                edge_y_c.append(y0)
                edge_y_c.append(y1)
                edge_y_c.append(None)
            else:

                x0, y0 = G.nodes[edge[0]]['lon'], G.nodes[edge[0]]['lat']#G.nodes[edge[0]]['pos']
                x1, y1 = G.nodes[edge[1]]['lon'], G.nodes[edge[1]]['lat']#G.nodes[edge[1]]['pos']
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        if len(edge_x) > 0:
            edge_dict = {
                    "x": edge_x,
                    "y": edge_y,
                    "mode": "lines",
                    "line": {"width":0.5,"color":"#888"}
                }
            #
        frame["data"].append(edge_dict)
        if len(edge_x_c) > 0:
            edge_dict_c = {
                "x": edge_x_c,
                "y": edge_y_c,
                "mode": "lines",
                "line": {"width":0.5,"color":"red"}
            }
        frame["data"].append(edge_dict_c)
        
            
        
        
        
        
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['lon'], G.nodes[node]['lat']#G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_dict = {
                "x": node_x,
                "y": node_y,
                "mode": "markers",
                "hoverinfo": "text",
                "marker": {"showscale":True, "reversescale":True,"color":[],"size":10,"colorscale":"Agsunset","cmin":0,
                        "colorbar":{"thickness":15,"title":"Complexity","xanchor":"left","titleside":"right"}
                        },
                "line_width":2
        }

        

        
        
        node_contributions = []
        node_text = []
        for ac, contribution in comp.individual_contributions.items():
            node_contributions.append(contribution)
            node_text.append(f"{ac} contributes {round(contribution,2)*100}%")
        
        node_dict["marker"]["color"] = node_contributions
        node_dict["text"] = node_text
        
        
        
        frame["data"].append(node_dict)
        
        fig_dict["frames"].append(frame)
        
        slider_step = {"args": [
            [t],
            {"frame": {"duration": 3000, "redraw": True},
            "mode": "immediate",
            "transition": {"duration": 3000}}
        
        
        ],
            "label": t,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig.update_yaxes(range=[bounding_box[0][0], bounding_box[0][1]])
    fig.update_xaxes(range=[bounding_box[1][0], bounding_box[1][1]])
    #fig.update_xaxes(range=[axis_range[0], axis_range[1]])
    #fig.update_yaxes(range=[axis_range[2], axis_range[3]])
    #fig.show()
    return fig


def table(final_communities):
    from collections import OrderedDict
    data = OrderedDict(
        [
            ("ID", []),
            ("All members",[]),
            ("Added",[]),
            ("Removed",[]),
            ("Start", []),
            ("End",[]),
            ("Avg. Contribution", [])
        ]
    )

    for community in final_communities:
        name = community.id
        data['ID'].append(community.id)
        all_members = list(set(community.added.keys()) | set(community.removed.keys()))
        all_members = [str(e) for e in all_members]
        members = "\n".join(all_members)
        data['All members'].append(members)
        added = ""
        for key,val in community.added.items():
            added += f"{key}:{val}\n"
        removed = ""
        #print(community.percentage)
        for key,val in community.removed.items():
            removed += f"{key}:{val}\n"
        data["Added"].append(added)
        data["Removed"].append(removed)
        data["Start"].append(community.start_time)
        data["End"].append(community.end_time)
        data["Avg. Contribution"].append(f"{round(np.mean(community.percentage),2)*100}%")

    df = pd.DataFrame(
    OrderedDict([(name, col_data ) for (name, col_data) in data.items()])
)
    return DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_action='none',
    style_table={'height': '400px', 'overflowY': 'auto'},
    style_cell={
 'whiteSpace': 'pre-line'
 }
)





'''
def table(final_communities):

    data_matrix = [['ID','All members','Added','Removed','Start time', 'End time',"Avg.<br>Contribution"]]
    for community in final_communities:
        name = community.id
        all_members = list(set(community.added.keys()) | set(community.removed.keys()))
        all_members = [str(e) for e in all_members]
        members = ",".join(all_members)
        added = ""
        for key,val in community.added.items():
            added += f"{key}:{val}<br>"
        removed = ""
        #print(community.percentage)
        for key,val in community.removed.items():
            removed += f"{key}:{val}<br>"
        data_matrix.append([name,members,added,removed,community.start_time,community.end_time,f"{round(np.mean(community.percentage),2)*100}%"])
    fig =  ff.create_table(data_matrix)
    fig.update_layout(
    title_text = 'Summary of complex communities',
    #autosize=True,
    #height = 400,
    #width=1000,
    margin = {'t':75, 'l':10,'r':10},
    
)
    #fig.show()
    return fig
'''

def heatmap(T,final_communities):
    y = []
    z = []
    for community in final_communities:
        y.append(f"Community {community.id}")
    y.append('Pool')

    timestep = T[1] - T[0] if len(T) > 1 else 1
    
    

    for community in final_communities:
        z_i = []
        for t in T:
            if t >= community.start_time and t < community.end_time:
                perc_idx = int((t - community.start_time)/timestep)
                z_i.append(round(community.percentage[perc_idx],2))
            else:
                z_i.append(None)
        z.append(z_i)
    z_pool = []
    for idx,t in enumerate(T):
        rest = [z_i[idx] for z_i in z]
        comm_sum = sum(x if x is not None else 0 for x in rest)
        z_pool.append(1.- comm_sum)
    
    z.append(z_pool)
        
    
    fig = go.Figure(data=go.Heatmap(

        z = z,
        x = [str(t) for t in T],
        y = y,
        hoverongaps = False,
        reversescale = True
    ))
    fig.update_layout(
    title_text = 'Heatmap of community contributions',
    #height = 400,
    margin = {'t':75, 'l':10,'r':10},
    
    )
    return fig
    

