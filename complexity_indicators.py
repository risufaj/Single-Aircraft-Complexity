import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import pandas as pd
from math import*

# Constants
nm  = 1852.  # m       1 nautical mile
def kwikqdrdist_matrix(lata, lona, latb, lonb):
    """
    Distance function from bluesky
    Gives quick and dirty qdr[deg] and dist [nm] matrices
       from lat/lon vectors. (note: does not work well close to poles)
    """

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata.T)
    dlon    = np.radians(((lonb - lona.T)+ 180) % 360 - 180)
    cavelat = np.cos(np.radians(latb + lata.T) * 0.5)

    dangle  = np.sqrt(np.multiply(dlat, dlat) +
                      np.multiply(np.multiply(dlon, dlon),
                                  np.multiply(cavelat, cavelat)))
    dist    = re * dangle / nm

    

    return dist



'''
    CC method from networkx is normalized against the biggest weight present in the graph. Which means that when you calculate it for a subgraph the value could with respect to the whole graph as the maximum weight in subgraph could be different.
    Therefore, I have to change slightly the way CC is calculated to fix this issue as in our case the theoretical max weight is always 1
'''
def _weighted_triangles_and_degree_iter(G, nodes=None, weight="weight"):
    """Return an iterator of (node, degree, weighted_triangles).

    Used for weighted clustering.
    Note: this returns the geometric average weight of edges in the triangle.
    Also, each triangle is counted twice (each direction).
    So you may want to divide by 2.

    """
    max_weight = 1
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This avoids counting twice -- we double at the end.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += sum(
                np.cbrt([(wij * wt(j, k) * wt(k, i)) for k in inbrs & jnbrs])
            )
        yield (i, len(inbrs), 2 * weighted_triangles)

def clustering(G, nodes=None, weight=None):
    
    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes)
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
    else:
        # The formula 2*T/(d*(d-1)) from docs is t/(d*(d-1)) here b/c t==2*T
        if weight is not None:
            td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t in td_iter}
        else:
            td_iter = _triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter}
    if nodes in G:
        # Return the value of the sole entry in the dictionary.
        return clusterc[nodes]
    return clusterc



class Complexity:
    '''
    class to wrap the complexity indicators and the two step algorithm to find complex communities and aircraft
    '''
    def __init__(self, G, threshold) -> None:
        self.G = G
        self.threshold = threshold
        self.communities = []
        self.complex_communities = []
        self.calculated = False
        self.individual_contributions = {}
        self.individual_complexity = {}
        self.conf_information = {}
        self.tot_comp = 0
        self.is_complex = {key:False for key in self.G.nodes}



    def individual_strength(self):
        results = {}
        for n in self.G.nodes:
            edges = self.G.edges(n,data=True)
            if len(edges) > 0:
                results[n] = max([edge[2]['weight'] for edge in edges])
            else:
                results[n] = 0.
        self.conf_information = results



    def strength(self, G=None,mean=False):
        if G is None: #if you don't give this method a Graph parameter it works with the Graph of the instance. This is the same for all methods in this class
            G = self.G 
        node_degrees = G.degree(weight = "weight")

        if mean is False:
            return node_degrees


            # Mean of the degrees 
        degrees_mean = np.mean([degree[1] for degree in node_degrees])
        return degrees_mean
    
    
    
    
    def cc(self, G=None,mean=False):
        """
        Return the clustering coefficient for every node or the mean value if 'mean' parameter is True
        """
        if G is None:
            G = self.G
        
        
        return list(clustering(G,weight='weight').items())
    
    def nnd(self, G=None,mean=False):
        if G is None:
            G = self.G
        #nodes_nn_degree = list(nx.average_neighbor_degree(self.G, weight = "weight").values())
        nodes_nn_degree = list(nx.average_neighbor_degree(G,weight='weight').items())
        
        return nodes_nn_degree
        #if mean:
        #    return np.mean(nodes_nn_degree)
        #else:
        #    return nodes_nn_degree

    def graph_complexity(self,G=None):
        '''
        calculate the overall complexity of the graph for each component. returns a dict with with indicators and values
        '''
        if G is None:
            G = self.G
            st = dict(self.strength(G))
            cc = dict(self.cc(G))
            nnd = dict(self.nnd(G))
            self.individual_complexity = {key:[st[key]]+[cc[key]]+[nnd[key]] for key in st.keys()}
        else:
            st = dict(self.strength(G))
            cc = dict(self.cc(G))
            nnd = dict(self.nnd(G))
            
            
        tot_st = sum([val for key,val in st.items()])
        tot_cc = sum([val for key,val in cc.items()])
        tot_nnd = sum([val for key, val in nnd.items()])
        

        return {'strength':tot_st, 'cc':tot_cc,'nnd':tot_nnd}
    
    def get_complex_communities(self):
        '''
        get complex connected components (communities) in graph
        '''

        self.complex_communities  = []
        self.communities = list((self.G.subgraph(c) for c in nx.connected_components(self.G)))
        self.tot_comp = self.graph_complexity()

        for comm in self.communities:
            comp = self.graph_complexity(comm)
            respon_st = comp['strength']/self.tot_comp['strength'] if self.tot_comp['strength'] > 0 else 0
            respon_cc = comp['cc']/self.tot_comp['cc'] if self.tot_comp['cc'] > 0 else 0
            respon_nnd = comp['nnd']/self.tot_comp['nnd'] if self.tot_comp['nnd'] > 0 else 0
            complexity = np.array([respon_st, respon_cc, respon_nnd])
            percentage = complexity[complexity != 0].mean()#np.mean([respon_st, respon_cc, respon_nnd])
            
            if percentage >= self.threshold:
                self.complex_communities.append((comm,percentage,comp['strength'],comp['cc'],comp['nnd']))
        
        
        self.calculated = True
        return self.complex_communities
    
    
    def granular_complexity(self, G=None, per_sector=True):
        '''
        performs the two step algorithm
        '''
        if G is None:
            G = self.G
        if not self.calculated:
            self.get_complex_communities()
        #print("CALCULATED GRANULAR COMPLEXITY")
        #print(f"The graph we are investigating has {len(self.complex_communities)} complex communities \n")
        
        i=0
        #for com in self.complex_communities:
            #i+=1
            #members,percentage,st,cc,nnd = com[0],com[1],com[2],com[3],com[4]
            
         
            #print(f"Complex community {i} with members {members.nodes} is responsible for {round(percentage*100,2)}% of the overall complexity. The contributions of individual aircraft to complexity are as follows:\n")
        for ac,val in self.individual_complexity.items():
            if not per_sector:
                pass
            else:
                st, cc, nnd = self.tot_comp['strength'], self.tot_comp['cc'],self.tot_comp['nnd']
                ac_st, ac_cc, ac_nnd = val[0], val[1], val[2]
                respon_st = ac_st / st if st > 0 else 0
                respon_cc = ac_cc / cc if cc > 0 else 0
                respon_nnd = ac_nnd / nnd if nnd > 0 else 0
                complexity = np.array([respon_st,respon_cc,respon_nnd])
                contribution = complexity[complexity != 0].mean()#np.mean([respon_st,respon_cc,respon_nnd])
                contribution = 0 if np.isnan(contribution) else contribution
                self.individual_contributions[ac] = contribution
        
        #determine which ac are in complex community
        for com in self.complex_communities:
            members = com[0].nodes
            for member in members:
                self.is_complex[member] = True

        #print(f"THESE ARE INDIVIDUAL CONTRIBUTIONS {self.individual_contributions}")

        
        
    

    def draw_graph(self):
        if not self.calculated:
            _ = self.get_complex_communities()
        plt.axis('off')
        for i,subgraph in enumerate(self.communities):
            pos = nx.spring_layout(subgraph,seed=1)
            if i != 0:
                #continue
                for k,v in pos.items():
                    v[0] = v[0] + 3
                
            nx.draw_networkx_nodes(subgraph, pos=pos,node_size=300, alpha=0.1,label=subgraph.nodes())
            nx.draw_networkx_labels(subgraph,pos=pos)
            nx.draw_networkx_edges(subgraph, pos=pos, alpha=0.2)
            labels = nx.get_edge_attributes(subgraph,'weight')
            nx.draw_networkx_edge_labels(subgraph,pos,edge_labels=labels)
            

    
    def draw_complex_communities(self):
        if not self.calculated:
            _ = self.get_complex_communities()
        vmin = 0
        vmax = 1
        for com in self.complex_communities:
            members,percentage,st,cc,nnd = com[0],com[1],com[2],com[3],com[4]
            contribution = [self.individual_contributions[key] for key in members.nodes]
            pos = nx.spring_layout(members,seed=1)
            for k,v in pos.items():
                # Shift the y value of the position of each node 10 down
                v[0] = v[0] +60
            cmap = plt.cm.coolwarm
            vmax = np.max(contribution)
            nx.draw_networkx(members, pos=pos,with_labels=True,node_color=contribution,cmap=cmap,vmin=vmin,vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            labels = nx.get_edge_attributes(members,'weight')
            nx.draw_networkx_edge_labels(members,pos,edge_labels=labels)
        


    

def temporal_granular_complexity(T,graphs,cthresh=None):
    '''
    Extend granular complexity to the time domain
    '''
    communities = {} #key is timestep value is list of Community objects
    print(f"TIME IS {len(T)}")
    no_comms = 0
    timestep = T[1] - T[0] if len(T) > 1 else 1#30
    comp_objects = []

    if cthresh is None:
        cthresh = 0.5

    for idx,t in enumerate(T):

    #generate graph from data
        G = graphs[idx]#nx.Graph()
        comp = Complexity(G,threshold=cthresh)
        
        comp.granular_complexity()

        for community in comp.complex_communities:
            curr_members = community[0].nodes
            #get complex communities of last time step (only if there is a timestep before

            if t == 0:
                if t in communities:
                    communities[t].append(Community(no_comms+1,curr_members,t,t,community[1]))
                    no_comms += 1
                else:
                    communities[t] = [Community(no_comms+1,curr_members,t,t,community[1])]
                    no_comms += 1
            else:
                prev_comm = communities.get(t-timestep,[])#prev_comm = communities[t - timestep]
                if len(prev_comm) == 0:

                        if t in communities:
                            communities[t].append(Community(no_comms+1,curr_members,t,t,community[1]))
                            no_comms += 1
                        else:
                            communities[t] = [Community(no_comms+1,curr_members,t,t,community[1])]
                            no_comms += 1
                else:
                    scores = []
                    for prev in prev_comm:
                        scores.append(jaccard_similarity(curr_members,prev.members))
                    max_similarity = np.argmax(scores)

                    if scores[max_similarity] == 0.:
                        if t in communities:
                            communities[t].append(Community(no_comms+1,curr_members,t,t,community[1]))
                            no_comms += 1
                        else:
                            communities[t] = [Community(no_comms+1,curr_members,t,t,community[1])]
                            no_comms += 1
                    else:
                        #deal with changing communities
                        #modified community needs to be added to the current timestep as basically the existing previous community but modified
                        most_similar = prev_comm[max_similarity]
                        if scores[max_similarity] == 1.:
                            most_similar.update_community(t,community[1])
                        else:
                            unique_members = list(set(most_similar.members) ^ set(curr_members))

                            for member in unique_members:
                                if member in most_similar.members: #member was previously but not anymore, need to remove it
                                    most_similar.remove_member(member,t)
                                else: #member was not previously there, need to add it
                                    most_similar.add_member(member,t)



                            most_similar.update_community(t,community[1])

                        if t in communities:
                            communities[t].append(most_similar)
                        else:
                            communities[t] = [most_similar]
        comp_objects.append(comp)
    final_communities = []
    for key,val in communities.items():
        for community in val:

            if any(c.id == community.id for c in final_communities):
                continue
            final_communities.append(community)
    
    return final_communities, comp_objects             



def graph_from_data(file, min_thresh=5, max_thresh=30):
    #print(f" JEMI TE GRAPH FROM DATA ME FAJL {file}")
    df = pd.read_csv(file)
    times = []
    graphs = []
    min_lat = 0
    min_lon = 0
    max_lat = 0 
    max_lon = 0
    df.columns = df.columns.str.replace(' ','')
    #print(df.columns)
    
    min_lats = []
    max_lats = []

    min_lon = []
    max_lon = []

    mask =    (df['time'] >= 12500.0) & (df['time'] <= 14300.0)#(df['time'] >= 2200.0) & (df['time'] <= 4000.0)#(df['time'] >= 0.) & (df['time'] <= 4500.)## (df['time'] >= 4500.0) & (df['time'] <= 6700.0)#
    df = df.loc[mask]


    groups = df.groupby('time')
    for name, group in groups:
        times.append(name)
        ids = group['aircraft_id'].to_numpy()
        lats = group['lat'].to_numpy()
        lons = group['lon'].to_numpy()
        min_lats.append(min(lats))
        max_lats.append(max(lats))

        min_lon.append(min(lons))
        max_lon.append(max(lons))

        I = np.eye(len(ids))
        dist = kwikqdrdist_matrix(np.asmatrix(lats), np.asmatrix(lons),
                                        np.asmatrix(lats), np.asmatrix(lons))


        dist = np.asarray(dist) + 1e9 * I

        dist = (max_thresh - dist)/(max_thresh - min_thresh)

        dist[dist > 1] = 1
        dist[dist < 0] = 0
        dist = np.round(dist,2)
        #print(dist)
        G = nx.from_numpy_matrix(dist, parallel_edges=False, create_using=None)
        G = nx.relabel_nodes(G, {i: ids[i] for i in G.nodes})
        
        pos = {}
        for idx,node in enumerate(list(G.nodes)):
            pos[node] = {'lat':lats[idx],'lon':lons[idx]}
        nx.set_node_attributes(G, pos)
        graphs.append(G)
    
    margin = .4
    
    lat_min = min(min_lats) - margin
    lat_max = max(max_lats) + margin
    long_min = min(min_lon) - margin
    long_max = max(max_lon) + margin

    return times,graphs,[lat_min, lat_max, long_min,long_max]

class Community:
    def __init__(self,name, members, start_time, end_time,percentage):
        self.id = name
        self.members = list(members)
        self.added = {member:start_time for member in list(members)}
        self.removed = {}
        self.start_time = start_time
        self.end_time = end_time
        self.percentage = [percentage]
    
    def add_member(self,member,time):
        self.members.append(member)
        self.added[member] = time
        
    def update_community(self,time,percentage):
        self.end_time = time
        self.percentage.append(percentage)
        
    
    def end_community(self,time):
        self.end_time = time
    
    def remove_member(self,member,time):
        self.members.remove(member)
        self.removed[member] = time

 
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def edge_density(graph):

    # Sum of the weights 
    sum_w = sum([d["weight"] for _, _, d in graph.edges(data=True)])
    V = graph.number_of_nodes()
    return sum_w / (V*(V-1)/2)

