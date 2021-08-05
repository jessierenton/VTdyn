import numpy as np
import pandas as pd
import igraph

def number_proliferating(tissue,alpha):
    Zp = sum(tissue.mesh.areas>=np.sqrt(3)/2*alpha)
    return Zp

def get_number_proliferating_neighbours(tissue,neighbours,alpha):
    return sum(tissue.mesh.areas[neighbours] >= np.sqrt(3)/2*alpha)

def number_proliferating_neighbours(tissue,alpha):
    proliferating = np.where(tissue.mesh.areas>=np.sqrt(3)/2*alpha)[0]
    if len(proliferating) == 0:
        return np.array([0]) 
    return np.array([get_number_proliferating_neighbours(tissue,tissue.mesh.neighbours[i],alpha) for i in proliferating])

def number_proliferating_neighbours_distribution(history,alpha):
    data = [np.bincount(number_proliferating_neighbours(tissue,alpha)) for tissue in history]
    Zp = [number_proliferating(tissue,alpha) for tissue in history]
    maxlen = max(len(nnp) for nnp in data)
    data = [np.pad(nnp,(0,maxlen-len(nnp)),'constant') for nnp in data]
    df = pd.DataFrame([{'np_{:d}'.format(i):f for i,f in enumerate(nnp)} for nnp in data])
    df.insert(0,'Zp',Zp)
    df.insert(1,'clusters',number_clusters(history,alpha))
    return df
    
def mean_number_proliferating_neighbours_df(histories,params):
    data = [pd.DataFrame([{'alpha':alpha,'db':db,'Zp':number_proliferating(tissue,alpha),'nn_p':np.mean(number_proliferating_neighbours(tissue,alpha))} for tissue in                   history])
                for history,(alpha,db) in zip(histories,params)]
    return pd.concat(data)

def number_proliferating_neighbours_distribution_df(histories,params):
    return pd.concat([number_proliferating_neighbours_distribution(history,alpha).assign(db=db,alpha=alpha)
                for history,(alpha,db) in zip(histories,params)])

def create_pgraph(tissue,alpha):
    proliferating = np.where(tissue.mesh.areas>=np.sqrt(3)/2*alpha)[0]
    edges = list(set([tuple(sorted([i,np.where(proliferating==neighbour)[0][0]] ))
                for i,cell_id in enumerate(proliferating) 
                    for neighbour in tissue.mesh.neighbours[cell_id]
                        if neighbour in proliferating] )  )           
    return igraph.Graph(n=len(proliferating),edges=edges)

def number_clusters(history,alpha):
    return [len(create_pgraph(tissue,alpha).clusters()) for tissue in history]

def cluster_df(histories,params):
    data = [pd.DataFrame([{'alpha':alpha,'db':db,'Zp':number_proliferating(tissue,alpha),'nclusters':len(create_pgraph(tissue,alpha).clusters())}
                    for tissue in history])
                for history,(alpha,db) in zip(histories,params)]
    return pd.concat(data)