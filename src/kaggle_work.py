import networkx as nx
import pandas as pd
#from collections import defaultdict
#import sys
#import drawing
import numpy as np
from os.path import join

def read_nodeadjlist(filename):
    G = nx.Graph()
    for line in open(filename):
        e1, es = line.split(':')
        es = es.split()
        for e in es:
            if e == e1: continue
            #print e1, e
            G.add_edge(int(e1),int(e))
    return G

_featuremap = {}
def load_features(filename='../features.txt'):
    global _featuremap
    if not _featuremap:
        ppls = {}
        for line in open(filename):
            features = line.split(' ')
            person = int(features[0])
            features = features[1:]
            ppls[person] = {}
            for f in features:
                i = f.rfind(';')
                ppls[person][f[:i]] = int(f[i+1:])
        _featuremap = ppls
    return _featuremap

# def train(M, n_components=2):
#     from sklearn import mixture
#     g = mixture.GMM(n_components=n_components)
#     g.fit(M)
#     return g

def main(ego, by='node'):
    fname = join('../Data/egonets', str(ego)+'.egonet')
    G = read_nodeadjlist(fname)
    featuremap = load_features()
    print 'loaded features...'
    #df = pd.DataFrame(featuremap)
    #all_persons = df.columns 
    #features = df.index
    #ego_features = featuremap[ego]
    if by == 'node':
        return create_features_by_node(ego, G, featuremap)
    elif by == 'edge':
        return create_features_by_edge(ego, G, featuremap)

def create_features_by_node(ego, G, featuremap):
    ego_friends = G.nodes()
    
    m = {}
    for p1 in ego_friends:
        p1_m = featuremap[p1]
        row = {}
        for k,v in p1_m.iteritems():
            keyval = '%s=%s'%(k, v)
            row[keyval] = 1
        m[p1] = row

    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    return M.T

def create_features_by_edge(ego, G, featuremap):
    """
    The affinity propagation problem uses an edge model? What is the appeal
    of using the edges instead of nodes to figure out clusters? 
    """
    ego_friends = G.nodes()
    #ego_features = featuremap[ego]   
    m = {}
    for p1 in ego_friends:
        p1_m = featuremap[p1]
        # need to add the ego
        for p2 in ego_friends:
            #if p1==p2 or (p2,p1) in m:
            if p1==p2 or '%s_%s'%(p2,p1) in m:
                continue
            p2_m = featuremap[p2]
            row = {}
            keys = set(p1_m.keys())
            keys.update(p2_m.keys())
            for k in keys:
                try:
                    #row['same_'+k] = p1_m[k]==p2_m[k]
                    if p1_m[k]==p2_m[k]:
                        row['same_'+k] = 1
                    else:
                        row['same_'+k] = -1
                except:
                    pass
            m['%s_%s'%(p2,p1)] = row
    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    return M.T
    
def mean_nozero(M):
    return np.mean([len(M.ix[i].nonzero()[0]) for i in M.index])    
    
    #cc = nx.connected_components(G)
    #return c

#if __name__ == '__main__':
#	main()
