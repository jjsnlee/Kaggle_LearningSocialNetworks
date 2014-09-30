import networkx as nx
import pandas as pd
#from collections import defaultdict
#import sys
#import drawing
import numpy as np
from os.path import join
import itertools

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
def training_features(filename='../features.txt'):
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

def train(ego_ds, covar_type='diag'):
    from sklearn import mixture
    
    n_components = len(ego_ds.labels)
    
    g = mixture.GMM(n_components=n_components, init_params='wc', covariance_type=covar_type)
    samples = ego_ds.M
    targets = ego_ds.friend_targets
    
    lbl_keys = [ego_ds.label_keys[lbl] for lbl in ego_ds.labels]
    g.means_ = np.array([samples[targets&lbl].mean(axis=0) for lbl in lbl_keys])
    
    g.fit(samples)
    return g

def trainKM(EgoDS, n_components=2, covar_type='diag'):
    from sklearn import cluster
    g = cluster.KMeans(n_clusters=n_components, covariance_type=covar_type)
    g.fit(EgoDS.M.values)
    return g

class EgoDataSet():
    def __init__(self, ego, by='edge'):
        self.ego = ego
        fname = join('../Data/egonets', str(ego)+'.egonet')
        G = read_nodeadjlist(fname)
        featuremap = training_features()
        print 'loaded features...'        
        self.G = G
        circles = training_circles(ego)
        
        #assert by in ('node', 'edge')
        assert by in ('edge')
        if by == 'node':
            M = create_features_by_node(ego, G, featuremap)
        elif by == 'edge':
            M = create_features_by_edge(ego, G, featuremap)
            all_friends = set(M.index)
            _circles = {}
            # need to check pairwise to see if there are any edges
            for c, friends in circles.iteritems():
                for f1 in friends:
                    egokey = 'EGO_%s'%(f1)
                    _circles[egokey] = c
                    f1_matched = False
                    for f2 in friends:
                        if f1==f2:
                            continue
                        matched_key = None
                        key_ = '%s_%s'%(f1,f2)
                        if key_ in all_friends: 
                            matched_key = key_
                        else:
                            key_ = '%s_%s'%(f2,f1)
                            if key_ in all_friends:
                                matched_key = key_
                        if matched_key:
                            if matched_key in _circles and _circles[matched_key]!=c:
                                if not isinstance(_circles[matched_key], set):
                                    _circles[matched_key] = set([_circles[matched_key]])
                                _circles[matched_key].add(c)
                            else:
                                _circles[matched_key] = c
                            f1_matched = True
                    if not f1_matched:
                        print 'No mutual cnxs for', f1
            matched_keys = set(_circles.keys())
            
            # A circle for those not in any circles...
            unmatched = dict.fromkeys(list(all_friends.difference(matched_keys)), '__NO_CIRCLE__')
            _circles.update(unmatched)
            circles = _circles
        
        self.circles = circles

        labels = set()
        for v in circles.values():
            if isinstance(v, set):
                labels.update(v)
            else:
                labels.add(v)
        labels = list(labels)
        labels.sort()
        
        #list(set(itertools.chain(*circles.values())))
        self.M = M
        self.friends = M.index
        
        lbl_keys = dict([(l,2**i) for i,l in enumerate(labels)])
        def score(i):
            lbl = circles[i]
            if isinstance(lbl, set):
                return sum([lbl_keys[l] for l in lbl])
            return lbl_keys[lbl]
        
        self.label_keys = lbl_keys
        self.labels = labels
        self.friend_targets = pd.Series([score(i) for i in M.index], index=M.index)
        
        #self.friend_features = None
        #df = pd.DataFrame(featuremap)
        #all_persons = df.columns 
        #features = df.index
        #ego_features = featuremap[ego]
    
    @property
    def sample_labels(self):
        return self.circles
    @property
    def samples(self):
        return self.friends

def main(ego, by='node'):
    ds = EgoDataSet(ego, by=by)
    return ds

def training_circles(ego):
    fname = join('../Data/Training', str(ego)+'.circles')
    circles = {}
    for line in open(fname):
        e1, es = line.split(':')
        friends = es.strip().split(' ')
        circles[e1] = friends
        #circles.update(dict([(f, e1) for f in friends]))
    return circles

def create_features_by_node(ego, G, featuremap):
    ego_friends = G.nodes()
    m = {}
    for friend in ego_friends:
        fr_m = featuremap[friend]
        row = {}
        for k,v in fr_m.iteritems():
            keyval = '%s=%s'%(k, v)
            row[keyval] = 1
        m[friend] = row
    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    return M.T

def create_features_by_edge(ego, G, featuremap):
    """
    The affinity propagation problem uses an edge model? What is the appeal
    of using the edges instead of nodes to figure out clusters? 
    """
    
    def union_features(m1, m2):
        row = {}
        keys = set(m1.keys())
        keys.update(m2.keys())
        for k in keys:
            try:
                #row['same_'+k] = p1_m[k]==p2_m[k]
                if m1[k]==m2[k]:
                    row['same_'+k] = 1
                else:
                    row['same_'+k] = -1
            except:
                pass
        return row
    
    ego_friends = G.nodes()
    ego_features = featuremap[ego]   
    m = {}
    for f1 in ego_friends:
        f1_m = featuremap[f1]
        m['EGO_%s'%(f1)] = union_features(ego_features, f1_m)
        # need to add the ego
        for f2 in ego_friends:
            if f1==f2 or '%s_%s'%(f2,f1) in m:
                continue
            f2_m = featuremap[f2]
            m['%s_%s'%(f1,f2)] = union_features(f1_m, f2_m)

    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    return M.T
    
def mean_nozero(M):
    return np.mean([len(M.ix[i].nonzero()[0]) for i in M.index])    
    
    #cc = nx.connected_components(G)
    #return c

#if __name__ == '__main__':
#	main()
