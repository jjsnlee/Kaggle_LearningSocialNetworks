import networkx as nx
import pandas as pd
#from collections import defaultdict
#import sys
#import drawing
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import itertools
from kaggle_data import training_features, training_circles

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

def train1(ego_ds, covar_type='diag'):
    from sklearn import mixture
    
    labels = ego_ds.labels
    n_components = len(labels)
    
    g = mixture.GMM(n_components=n_components, init_params='wc', covariance_type=covar_type)
    samples = ego_ds.M
    targets = ego_ds.friend_targets
    
    lbl_keys = [ego_ds.label_keys[lbl] for lbl in ego_ds.labels]
    g.means_ = np.array([samples[targets&lbl].mean(axis=0) for lbl in lbl_keys])
    
    g.fit(samples)
    
    # Need to recover these to the correct bit format!
    def xform(i):
        return ego_ds.label_keys[ego_ds.labels[i]]
    preds = [xform(p) for p in g.predict(samples)]
    print 'preds:', preds
    train_accuracy = np.mean(preds==targets)*100
    print 'train_accuracy: %.1f%%' % train_accuracy
    return g

def train(ds, samples=None, covar_type='diag', include_model=False, method='gmm'):
    from sklearn import feature_selection 
    
    labels = ds.labels
    targets = ds.friend_targets
    if samples is None:
        samples = ds.M

    # for testing remove ones which don't belong to a circle...
#     labels = labels[1:]
#     samples = samples[targets>0]
#     targets = targets[targets>0]
#     print 'labels:', labels
#     print 'samples.shape:', samples.shape, 'targets.shape:', targets.shape

    n_components = len(labels)

    rslts = {}

    #sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    # If I remove the ones with zero variance it still gets to the same answer
    sel = feature_selection.VarianceThreshold()
    sel.fit_transform(samples)
    samples = samples[samples.columns[sel.get_support()]]
    print 'samples:', samples.shape
    
    try:
        if method=='gmm':
            from sklearn import mixture
            model = mixture.GMM(n_components=n_components, init_params='wc', 
                            #n_iter=500,
                            #n_init=n_init, # doesn't make sense to set if we are setting the means
                            covariance_type=covar_type)
            model.means_ = np.array([samples[targets==i].mean(axis=0) for i in xrange(len(labels))])
            model.fit(samples)
        elif method=='LR':
            from sklearn import linear_model, decomposition
            from sklearn.pipeline import Pipeline
            from sklearn.grid_search import GridSearchCV
            logistic = linear_model.LogisticRegression()
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
            estimator = GridSearchCV(pipe, {}
                                     #dict(pca__n_components=n_components, logistic__C=Cs)
                                     )
            estimator.fit(samples, targets)
            model = estimator
            
        # Need to cover these to the correct bit format!
        preds = model.predict(samples)
        #print 'preds:', preds
        train_accuracy = np.mean(preds==targets)*100
        print 'train_accuracy: %.1f%%' % train_accuracy
        #return g
    except Exception as e:
        import traceback
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s' % (ds.ego, st)        
        preds = model = train_accuracy = None
    
    rslts['n_samples'] = len(samples.index)
    rslts['n_components'] = n_components
    rslts['non_trivial_features'] = samples.columns
    rslts['n_non_trivial_features'] = len(samples.columns)
    rslts['train_accuracy'] = train_accuracy
    if include_model:
        rslts['model'] = model
        rslts['preds'] = preds
    
    # Ultimately the real evaluation needs to be how these map to the actual circles
    # So should do some conversion, and use that for the ultimate accuracy
    
#     circle_rslts = {}
#     for pred in preds.unique():
#         samples_with_pred = samples.index[preds==pred]
#         if pred > 0:
#             circle_rslts[pred] = set(samples_with_pred)
    
    return rslts

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
        self.G = G
        
        assert by in ('node', 'edge')
        if by == 'node':
            M, circles = create_features_by_node(ego, G, featuremap)
        elif by == 'edge':
            M, circles = create_features_by_edge(ego, G, featuremap)
            
        #print 'generated features...'
        
        self.circles = circles

        labels = set()
        labels.add(None)
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
        
        #lbl_keys = dict([(l,2**i) for i,l in enumerate(labels)])
        lbl_keys = dict([(l,i) for i,l in enumerate(labels)])
        def score(i):
            lbl = circles.get(i)
            if isinstance(lbl, set):
                # arbitrarily take the first one for now
                return lbl_keys[list(lbl)[0]]
                #return sum([lbl_keys[l] for l in lbl])
            return lbl_keys[lbl]

        self.label_keys = lbl_keys
        self.labels = labels
        self.friend_targets = pd.Series([score(i) for i in M.index], index=M.index)
        
        # Hack to seed the data
        # Interesting, if all the other features are binary, then adding a non-binary one
        # (with the label) actually degraded the performance... 
#         for lbl_idx in range(len(labels)):
#             X = self.friend_targets==lbl_idx
#             X = X.map({ True:1, False:0})
#             M['Label %s'%lbl_idx] = X
    
    @property
    def sample_labels(self):
        return self.circles
    @property
    def samples(self):
        return self.friends

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
                # Need to explicitly make this 0/1
                if m1[k]==m2[k]:
                    row['%s=%s'%(k, m1[k])] = 1
            except:
                pass
        return row
    # this can take some time...
    #cliques = [set(c) for c in nx.find_cliques(G)]
    cliques = set()
    for cliq in nx.find_cliques(G):
        for i,c1 in enumerate(cliq):
            for c2 in cliq[i+1:]:
                cliques.add('%s_%s'%(c1,c2))
                cliques.add('%s_%s'%(c2,c1))
    
    print 'Generated cliques'
    
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
            key = '%s_%s'%(f1,f2)
            m[key] = union_features(f1_m, f2_m)
            if key in cliques:
                m[key]['in_clique'] = 1

    print 'Generated features'

    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    M = M.T

    circles = training_circles()[ego]
    all_friends_edges = set(M.index)
    
    from collections import defaultdict
    overlaps = defaultdict(int)
    
    emptyset = frozenset()
    print 'Checking circles'
    _circles = {}
    for f_edge in all_friends_edges:
        f1,f2 = f_edge.split('_') 
        f1c = circles.get(f1, emptyset)
        f2c = circles.get(f2, emptyset)
        intersect = f1c.intersection(f2c)
        #if f1c==f2c or f1=='EGO':
        if len(intersect)>0:
            if len(intersect)>1:
                overlaps[len(intersect)]+=1
                #print len(intersect), intersect 
            _circles[f_edge] = list(intersect)[0]
        elif f1=='EGO' and f2c:
            circles[f_edge] = list(f2c)[0]
    
    print 'overlaps:', overlaps
    return M, _circles
    
    # need to check pairwise to see if there are any edges
#     for c, friends in circles.iteritems():
#         for f1 in friends:
#             egokey = 'EGO_%s'%(f1)
#             _circles[egokey] = c
#             f1_matched = False
#             for f2 in friends:
#                 if f1==f2:
#                     continue
#                 matched_key = None
#                 key_ = '%s_%s'%(f1,f2)
#                 if key_ in all_friends: 
#                     matched_key = key_
# #                         else:
# #                             key_ = '%s_%s'%(f2,f1)
# #                             if key_ in all_friends:
# #                                 matched_key = key_
# #                         if matched_key:
#                     # This is only to support users being in more than one circle
#                     if matched_key in _circles and _circles[matched_key]!=c:
#                         if not isinstance(_circles[matched_key], set):
#                             _circles[matched_key] = set([_circles[matched_key]])
#                         _circles[matched_key].add(c)
#                     else:
#                         _circles[matched_key] = c
#                     f1_matched = True
#             if not f1_matched:
#                 print 'No mutual cnxs for', f1
    #matched_keys = set(_circles.keys())
    # A circle for those not in any circles...
    #unmatched = dict.fromkeys(list(all_friends_edges.difference(matched_keys)), '__NO_CIRCLE__')
    #_circles.update(unmatched)
    #circles = _circles

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
    M = M.T
    
    circles = training_circles()[ego]

    # A circle for those not in any circles...    
    all_friends = set(M.index)
    matched_keys = set(circles.keys())
    unmatched = dict.fromkeys(list(all_friends.difference(matched_keys)), '__NO_CIRCLE__')
    circles.update(unmatched)        
    
    return M, circles

def main():
    print 'Starting:', get_ts()
    rslts = {}
    egocircles = training_circles().keys()
    
    print '# egocircles:', len(egocircles)
    for i, ego in enumerate(egocircles):
        print '-'*80
        print 'Executing for %s (#%s)' %(ego, i)
        ds = EgoDataSet(ego, by='edge')
        print 'Initialized ego'
        rslts[ego] = train(ds)

    print 'Done:', get_ts()
    return rslts

def draw_results_graph(ego_ds, model):
    samples = ego_ds.M
    preds = model.predict(ego_ds.M)
    G = ego_ds.G.copy()
    
    #plt.title(str())
    plt.xticks([])
    plt.yticks([])
    
    train_accuracy = np.mean(preds==ego_ds.friend_targets)*100
    print 'train_accuracy: %.1f%%' % train_accuracy
    
    lbl_clrs = itertools.cycle(['w', 'r', 'g', 'b', 'c', 'm'])
    
    predvals = pd.Series(preds).unique()
    predvals.sort()
    
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.spectral_layout(G)

    for lblidx, clr in zip(predvals, lbl_clrs):
        
        lbl = ego_ds.labels[lblidx]
        
        edges = samples.index[preds==lblidx]
        
        print 'lblidx:', lblidx, lbl, len(edges)
        if lbl=='__NO_CIRCLE__':
            lbl = 'No Circle'
        
        _nodes = set()
        #_edges = []
        for e in edges:
            n1,n2 = e.split('_')
            n2=int(n2)
            _nodes.add(n2)

            if n1!='EGO':
                n1=int(n1)
                #_edges.append((n1,n2))
                _nodes.add(n1)
        
        nx.draw_networkx_edges(G, pos, edge_color=clr, width=1, alpha=0.5)
        #nx.draw_networkx_edges(G, pos, edgelist=_edges, edge_color=clr, width=1, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=_nodes, node_color=clr, width=1, alpha=0.5, label=lbl)
        
    #nx.draw_networkx_labels(G, pos_lbls, alpha=0.5)
    nx.draw_networkx_labels(G, pos, alpha=0.5)
    
    plt.legend()
    plt.show()    

def create_dumb_data(ego_ds):
    """OK this works perfectly, as expected..."""
    X = {}
    features = ego_ds.M.columns
    for idx in ego_ds.samples:
        target = ego_ds.friend_targets[idx]
        X[idx] = dict([(f, target) for f in features])
    return pd.DataFrame(X).T

from datetime import datetime
import time
def get_ts():
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    
def mean_nozero(M):
    return np.mean([len(M.ix[i].nonzero()[0]) for i in M.index])    
    
    #cc = nx.connected_components(G)
    #return c

#if __name__ == '__main__':
#	main()
