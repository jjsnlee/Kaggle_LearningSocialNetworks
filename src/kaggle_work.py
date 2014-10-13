import networkx as nx
import pandas as pd
import numpy as np
import kaggle_data as kd
import re

import socialCircles_metric as evaler

def select_features():
    pass

def train(ds, method='gmm', covar_type='diag', include_model=False, 
          uf=None,
          feature_threshhold=20,
          verbose=False):

    labels = ds.labels
    targets = ds.friend_targets
    samples = ds.M
    # Remove columns with less than 5 incidents 
    samples = samples[samples.columns[samples.sum()>feature_threshhold]]
#     features = set(samples.columns)
#     xre = re.compile('(^.+;)id=(\d+)$')
#     for f in features:
#         m = xre.match(f)
#         if m:
#             pass

    # for testing remove ones which don't belong to a circle...
#     labels = labels[1:]
#     samples = samples[targets>0]
#     targets = targets[targets>0]
#     print 'labels:', labels
#     print 'samples.shape:', samples.shape, 'targets.shape:', targets.shape

    n_components = ds.n_components

    rslts = {}

    #sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    # If I remove the ones with zero variance it still gets to the same answer
    from sklearn import feature_selection
    if samples.shape[1] > 1:
        sel = feature_selection.VarianceThreshold()
        sel.fit_transform(samples)
        samples = samples[samples.columns[sel.get_support()]]
        print 'samples:', samples.shape

    from sklearn.svm import LinearSVC    
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    #print 'len(samples):', len(samples)
    if uf is None:
        #if len(samples) < 1e5:
        #svc = SVC(kernel="linear")
        svc = LinearSVC(dual=False, 
                        #verbose=verbose
                        )
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        print 'Starting feature elimination:', get_ts()
        rfecv = RFECV(estimator=svc, step=.1, 
                      cv=StratifiedKFold(targets, 2),
                      scoring='accuracy',
                      verbose=verbose
                      )
        rfecv.fit(samples, targets)
        print 'Completed feature elimination:', get_ts()
        print("Optimal number of features : %d" % rfecv.n_features_)
        samples = samples[samples.columns[rfecv.support_]]
        
    elif uf: # might have just set it to False
        #col_support = [c.split('=')[0] in uf for c in samples.columns]
        col_support = [c in uf for c in samples.columns]
        samples = samples[samples.columns[col_support]]
    
    print 'samples:', samples.shape
    circle_rslts = []
    preds = model = train_accuracy = None
    try:
        if method=='gmm':
            from sklearn.pipeline import Pipeline
            from sklearn import mixture
            model = mixture.GMM(n_components=n_components, 
                                #init_params='wc', 
                            #n_iter=500,
                            #n_init=n_init, # doesn't make sense to set if we are setting the means
                            covariance_type=covar_type)
            #model.means_ = np.array([samples[targets==i].mean(axis=0) for i in xrange(len(labels))])
            model.fit(samples)

            preds = model.predict(samples)
            #print 'preds:', preds
            train_accuracy = np.mean(preds==targets)*100
            print 'GMM train_accuracy: %.1f%%' % train_accuracy
            
            circle_rslts = [samples.index[preds==pred] for pred in np.unique(preds)]
            # discard the frst one, which is really NO CIRCLE
            circle_rslts = circle_rslts[1:]
            
#         elif method=='km':
#             from sklearn import cluster
#             model = cluster.KMeans(n_clusters=n_components)
#             model.fit(samples)
#             
#             preds = model.predict(samples)
#             #print 'preds:', preds
#             circle_rslts = [samples.index[preds==pred] for pred in np.unique(preds)]
#             # discard the frst one, which is really NO CIRCLE
#             circle_rslts = circle_rslts[1:]

        elif method=='sc':
            from sklearn import cluster
            model = cluster.SpectralClustering(n_clusters=n_components)
            model.fit(samples)
            labels = model.labels_
            print 'labels', labels
            # How do I evaluate this?
            #preds = model.predict(samples)
            #print 'preds:', preds
            circle_rslts = [samples.index[labels==pred] for pred in np.unique(labels)]
            # discard the frst one, which is really NO CIRCLE
            circle_rslts = circle_rslts[1:]
       
        elif method=='rbm':
            from sklearn import neural_network
            model = neural_network.BernoulliRBM(n_components=100,
                                                #random_state=0, 
                                                #n_iter=npasses,
                                                verbose=verbose
                                                )
            X = model.fit_transform(samples)
            
#         elif method=='LR':
#             from sklearn import linear_model, decomposition
#             from sklearn.pipeline import Pipeline
#             from sklearn.grid_search import GridSearchCV
#             logistic = linear_model.LogisticRegression()
#             pca = decomposition.PCA()
#             pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
#             estimator = GridSearchCV(pipe, {}
#                                      #dict(pca__n_components=n_components, logistic__C=Cs)
#                                      )
#             estimator.fit(samples, targets)
#             model = estimator
    except Exception as e:
        import traceback
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s' % (ds.ego, st)        

    loss, trivial_loss = eval_model(ds, circle_rslts)

    rslts['n_samples'] = len(samples.index)
    rslts['n_components'] = n_components
    rslts['non_trivial_features'] = samples.columns
    print 'non_trivial_features', samples.columns
    rslts['n_non_trivial_features'] = len(samples.columns)
    rslts['train_accuracy'] = train_accuracy
    rslts['loss'] = loss
    rslts['trivial_loss'] = trivial_loss
    if include_model:
        rslts['model'] = model
        rslts['preds'] = preds
    
    # Ultimately the real evaluation needs to be how these map to the actual circles
    # So should do some conversion, and use that for the ultimate accuracy
    
#     rslts['predicted'] = ';'.join(['%s' % (circle, ' '.join(friends)) for circle, friends in circle_rslts])
        
    return rslts

def eval_model(ds, circle_rslts):
    circled_friends = set()
    other_circles = []
    for circle in circle_rslts:
        curr_circle = set()
        other_circles.append(curr_circle)
        for edge in circle:
            users = edge.split('_')
            if users[0]!='EGO':
                curr_circle.add(int(users[0]))
            if len(users)>1:
                curr_circle.update(map(int, users[1:]))
        circled_friends = circled_friends.union(curr_circle)
    all_friends = set(ds.friends)
    other_circles.append(all_friends.difference(circled_friends))
    gCircles = kd.get_training_data()[ds.ego]
    loss = evaler.loss1(gCircles, other_circles)
    print 'loss=', loss
    trivial_loss = evaler.loss1(gCircles, [all_friends])
    print 'trivial_loss=', trivial_loss
    return loss, trivial_loss

class EgoDataSet():
    def __init__(self, ego, by='edge'):
        self.ego = ego
        G = kd.load_ego_graph(ego)
        self.G = G
        
        assert by in ('node', 'edge', 'edge2')
        if by == 'node':
            M = create_features_by_node(ego, G)
        elif by == 'edge':
            M = create_features_by_edge(ego, G)
        elif by == 'edge2':
            M = create_features_by_edge2(ego, G)
            
        #print 'generated features...'
        circles = _get_circles(ego, set(M.index))
        self.circles = circles

        labels = [None] + list(kd.training_circles()[ego].keys())
        labels.sort()
        
        self.M = M
        #lbl_keys = dict([(l,2**i) for i,l in enumerate(labels)])
        lbl_keys = dict([(l,i) for i,l in enumerate(labels)])
        def score(i):
            lbl = circles.get(i)
            if isinstance(lbl, set):
                # arbitrarily take the first one for now
                return lbl_keys[list(lbl)[0]]
            return lbl_keys[lbl]

        self.label_keys = lbl_keys
        self.labels = labels
        self.friend_targets = pd.Series([score(i) for i in M.index], index=M.index)
        self.n_components = len(self.labels)
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
#     @property
#     def samples(self):
#         return self.M.index
    @property
    def friends(self):
        return self.G.nodes()

def learn_number_clusters(ds, method='ap'):
    if method=='ap':
        from sklearn import cluster, covariance
        edge_model = covariance.GraphLassoCV(verbose=True)
        # standardize the time series: using correlations rather than covariance
        # is more efficient for structure recovery
        X = ds.M.values.copy().T
        X /= X.std(axis=0)
        print '--- B'
        edge_model.fit(X)
        _, labels = cluster.affinity_propagation(edge_model.covariance_)
        return labels

    elif method=='rbm':
        from sklearn import neural_network
        model = neural_network.BernoulliRBM(n_components=100,
                                            #random_state=0, 
                                            #n_iter=npasses,
                                            verbose=True
                                            )
        X = model.fit_transform(ds.M)
        return X

def create_features_by_edge2(ego, G):
    ego_cliques = kd.get_ego_cliques(ego)
    print 'About to generate cliques', len(ego_cliques)
    featuremap = kd.training_features()    
    ccs = [set(c) for c in nx.connected_components(G)]
    m = {}
    for cliq in ego_cliques:
        sample = '_'.join(map(str, cliq))
        m[sample] = union_features2([featuremap[user] for user in cliq])
        cliq = set(cliq)
        for cidx, cc in enumerate(ccs):
            if not cliq.difference(cc):
                m[sample]['cc_%s_%s'%(ego, cidx)] = 1
            else:
                #m[sample]['cc_none'] = 1
                pass
        if len(m[sample])==0:
            del m[sample]

    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    M = M.T
    return M

def union_features2(users):
    row = {}
    feature_keys = set([k for u in users for k in u.keys()])
    for k in feature_keys:
        try:
            # Need to explicitly make this 0/1
            feat_vals = np.array([u[k] for u in users])
            if feat_vals.mean()==feat_vals[0]:
                row['%s=%s'%(k, users[0][k])] = 1
        except:
            pass
    return row

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

def create_features_by_edge(ego, G):
    """
    The affinity propagation problem also uses an edge model? 
    What is the appeal of using the edges instead of nodes to figure out clusters? 
    """
    ego_cliques = kd.get_ego_cliques(ego)
    print 'About to generate cliques', len(ego_cliques)
    cliques = set()
    for cliq in ego_cliques:
        for i,c1 in enumerate(cliq):
            for c2 in cliq[i+1:]:
                # faster to hash ints? even tuples
                cliques.add((c1,c2))
                cliques.add((c2,c1))
                #cliques.add('%s_%s'%(c1,c2))
                #cliques.add('%s_%s'%(c2,c1))
    
    print 'Generated cliques', len(cliques)

    featuremap = kd.training_features()    
    ego_friends = G.nodes()
    ego_features = featuremap[ego]   
    ccs = [set(c) for c in nx.connected_components(G)]
    #non_ccs = set()
    #ccs = [non_ccs] + ccs

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
            if (f1,f2) in cliques:
                m[key]['in_clique'] = 1
            
            # should add a non-connected feature?
            for cidx, cc in enumerate(ccs):
                if f1 in cc and f2 in cc:
                    m[key]['cc_%s_%s'%(ego, cidx)] = 1
                else:
                    m[key]['cc_none'] = 1

    print 'Generated clique features'

    M = pd.DataFrame(m)
    M.fillna(0, inplace=True)
    M = M.T
    return M

def _get_circles(ego, all_friends_edges, raw_circles=None):
    if raw_circles is None:
        raw_circles = kd.processed_circles(ego)
    from collections import defaultdict
    overlaps = defaultdict(int)
    emptyset = frozenset()
    print 'Checking circles'
    circles = {}
    for f_edge in all_friends_edges:
        users = f_edge.split('_')
        user_edge_circles = [raw_circles.get(u, emptyset) for u in users]
        intersect = reduce(lambda a,b: a.intersection(b), user_edge_circles) 
        #if f1c==f2c or f1=='EGO':
        if len(intersect)>0:
            if len(intersect)>1:
                overlaps[len(intersect)]+=1
                #print len(intersect), intersect 
            circles[f_edge] = list(intersect)[0]
#         elif f1=='EGO' and f2c:
#             # wow this causes some bad vibes...
#             #circles[f_edge] = list(f2c)[0]
#             pass
    
    print 'overlaps:', overlaps
    return circles

def main(method='gmm', by='edge', verbose=False, feature_threshhold=20, uf=None): #uf=uniq_features, 
    print 'Starting:', get_ts(), 'using', method
    rslts = {}
    egocircles = kd.training_circles().keys()
    import kaggle_ego_cache as kec
    kec._egocache={}
    
    total_loss = total_trivial_loss = 0
    print '# egocircles:', len(egocircles)
    for i, ego in enumerate(egocircles):
        print '-'*80
        print 'Executing for %s (#%s)' %(ego, i)
        ds = kec.get_ego(ego, by=by)
        print 'Initialized ego'
        rslts[ego] = train(ds, method, uf=uf, 
                           verbose=verbose, 
                           feature_threshhold=feature_threshhold)
        total_loss += rslts[ego]['loss']
        total_trivial_loss += rslts[ego]['trivial_loss']

    print 'Total loss:', total_loss
    print 'Total Trivial loss:', total_trivial_loss
    print 'Done:', get_ts()
    return rslts

def learn_labels(ego):
    X = ego.M
    labels = ego.targets
    
    from sklearn.semi_supervised import label_propagation
    label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
    label_spread.fit(X, labels)

def create_features_by_node(ego, G):
    ego_friends = G.nodes()
    featuremap = kd.training_features()
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
    
    circles = kd.training_circles()[ego]

    # A circle for those not in any circles...    
    all_friends = set(M.index)
    matched_keys = set(circles.keys())
    unmatched = dict.fromkeys(list(all_friends.difference(matched_keys)), '__NO_CIRCLE__')
    circles.update(unmatched)        
    
    return M, circles

uniq_features = {'birthday',
 'education;classes;name',
 'education;concentration;id',
 'education;concentration;name',
 'education;degree;name',
 'education;school;id',
 'education;school;name',
 'education;type',
 'education;year;id',
 'education;year;name',
 'first_name',
 'hometown;id',
 'hometown;name',
 'in_clique',
 'languages;id',
 'languages;name',
 'last_name',
 'locale',
 'location;id',
 'middle_name',
 'work;employer;id',
 'work;employer;name',
 'work;end_date',
 'work;location;name',
 'work;position;id',
 'work;position;name',
 'work;start_date'}

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
