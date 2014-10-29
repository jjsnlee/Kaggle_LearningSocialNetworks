import networkx as nx
import pandas as pd
import numpy as np
import kaggle_data as kd
import os
from os.path import join
#import re
import socialCircles_metric as evaler

def train(ds, method='gmm', covar_type='diag', include_model=False, 
          uf=False,
          feature_threshhold=20,
          verbose=False):

    labels = ds.labels
    targets = ds.friend_targets
    samples = ds.M
    # Remove columns with less than 5 incidents 
    samples = samples[samples.columns[samples.sum()>feature_threshhold]]

    #samples = samples[samples.T.sum()>0]
    #targets = targets[samples.T.sum()>0]
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
    
    #print 'samples:', samples.shape
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
            #circle_rslts = circle_rslts[1:]
            
#         elif method=='km':
#             from sklearn import cluster
#             model = cluster.KMeans(n_clusters=n_components)
#             model.fit(samples)
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
        rslts['circle_rslts'] = circle_rslts
    
    return rslts

def eval_model(ds, circle_rslts):
    circled_friends = set()
    other_circles = []
    largest_circle = (0,None)
    for circle in circle_rslts:
        curr_circle = set()
        other_circles.append(curr_circle)
        for users in circle:
            curr_circle.update([int(m) for m in users if m]) # could be 0, as padded for the multi-index
        circled_friends = circled_friends.union(curr_circle)
        if len(curr_circle)>largest_circle[0]:
            largest_circle = (len(curr_circle), curr_circle)

    all_friends = set(ds.friends)
    largest_circle = largest_circle[1].union(all_friends.difference(circled_friends))
    #other_circles.append(all_friends.difference(circled_friends))
    gCircles = kd.get_training_data()[ds.ego]
    loss = evaler.loss1(gCircles, other_circles)
    print 'loss=', loss
    trivial_loss = evaler.loss1(gCircles, [all_friends])
    print 'trivial_loss=', trivial_loss
    return loss, trivial_loss

#r=kwk.main(feature_threshhold=3,clique_subset_size=20,uf=False,
#     egofilter=lambda x:x not in [9947,3735,23299,18005,15672,1357,5881,12800,26492,345,3059,4829,16203])

class EgoDataSet():
    def __init__(self, ego, by='edge2', clique_subset_size=10):
        self.ego = ego
        G = kd.load_ego_graph(ego)
        self.G = G
        
        assert by in ('node', 'edge2')
        connected_cmps = len(list(nx.connected_components(G)))
        
        #M = self.read(ego, by, clique_subset_size)
        M = None
        if M is None:
            if by == 'node':
                M = create_features_by_node(ego, G)
            elif by == 'edge2':
                M = create_features_by_edge2(ego, G, clique_subset_size)
            self.write(M, ego, by, clique_subset_size)
            
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
        
        #self.n_components = len(self.labels)
        self.n_components = connected_cmps+1

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
    def friends(self):
        return self.G.nodes()
    @property
    def shape(self):
        return self.M.shape
    def read(self, ego, by, clique_subset_size):
        fname = join(kd.DATA_DIR, 'egofeaturesM', '%s_%s_%s.pkl'%(ego, by, clique_subset_size))
        if os.path.exists(fname):
            return pd.read_pickle(fname)
        return None
    def write(self, M, ego, by, clique_subset_size):
        fname = join(kd.DATA_DIR, 'egofeaturesM', '%s_%s_%s.pkl'%(ego, by, clique_subset_size))
        M.to_pickle(fname)

def learn_number_clusters(ds, method='ap'):
    if method=='ap':
        from sklearn import cluster, covariance
#         _, labels = cluster.affinity_propagation(ds.M)
#         return labels
#     elif method=='ap2':
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

def create_features_by_edge2(ego, G, clique_subset_size=10):
    print 'About to generate cliques'
    from itertools import combinations

    featuremap = kd.training_features()    
    ccs = [set(c) for c in nx.connected_components(G)]
    m = {}
    def_dtype = ','.join(['i']*clique_subset_size)
    ncliques = 0
    max_clique_len = 0

    thresh = 0 
    comb_upper_bound = kd.get_comb_upper_bound(ego, clique_subset_size)
    if comb_upper_bound > 1e8:
        print 'Skipping ego, due to upper bound of %.2fM combinations.' \
            % (comb_upper_bound/1e6)
        return
    elif comb_upper_bound > 1e4:
        thresh = 1e4/comb_upper_bound
        print 'Will use threshold of [%.4f] to downsample, due to upper bound of %.2fM combinations.' \
            % (thresh, comb_upper_bound/1e6)
    
    # Some egos can run up to 8.4 million iterations, so the extra work with 
    # the generators and memory minimization 
    for cliq in kd.get_ego_cliques(ego):
        curr_size_cs = clique_subset_size
        if len(cliq)<curr_size_cs:
            curr_size_cs = len(cliq)
            dt = ','.join(['i']*curr_size_cs)
        else:
            dt = def_dtype
            
        #comb_upper_bound = comb(len(cliq),curr_size_cs)
        # from http://numpy-discussion.10968.n7.nabble.com/itertools-combinations-to-numpy-td16635.html
        for cliq_slice in np.fromiter(combinations(cliq, curr_size_cs), dtype=dt, count=-1):
            if thresh > 0 and np.random.random()>thresh:
                continue
            
            cliq_slice.sort()
            sample = tuple(cliq_slice)
            if sample not in m:
                max_clique_len = max(max_clique_len, len(sample))
                ncliques+=1
                if ncliques%500000==0:
                    print 'Processed %s cliques'%ncliques
                m[sample] = union_features([featuremap[user] for user in cliq_slice])
                cliq_slice = set(cliq_slice)
                in_cc = False
                for cidx, cc in enumerate(ccs):
                    # no difference, meaning all the elements in cliq are also in cc
                    if not cliq_slice.difference(cc):
                        m[sample]['cc_'+str(ego)+'_'+str(cidx)] = 1
                        in_cc = True
                if not in_cc:
                    m[sample]['cc_none'] = 1

    print '# of cliques:', ncliques

    #m = dict([('_'.join(map(str,k)),v) for k,v in m.iteritems()])
    def iterM(m, max_clique_len):
        for k in m.keys():
            v = m[k]
            del m[k]
            if len(v)==0:
                yield None, None
            else:
                if len(k)<max_clique_len:
                    k = list(k)
                    while len(k)<max_clique_len:
                        k.append(0)
                    k = tuple(k)
                ts = pd.Series(v)
                yield k,ts.to_sparse()

    def createM(rows):
        M = pd.DataFrame(rows)
        #M = M.to_sparse()
        return M.T

    rows = {}
    Ms = []
    for idx,(key,row) in enumerate(iterM(m, max_clique_len)):
        if idx%30000==0 and rows:
            M = createM(rows)
            print 'M.shape:', M.shape
            Ms.append(M)
            rows = {}
        if key:
            rows[key]=row
    if rows:
        Ms.append(createM(rows))
    #print '# of sparse rows:', len(rows)
    #M = pd.DataFrame(rows)
    M = pd.concat(Ms)
    print 'Created DataFrame M', M.shape
    #M = M.T
    M.fillna(0, inplace=True)
    return M

def union_features(users):
    row = {}
    feature_keys = set([k for u in users for k in u.keys()])
    for k in feature_keys:
        try:
            if k.startswith('id;'):
                continue
            # Need to explicitly make this 0/1
            feat_vals = np.array([u[k] for u in users])
            #if int(np.mean(feat_vals))==feat_vals[0]:
            if feat_vals.mean()==feat_vals[0]:
                row[str(k)+'='+str(users[0][k])] = 1
        except:
            pass
    return row

def _get_circles(ego, all_friends_edges, raw_circles=None):
    if raw_circles is None:
        raw_circles = kd.processed_circles(ego)
    from collections import defaultdict
    overlaps = defaultdict(int)
    emptyset = frozenset()
    print 'Checking circles'
    circles = {}
    for users in all_friends_edges:
        user_edge_circles = [raw_circles.get(u, emptyset) for u in users]
        intersect = reduce(lambda a,b: a.intersection(b), user_edge_circles) 
        if len(intersect)>0:
            if len(intersect)>1:
                overlaps[len(intersect)]+=1
                #print len(intersect), intersect 
            circles[users] = list(intersect)[0]
    
    print 'overlaps:', overlaps
    return circles

def main(method='gmm', by='edge2', verbose=False, feature_threshhold=20,
         clique_subset_size=10, 
         size_cutoff=200,
         uf=None, refresh=True, egofilter=None): #uf=uniq_features, 
    print 'Starting:', get_ts(), 'using', method, ', by:', by
    rslts = {}
    egocircles = kd.training_circles().keys()
    import kaggle_ego_cache as kec
    if refresh:
        kec._egocache={}
    
    if egofilter:
        if isinstance(egofilter, list):
            egocircles = [ec for ec in egocircles if ec in egofilter]
        else:
            egocircles = filter(egofilter, egocircles)
        print 'As a result of the filter, will just do', egocircles
    
    clique_info = kd.get_ego_cliques_mean_stats()
    egocircles.sort()
    total_loss = total_trivial_loss = 0
    print '# egocircles:', len(egocircles)
    for i, ego in enumerate(egocircles):
        print '-'*80
        print 'Executing for %s (#%s)' %(ego, i)
        
        if clique_info[ego]['total']>=size_cutoff:
            print 'Skipping ego [%s], which has %s cliques.' % (ego, clique_info[ego]['total'])
            continue
        try:
            #ds = kec.get_ego(ego, by=by, clique_subset_size=clique_subset_size)
            ds = EgoDataSet(ego, by=by, clique_subset_size=clique_subset_size)
            print 'Initialized ego'
#             from sklearn.grid_search import GridSearchCV
#             logistic = linear_model.LogisticRegression()
#             pca = decomposition.PCA()
#             pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
#             estimator = GridSearchCV(pipe, {}
#                                      #dict(pca__n_components=n_components, logistic__C=Cs)
#                                      )
            rslts[ego] = train(ds, method, uf=uf, 
                               verbose=verbose, 
                               feature_threshhold=feature_threshhold)
            total_loss += rslts[ego]['loss']
            total_trivial_loss += rslts[ego]['trivial_loss']
        except Exception as e:
            import traceback
            st = traceback.format_exc()
            print 'Problem running [%s]:\n%s' % (ego, st)

    print 'For ft=%s css=%s sc=%s' % (feature_threshhold, clique_subset_size, size_cutoff)
    print 'Total loss:', total_loss
    print 'Total Trivial loss:', total_trivial_loss
    print 'Done:', get_ts()
    return rslts

# def learn_labels(ego):
#     X = ego.M
#     labels = ego.targets
#     from sklearn.semi_supervised import label_propagation
#     label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
#     label_spread.fit(X, labels)

def create_features_by_node(ego, G):
    """
    The affinity propagation problem also uses an edge model? 
    What is the appeal of using the edges instead of nodes to figure out clusters? 
    """
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
