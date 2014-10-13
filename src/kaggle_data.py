from os.path import join
from os import listdir
import pandas as pd
import networkx as nx
import json
import os
import zipfile
from json.encoder import JSONEncoder

DATA_DIR = '../Data'
TRAINING_DIR = join(DATA_DIR, 'Training')
EGONET_DIR = join(DATA_DIR, 'egonets')

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
        print 'loaded features...'
    return _featuremap.copy()

from collections import defaultdict
_ego_circles = {}
def training_circles():
    global _ego_circles
    if not _ego_circles:
        print 'Loading ego circles...'
        for f in listdir(TRAINING_DIR):
            if not f.endswith('.circles'):
                continue
            
            fname = join(TRAINING_DIR, f)
            circles = {}
            for line in open(fname):
                circle, _friends = line.split(':')
                friends = _friends.strip().split(' ')
                circles[circle] = friends

            ego = int(f.replace('.circles', ''))
            _ego_circles[ego] = circles
        #return circles
    return _ego_circles.copy()

def processed_circles(ego):
    _circles = training_circles()[ego]
    circles = defaultdict(set)
    for c, vals in _circles.iteritems():
        #circles.update(dict([(int(v),c) for v in vals]))
        for v in vals:
            circles[v].add(c)
            #circles[int(v)].add(c)
    return circles

def get_ego_cliques(ego):
    #if ego in [5881]:
    if ego==5881:
        print 'In get_ego_cliques, skipping ego', ego
        return {}
    
    ego_cliques_dmp = join(DATA_DIR, 'cliques', 'cliques_%s.zip'%ego)

    if os.path.exists(ego_cliques_dmp):
        print 'Loading cliques for ego:', ego
        with zipfile.ZipFile(ego_cliques_dmp, mode='r') as zf:
            ego_cliques = []
            for f in zf.namelist():
                ego_cliques+=json.loads(zf.read(f))
    else:
        print 'Processing cliques: nx.find_cliques, ego:', ego
        G = load_ego_graph(ego)
        # this can take some time...
        
        # http://pymotw.com/2/zipfile/
        with zipfile.ZipFile(ego_cliques_dmp, mode='w') as zf:
            fileno = 1
            ego_cliques = []
            for idx, clqs in enumerate(nx.find_cliques(G)):
                if idx%100000==0 and ego_cliques:
                    _write_cliques_file(zf, fileno, ego_cliques)
                    fileno += 1
                    ego_cliques = []
                ego_cliques.append(clqs)
            _write_cliques_file(zf, fileno, ego_cliques)
    return ego_cliques

def _write_cliques_file(zf, fileno, ego_cliques):
    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED
    json_rslt = json.dumps(ego_cliques, ensure_ascii=False, indent=True)
    zf.writestr('files_%s.json'%(fileno), json_rslt, compress_type=compression)

def get_ego_kclique_communities(ego):
    if ego in [5881, 12800]:
        print 'In get_ego_kclique_communities, skipping ego', ego
        return {}

    ego_kcc_dmp = join(DATA_DIR, 'cliques', 'kcc_%s.zip'%ego)
    if os.path.exists(ego_kcc_dmp):
        with zipfile.ZipFile(ego_kcc_dmp, mode='r') as zf:
            ccs = json.loads(zf.read('files1.json'))
    else:
        ego_cliques = get_ego_cliques(ego)
        print 'Processing k-clique communities: nx.find_cliques, ego: %s, # cliques: %s' %\
            (ego, len(ego_cliques))
        G = load_ego_graph(ego)
        ccs = [list(cc) for cc in nx.k_clique_communities(G, 6, cliques=ego_cliques)]
        try:
            import zlib
            compression = zipfile.ZIP_DEFLATED
        except:
            compression = zipfile.ZIP_STORED
        json_rslt = json.dumps(ccs, ensure_ascii=False, indent=True)
        with zipfile.ZipFile(ego_kcc_dmp, mode='w') as zf:
            zf.writestr('files1.json', json_rslt, compress_type=compression)
    return ccs

_training_data=None
def get_training_data():
    global _training_data
    if _training_data is None:
        tc = training_circles()
        d = {}
        for ego,circles in tc.iteritems():
            x = [set([int(_v) for _v in v]) for v in circles.values()]
            d[ego] = x
            friends_in_circles = reduce(lambda s,a:s.union(a), x, set())
            G = load_ego_graph(ego)
            friends_not_in_circles = set(G.nodes()).difference(friends_in_circles)
            x.append(friends_not_in_circles)
        _training_data = d
    return _training_data

def load_ego_graph(ego):
    fname = join(EGONET_DIR, str(ego)+'.egonet')
    G = read_nodeadjlist(fname)
    return G

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

import matplotlib.pyplot as plt
import numpy as np

def plot_network_metrics():
    tc = training_circles()
    _len_per_circle = [(k, [len(x) for x in c.values()]) for k,c in tc.iteritems()]
    len_per_circle = [l2 for l1 in _len_per_circle for l2 in l1[1]]
    
    print 'MIN:', np.min(len_per_circle)
    
    nrows = 2
    n = 1
    
    x,orig_y = zip(*_len_per_circle)
    idx = range(len(x))

    # the histogram of the data
#     num_bins = np.max(len_per_circle)
#     plt.subplot(nrows, 1, n)
#     _n,_bins,_patches = plt.hist(len_per_circle, num_bins, 
#                                  #normed=1, 
#                                  facecolor='green', alpha=0.5)
#     plt.xlabel('# People in Circle')
#     plt.title(r'Members in Circles')
#     plt.grid(True)
# 
#     n+=1
#     plt.subplot(nrows, 1, n)
#     y = [np.mean(_y) for _y in orig_y]
#     plt.bar(idx, y, alpha=0.5)    
#     plt.xticks(idx, x, rotation=70)
#     plt.title(r'Avg # of Members in Circles')
#     plt.xlabel('Circles')
#     plt.ylabel('Avg # of Members')
#     plt.grid(True)
#     n+=1

    plt.subplot(nrows, 1, n)
    y = np.array([len(_y) for _y in orig_y])
    plt.bar(idx, y, alpha=0.5)
    plt.xticks(idx, x, rotation=70)
    plt.title(r'# of Circles per Ego')
    plt.grid(True)
    
    n+=1
    plt.subplot(nrows, 1, n)
    ccs = []
    for ego,_ in _len_per_circle:
        G = load_ego_graph(ego)
        #ncc = len(nx.connected_components(G))
        # This can get a bit heavy calc wise as well
        #ego_cliques = get_ego_cliques(ego)
        #ncc = len(list(nx.k_clique_communities(G, 6, cliques=ego_cliques)))
        ncc = len(get_ego_kclique_communities(ego))
        ccs.append(ncc)
    ccs = np.array(ccs)
    plt.bar(idx, ccs-y, alpha=0.5, facecolor='red')
    plt.xticks(idx, x, rotation=70)
    plt.title(r'Connected Components - Actual Circles')
    plt.ylabel('Difference')
    plt.grid(True)
    
    plt.tight_layout()    
    plt.show()

def save_results(label, rslts):
    fname = join(DATA_DIR, label+'.json')
    with open(fname, 'w') as fh:
        fh.write(json.dumps(rslts, ensure_ascii=False, 
                            cls=JSONContentEncoder, 
                            indent=True))
def load_results(label):
    fname = join(DATA_DIR, label+'.json')
    rslts = json.loads(open(fname, 'r').read())
    # convert the keys to ints for convenience
    return dict([(int(k),v) for k,v in rslts.iteritems()])

class JSONContentEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Index):
            return list(obj)
        else:
            return obj

def analyze_features_for_ego(rslts, ego_ds):
    ego_rslts = rslts[ego_ds.ego]
    ntf = ego_rslts['non_trivial_features']
    M = ego_ds.M
    trivf = set(M.columns).difference(ntf)
    X = M.ix[:,ntf]
    Y = M.ix[:,trivf]
    return X,Y
def analyze_features(rslts):
    from kaggle_ego_cache import get_ego
    egos = training_circles().keys()
    non_triv = []
    triv = []
    for i, ego in enumerate(egos):
        ego_ds = get_ego(ego)
        N,T=analyze_features_for_ego(rslts, ego_ds)
        Nsum = N.sum()
        Tsum = T.sum()
        non_triv.append((Nsum.mean(), Nsum.std()))
        triv.append((Tsum.mean(), Tsum.std()))
        
        print 'Ego: %s, NT(%.2f, min:%i, max:%i), T(%.2f, min:%i, max:%i)' % \
            (ego, Nsum.mean(), Nsum.min(), Nsum.max(),
             Tsum.mean(), Tsum.min(), Tsum.max())

    return
    plt.subplot(1, 1, 1)
    width = .5
    idx = np.arange(len(egos))
    nt_mns, nt_stds = zip(*non_triv)
    bar1 = plt.bar(idx, nt_mns, width, alpha=0.5, yerr=nt_stds)
    t_mns, t_stds = zip(*triv)
    bar2 = plt.bar(idx+width, t_mns, width, alpha=0.5, yerr=t_stds, facecolor='red')
    
#     grp_mns = zip(nt_mns, t_mns)
#     grp_stds = zip(nt_stds, t_stds) 
#     # From http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
#     # Bar graphs expect a total width of "1.0" per group
#     # Thus, you should make the sum of the two margins
#     # plus the sum of the width for each entry equal 1.0.
#     # One way of doing that is shown below. You can make
#     # The margins smaller if they're still too big.
#     margin = 0.05
#     width = (1.-2.*margin)/len(egos)
#     for i, vals in enumerate(grp_mns):
#         print "plotting: ", vals
#         # The position of the xdata must be calculated for each of the two data series
#         xdata = idx+margin+(i*width)
#         # Removing the "align=center" feature will left align graphs, which is what
#         # this method of calculating positions assumes
#         _rects = plt.bar(xdata, vals, width,
#                          yerr=grp_stds[i]
#                          )
    
    plt.legend((bar1[0], bar2[0]), ('Non-trivial', 'Trivial'))
    plt.xticks(idx, egos, rotation=70)
    #plt.title(r'')
    plt.ylabel('Counts')
    plt.grid(True)
    
    plt.subplots_adjust(left=0.525)
    #plt.tight_layout()    
    plt.show()
    

def get_num_of_circles():
    tc = training_circles()
    return [(k, len(c.values())) for k,c in tc.iteritems()]    
#     tc = training_circles()
#     def flatten(vals):
#         allv = set()
#         for v in vals:
#             allv = allv.union(v)
#         return allv
#     #return [(ego, len(flatten(v.values()))) for ego,v in tc.iteritems()]
#     return [(ego, len(flatten(v.values()))) for ego,v in tc.iteritems()]
#     #return [(ego, len(set(v.values()))) for ego,v in tc.iteritems()]

# def get_acc_as_series(rslts):
#     return pd.Series([r['train_accuracy'] for r in rslts.values()])

def get_loss(rslts):
    loss = pd.Series([r['loss'] for r in rslts.values()])
    trivial_loss = pd.Series([r['trivial_loss'] for r in rslts.values()])
    print 'loss=', loss.sum(), ', trivial_loss=', trivial_loss.sum()
    #return loss,trivial_loss
