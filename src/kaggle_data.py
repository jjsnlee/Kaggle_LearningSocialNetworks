from os.path import join
from os import listdir
import pandas as pd
import networkx as nx
import json
import os
import zipfile

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

def get_formatted_circles(ego):
    return [set([int(_v) for _v in v]) for v in training_circles()[ego].values()]

def processed_circles(ego):
    _circles = training_circles()[ego]
    circles = defaultdict(set)
    for c, vals in _circles.iteritems():
        #circles.update(dict([(int(v),c) for v in vals]))
        for v in vals:
            circles[v].add(c)
            #circles[int(v)].add(c)
    return circles

#_cliques = {}
def get_ego_cliques(ego):
    #cliques = [set(c) for c in nx.find_cliques(G)]
#     global _cliques
#     if ego not in _cliques:
#         if training_only:
#             egos = training_circles().keys()
#             cliques_dmp = join(DATA_DIR, 'cliques_train.zip')
#         else:
#             egos = [f.replace('.egonet') for f in listdir(EGONET_DIR) 
#                     if f.endswith('.egonet')]
#             cliques_dmp = join(DATA_DIR, 'cliques_test.zip')
    #if ego in [5881]:
    if ego==5881:
        print 'In get_ego_cliques, skipping ego', ego
        return {}
    
    ego_cliques_dmp = join(DATA_DIR, 'cliques', 'cliques_%s.zip'%ego)

    if os.path.exists(ego_cliques_dmp):
        print 'Loading cliques for ego:', ego
        zf = zipfile.ZipFile(ego_cliques_dmp, mode='r')
        ego_cliques = json.loads(zf.read('files.json'))
        #_cliques = json.loads(open(cliques_dmp, 'r').read())
        #_cliques = pickle.load(open(cliques_dmp, 'rb'))
    else:
        print 'Processing cliques: nx.find_cliques'
        print 'ego:', ego
        fname = join(EGONET_DIR, str(ego)+'.egonet')
        G = read_nodeadjlist(fname)
        # this can take some time...
        ego_cliques = list(nx.find_cliques(G))
        # http://pymotw.com/2/zipfile/
        try:
            import zlib
            compression = zipfile.ZIP_DEFLATED
        except:
            compression = zipfile.ZIP_STORED    
        json_rslt = json.dumps(ego_cliques, ensure_ascii=False, indent=True)
        #with open(cliques_dmp, 'w') as fh:
        #    fh.write(json_rslt)
        with zipfile.ZipFile(ego_cliques_dmp, mode='w') as zf:
            zf.writestr('files.json', json_rslt, compress_type=compression)

    return ego_cliques

def load_ego_graph(ego):
    fname = join('../Data/egonets', str(ego)+'.egonet')
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

def get_num_of_circles():
    tc = training_circles()
    def flatten(vals):
        allv = set()
        for v in vals:
            allv = allv.union(v)
        return allv
    #return [(ego, len(flatten(v.values()))) for ego,v in tc.iteritems()]
    return [(ego, len(flatten(v.values()))) for ego,v in tc.iteritems()]
    #return [(ego, len(set(v.values()))) for ego,v in tc.iteritems()]

def get_acc_as_series(rslts):
    return pd.Series([r['train_accuracy'] for r in rslts.values()])

