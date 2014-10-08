from os.path import isfile, join
from os import listdir
import pandas as pd
import networkx as nx

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
    return _featuremap

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
            _circles = {}
            for line in open(fname):
                circle, _friends = line.split(':')
                friends = _friends.strip().split(' ')
                _circles[circle] = friends
            circles = defaultdict(set)
            for c, vals in _circles.iteritems():
                #circles.update(dict([(int(v),c) for v in vals]))
                for v in vals:
                    circles[v].add(c)
                    #circles[int(v)].add(c)

            ego = int(f.replace('.circles', ''))
            _ego_circles[ego] = circles
        #return circles
    return _ego_circles

import json
import os
import zipfile

#_cliques = {}
def get_ego_cliques(ego):
    # this can take some time...
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
    ego_cliques_dmp = join(DATA_DIR, 'cliques', 'cliques_%s.zip'%ego)
            
    if os.path.exists(ego_cliques_dmp):
        zf = zipfile.ZipFile(ego_cliques_dmp, mode='r')
        ego_cliques = json.loads(zf.read('files.json'))
        #_cliques = json.loads(open(cliques_dmp, 'r').read())
        #_cliques = pickle.load(open(cliques_dmp, 'rb'))

    else:
        print 'Processing cliques: nx.find_cliques'
        #egos = egos[:2]
        print 'ego:', ego
        fname = join(EGONET_DIR, str(ego)+'.egonet')
        G = read_nodeadjlist(fname)
        ego_cliques = list(nx.find_cliques(G))
        #_cliques = ego_cliques

#             output = open(cliques_dmp, 'wb')
#             # Pickle dictionary using protocol 0.
#             pickle.dump(_cliques, output)
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
            #zf.close()

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
    return [(ego, len(set(v.values()))) for ego,v in tc.iteritems()]

def get_acc_as_series(rslts):
    return pd.Series([r['train_accuracy'] for r in rslts.values()])

