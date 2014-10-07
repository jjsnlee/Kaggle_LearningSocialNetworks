from os.path import isfile, join
from os import listdir
import pandas as pd

TRAINING_DIR = '../Data/Training'

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

def get_num_of_circles():
    tc = training_circles()
    return [(ego, len(set(v.values()))) for ego,v in tc.iteritems()]

def get_acc_as_series(rslts):
    return pd.Series([r['train_accuracy'] for r in rslts.values()])

