
_egocache = {}
def get_ego(ego, by='edge', clique_subset_size=10):
    global _egocache
    key = (ego,by,clique_subset_size)
    if key not in _egocache:
        from kaggle_work import EgoDataSet
        _egocache[key] = EgoDataSet(ego, by=by, clique_subset_size=clique_subset_size)
    return _egocache[key]


