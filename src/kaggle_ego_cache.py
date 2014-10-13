
_egocache = {}
def get_ego(ego, by='edge'):
    global _egocache
    key = (ego,by)
    if key not in _egocache:
        from kaggle_work import EgoDataSet
        _egocache[key] = EgoDataSet(ego, by=by)
    return _egocache[key]

 
