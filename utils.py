import copy


def convert_or_recurse(d, conversion_fn):
    """Walk a tree of nested dicts d, applying conversion_fn at leaves."""
    keys = copy.deepcopy(list(d.keys()))
    for k in keys:
        if type(d[k]) == dict:
            convert_or_recurse(d[k], conversion_fn)

        else:
            d[k] = conversion_fn(d, k)

    return d


def convert_to_sweep_format(d, k):
    """Convert to W&B sweep format."""
    return {'value': d[k]}
