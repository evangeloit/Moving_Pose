import numpy as np

# BEST params for mhad : #[ 0.93  0.9   0.1   0.45  6.  ]

def wdistance(wv, fv1, fv2):
    return np.sum(wv * (fv1-fv2)**2)


def wvector(wpos, wvel, wacc):
    return np.tile(np.array([wpos, wpos, wpos, wvel, wacc]), 14)
