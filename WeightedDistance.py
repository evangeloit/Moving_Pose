import numpy as np

def wdistance(wv, fv1, fv2):
    return np.sum(wv * (fv1-fv2)**2)


def wvector(wpos, wvel, wacc):
    return np.tile(np.array([wpos, wpos, wpos, wvel, wacc]), 14)
