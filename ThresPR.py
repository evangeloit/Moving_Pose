import numpy as np
# from sklearn.preprocessing import normalize
# conf_1 = np.load('eval_mat.npy')
# conf = normalize(conf_1, axis=0, norm='max')

def belongsto(idSubject, idAction, label, thres, conf):
    scores = np.empty(11, dtype=float)

    i = 0
    for iSubject in range(0, 12):
        if iSubject != idSubject:
            score = conf[idAction * 12 + idSubject][label * 12 + iSubject]
            scores[i] = score
            i += 1

    return np.mean(scores) < thres