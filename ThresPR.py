import numpy as np
# from sklearn.preprocessing import normalize
# conf_1 = np.load('eval_mat.npy')
# conf = normalize(conf_1, axis=0, norm='max')

def belongsto(idSubject, idAction, subjects, label, thres, conf):
    scores = np.empty(subjects, dtype=float)

    i = 0
    for iSubject in range(0, subjects):
        if iSubject != idSubject:
            score = conf[idAction * subjects + idSubject][label * subjects + iSubject]

            scores[i] = score
            i += 1

    return np.mean(scores) < thres