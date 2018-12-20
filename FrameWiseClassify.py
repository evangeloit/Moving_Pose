# import numpy as np
from scipy.spatial import distance as dst
from Moving_Pose_Descriptor import MP_tools2 as mpt


def classframe(fv_in,database,k, dataPercent=None):

    """dataPercent : input argument a list of random numbers generated based on the database.
    Compare every incoming frame(feature_vector) with  a randomly picked percentage of database."""

    dist = []

    if dataPercent is None:

        for iframe in range(0, database.shape[0]):
            d = [dst.euclidean(fv_in, database[iframe][0]), database[iframe][1], database[iframe][2]]
            dist.append(d)
            # print "ok"
    else:

        for iframe in dataPercent:
            d = [dst.euclidean(fv_in, database[iframe][0]), database[iframe][1], database[iframe][2]]
            dist.append(d)

    # sort by distance
    dist.sort(key=mpt.getkey)

    # find most occurent class // most_common_occurence returns :(class, confidence )
    confidence_tuple = mpt.most_often_occurence(dist[0:k])

    return confidence_tuple