import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from scipy.spatial.distance import cdist
import numpy as np
import math
import matplotlib.pyplot as plt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
from Moving_Pose_Descriptor import confmat as cfm

evmat = np.load('evmat.npy')

new1 = np.empty((132, 132), dtype=float)

for iRow in range(0, 132):
    for iCol in range(0, 132):
        iSub1 = iRow % 12
        iAct1 = int(math.floor(iRow / 12))

        iSub2 = iCol % 12
        iAct2 = int(math.floor(iCol / 12))

        new1[iRow][iCol] = evmat[iSub1][iSub2][iAct1][iAct2]

print()