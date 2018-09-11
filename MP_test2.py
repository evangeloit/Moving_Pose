import numpy as np
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

input_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/alex_far_01_ldm.json"
model_name = "mh_body_male_custom"

##### My implementation / Import json #####
with open(input_dir) as f:
    data = json.load(f)

##### Check First frame for insuficient landmarks #####
LandSize = len(data['landmarks']['0'][model_name])
LandDesired = len(data['landmark_names'][model_name])

if LandSize == LandDesired:
    print("import:", LandSize, "Landmarks")
else:
    print('Insufficient Landmarks at import')
    exit()

##### Create 3D points List #####

points = []

for i in range(0, len(data['landmarks'])):
    strnum = str(i)
    # print(strnum)
    frames = []
    for item in data['landmarks'][strnum][model_name]:
        frames.extend(item)

    points.append(frames)

### Convert to numpy array for calculations ####

# i=len(range(0, len(data['landmarks']))) ### frames =1000
# j=len(data['landmarks'][strnum][model_name]) ### landmarks = 14
# k=3

p3d = np.array(points)
# tt = np.array(range(0,i*k*j)).reshape((i,j*k))

print(p3d.shape)
# print(len(points))

#### Derivatives Calculation ####

sz = p3d.shape #size of p3d array
StartFrame = 2 # Start from 3rd Frame's 3D points# pt1 = np.array([])
pt0 = []
pt1 = []
pt2 = []
ptm1 = []
ptm2 = []
for fr in range(StartFrame, sz[0]-StartFrame):
    Pt0 = []
    Pt1 = []
    Pt2 = []
    Ptm1 = []
    Ptm2 = []
    for cl in range(0, sz[1]):
        Pt0.extend([p3d[fr,cl]])
        Pt1.extend([p3d[fr+1, cl]])
        Pt2.extend([p3d[fr+2, cl]])
        Ptm1.extend([p3d[fr-1, cl]])
        Ptm2.extend([p3d[fr-2, cl]])

    pt0.append(Pt0)
    pt1.append(Pt1)
    pt2.append(Pt2)
    ptm1.append(Ptm1)
    ptm2.append(Ptm2)

Pt0 = np.array(pt0)
Pt1 = np.array(pt1)
Pt2 = np.array(pt2)
Ptm1 = np.array(ptm1)
Ptm2 = np.array(ptm2)


## Acc / Vec

vec = Pt1 - Ptm1
acc = Pt2 + Ptm2 - 2*Pt0

print(vec.shape)
print(acc.shape)

# Feature Vector
plt.interactive(True)
f_v = np.concatenate ((Pt0,vec,acc), axis=1)
print(f_v.shape)
print(f_v[996,0:])
# print(Pt0[996,:])


# PLOTS 
plt.imshow(f_v)
plt.show(block=True)
# plt.pause(100)
# plt.gca().set_aspect('auto')
# plt.savefig('filename.png', dpi=600)
