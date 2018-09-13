import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


input_dir = "alex_far_01_ldm.json"
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

p3d = np.array(points)
sz_p3d = p3d.shape #size of p3d array
# print(p3d.shape)

## Gaussian Filter 5 by 1 in time dimension
sigma = 1
w = 5  # windowSize
t = (((w - 1) / 2) - 0.5) / sigma  # truncate
f_gauss = gaussian_filter1d(p3d[:, 0], sigma=sigma, truncate=t)

for x in range(1, sz_p3d[1]):
        f_gauss = np.column_stack((f_gauss, gaussian_filter1d(p3d[:, x], sigma=sigma, truncate=t)))

# Feature Vector with filtered coordinates [1001 x 126]
p3d_gauss = np.concatenate((f_gauss, p3d[:, sz_p3d[1]:p3d.shape[0]]), axis=1)

print(p3d.shape)
print(p3d_gauss.shape)

#### Derivatives Calculation ####

StartFrame = 2 # Start from 3rd Frame's 3D points# pt1 = np.array([])
pt0 = []
pt1 = []
pt2 = []
ptm1 = []
ptm2 = []
for fr in range(StartFrame, sz_p3d[0]-StartFrame):
    Pt0 = []
    Pt1 = []
    Pt2 = []
    Ptm1 = []
    Ptm2 = []
    for cl in range(0, sz_p3d[1]):
        Pt0.extend([p3d_gauss[fr,cl]])
        Pt1.extend([p3d_gauss[fr+1, cl]])
        Pt2.extend([p3d_gauss[fr+2, cl]])
        Ptm1.extend([p3d_gauss[fr-1, cl]])
        Ptm2.extend([p3d_gauss[fr-2, cl]])

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

## Feature Vector
f_v = np.concatenate((Pt0, vec, acc), axis=1)
z = np.copy(f_v[996, :])
z = np.matlib.repmat(z, 4, 1)
f_vec_fil = np.vstack((f_v, z))

print(f_vec_fil.shape)

# PLOTS
plt.plot(p3d[:, 3], label='Before Smooth')
plt.plot(p3d_gauss[:, 3], label='After Smooth')

plt.xlabel('frames')
plt.ylabel('X coord')
plt.title('X coordinate before &\nafter Gaussian smoothing')
plt.legend()
plt.show()
