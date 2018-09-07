import numpy as np
import json
from scipy.ndimage import gaussian_filter

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
    frames=[]
    for item in data['landmarks'][strnum][model_name]:
        frames.extend(item)
        # print(item,i)
    points.append(frames)
# i=len(range(0, len(data['landmarks'])))
# j=len(data['landmarks'][strnum][model_name])
# k=3

tt=np.array(points)
#tt=np.array(range(0,i*k*j)).reshape((i,j*k))
print(points[998])
# print(len(points))




### Convert to numpy array for calculations ####
