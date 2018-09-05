import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as Mtr
import numpy as np
import json


input_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/alex_far_01_ldm.json"
model_name = "mh_body_male_custom"

##### Implementation with ModelTrackingResults scripts #####

# results = Mtr.ModelTrackingResults()
# results.load(input_dir)
# ldm3d = results.landmarks

# frames = -1
# for i in ldm3d:
#     print(ldm3d[i])
#     frames = frames + 1

# print(frames)

##### My implementation #####
with open(input_dir) as f:
    data = json.load(f)

##### Check First frame for insuficient landmarks #####
LandSize = len(data['landmarks']['0'][model_name])
LandDesired = len(data['landmark_names'][model_name])

if LandSize == LandDesired:
    print("You have Imported:", LandSize, "Landmarks")
else:
    print('Insufficient Landmarks at Import')
    exit()

##### Create Points List #####
points = []
for i in range(0,len(data['landmarks'])):
    strnum = str(i)
    # print(strnum)
    for item in data['landmarks'][strnum][model_name]:
            points.append(item)
            # print(point3d)

#### Convert to numpy array for calculations ###
point3d = np.matrix(points)
point3d = np.transpose(point3d)

#### Normalization of 3d points with Gaussian filter [5 1] ####
# GFpoint3d = gaussian_filter(point3d, sigma=1)


# print(len(point3d))
# print(point3d[0:,14]-point3d[0:,28])


##### Derivatives Calculation #####
sz = point3d.shape
# print(sz[1])
t1 = LandSize
t2 = 2*LandSize
StartFrame = 2*LandSize # Start from 3rd Frame's 3D points
count = 0

for col in range(StartFrame, sz[1] - StartFrame):
    for row in range(0, sz[0]):
        Pt0 = point3d[row, col] # Current Frame
        Pt1 = point3d[row, col + t1]
        Pt2 = point3d[row, col + t2]
        PtMinus1 = point3d[row, col - t2]
        PtMinus2 = point3d[row, col - t2]
        # print(PtMinus2)

        # Derivatives
        vel = Pt1 - PtMinus1
        acc = Pt2 + PtMinus2 - (2*Pt0)

        # print(vel,acc)



# how do i dynamically allocate the arrays
# Save Feature data into json...
# Calculate the First and Second Derivative
# Build the feature vector