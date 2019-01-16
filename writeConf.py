import numpy as np
import cv2
import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from Moving_Pose_Descriptor import databaseModify

# Create a black image
# img = np.zeros((512,512,3), np.uint8)
conf = np.load('db_opencv_conf.npy')
print()
def load_images_from_folder(folder , destanation_path ,conf, isub, iact):
    images = []

    # Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 70)
    fontScale = 1
    fontColor = (255, 0, 127)
    lineType = 2

    path = os.listdir(folder)
    path.sort()
    last = databaseModify.db_lengthOfSequence(conf, isub, iact)
    path = path[0:last]
    count = 0

    for filename in path:
        img = cv2.imread(os.path.join(folder, filename))
        cv2.putText(img, 'Confidence: ' + str(conf[count, 5]),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Display the image
        cv2.imshow("img", img)
        # cv2.waitKey(0)
        # Save image
        count += 1
        cv2.imwrite(destanation_path+ "/"+ str(count) +"_out.jpg", img)

        if img is not None:
            images.append(img)

    return images

dtpath = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/"
dest_path = "/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/test_sequences/"

subj_name = mpt.AlpNumSorter(os.listdir(dtpath))  # List of Subjects in the directory
# print(subj_name)
subj_name = subj_name[0:2]
# Num of actions in subject's path
a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj_name[0]), 'json')
a_no_ext = mpt.AlpNumSorter(a_no_ext)
a_no_ext = a_no_ext[0:3]
num_of_acts = len(mpt.AlpNumSorter(a_no_ext))

for subj in range(0, len(subj_name)):  # for every subject
    a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj_name[subj]), 'json')
    acts = mpt.AlpNumSorter(a)
    acts_no_ext = mpt.AlpNumSorter(a_no_ext)

    for act in range(0, num_of_acts):
        folder_path = os.path.join(dtpath, subj_name[subj], acts_no_ext[act])
        destanation_path = os.path.join(dest_path, subj_name[subj], acts_no_ext[act])
        images = load_images_from_folder(folder_path, destanation_path, conf, subj, act)
        print(subj, act)