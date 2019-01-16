import numpy as np
import cv2
import os
# Create a black image
# img = np.zeros((512,512,3), np.uint8)
conf = np.load('db_opencv_conf.npy')
print()
def load_images_from_folder(folder , destanation_path ,conf):
    images = []

    # Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (80, 40)
    fontScale = 1
    fontColor = (50, 200, 50)
    lineType = 2

    path = os.listdir(folder)
    path.sort()
    count =0
    # conf = [0.1]*len(path)
    for filename in path:
        img = cv2.imread(os.path.join(folder, filename))
        cv2.putText(img, 'Confidence: '+ str(conf[count , 5]),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Display the image
        cv2.imshow("img", img)
        cv2.waitKey(0)
        # Save image
        count += 1
        cv2.imwrite(destanation_path + str(count) +"_out.jpg", img)

        if img is not None:
            images.append(img)

    return images

folder_path = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/S01/mhad_s01_a01"
destanation_path = "/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/test_sequences/"

images = load_images_from_folder(folder_path, destanation_path, conf)