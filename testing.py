import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Moving_Pose_Descriptor import MP_tools2 as mpt

ws200_noise = np.load('best_params_for_High_noise_test2.npy')

# ws200_noise[:, 2:4] *= 0.1

sorted_bestparams = list(ws200_noise)

sorted_bestparams.sort(key=lambda x: x[0], reverse=True)

ws200_noise= np.array(sorted_bestparams)

average = np.average(ws200_noise[0:5,:],axis=0)

print()
# ws_noise = np.load('best_windowSize_noisy.npy')
# sigma = 1
# w = 4  # windowSize
# t = (((w - 1) / 2) - 0.5) / sigma  # truncate
#
# gauss_filtered_points = mpt.GaussFilter3dPoints(ws_noise, sigma, t)# kx = ws[:,1]
# print()
#
# # smoothed_points = gaussian_filter1d(gauss_filtered_points[:, 0], sigma=sigma, truncate=t)
# plt.plot(gauss_filtered_points[0:178,1], gauss_filtered_points[0:178,2], linestyle='-', marker='o', color='b', label='noise 0')
# plt.plot(gauss_filtered_points[181:358,1], gauss_filtered_points[181:358,2], linestyle='-', marker='o', color='g', label='noise 5')
# plt.plot(gauss_filtered_points[361:538,1], gauss_filtered_points[361:538,2], linestyle='-', marker='o', color='r', label='noise 10')
# plt.plot(gauss_filtered_points[541:719,1], gauss_filtered_points[541:719,2], linestyle='-', marker='o', color='c', label='nosie 15')
#
# plt.xlim(100, 600)
# plt.xlabel('wsizes')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper right')
# plt.title("Accuracy for wsizes with Noise in frames: " + "\nwindow size: 100:1000:5\nage_uncertainty_inFrames : 0:15:5\nFixed parameters = [k: 6, wPos: 0.9, wVel: 0.1, wAcc: 0.45]")
# # plt.savefig(goal_dir + actions[iAction])
# plt.show()
# plt.close('all')

print()