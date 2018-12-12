import numpy as np
import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
import matplotlib.pyplot as plt

conf = np.load('eval_mat.npy')
# pred_gt_conf = np.zeros((11, 11), dtype=int) # rows = predictec, cols = ground truth
#
# # majority vote
# ncorrect = 0
# for iRow in range(0, 132):
#     #conf[iRow][iRow] = 1000000 # remove diagonal
#
#     answers = np.full(12, 11, dtype=int) # response per subject
#
#     subjects = range(0, 12)
#     subjects = np.setdiff1d(subjects, np.array([iRow % 12]))
#
#     for iSubject in subjects:
#         subscripts = range(iSubject, 132, 12)
#
#         answers[iSubject] = np.argmin(conf[iRow][subscripts])
#
#     counts = np.bincount(answers)
#     answer = np.argmax(counts)
#     gt = int(np.floor(iRow / 12))
#
#     pred_gt_conf[gt][answer] += 1
#
#     if answer == gt:
#         ncorrect += 1
#
# accuracy = float(ncorrect) / 132
#
# precision = np.zeros(11, dtype=float)
# recall    = np.zeros(11, dtype=float)
# colwise = np.sum(pred_gt_conf, axis=0)
#
# for iAction in range(0, 11):
#     precision[iAction] = float(pred_gt_conf[iAction][iAction]) / colwise[iAction]
#     recall[iAction]    = float(pred_gt_conf[iAction][iAction]) / 12.0
#
# print("Accuracy: ", accuracy)
# print()
#exit()

pred_gt_conf = np.zeros((11, 11), dtype=int) # rows = predictec, cols = ground truth

# minimum of average errors
ncorrect = 0
for iRow in range(0, 132):
    #conf[iRow][iRow] = 1000000 # remove diagonal

    avg_err = np.zeros(11, dtype=float)

    # remove currect subject from comparison
    subjects = range(0, 12)
    subjects = np.setdiff1d(subjects, np.array([iRow % 12]))

    for iAction in range(0, 11):
        for iSubject in subjects:
            avg_err[iAction] += conf[iRow][iAction * 12 + iSubject]

    answer = np.argmin(avg_err)
    gt = int(np.floor(iRow / 12))

    pred_gt_conf[gt][answer] += 1

    if answer == gt:
        ncorrect += 1

accuracy = float(ncorrect) / 132

precision = np.zeros(11, dtype=float)
recall    = np.zeros(11, dtype=float)
colwise = np.sum(pred_gt_conf, axis=0)

for iAction in range(0, 11):
    precision[iAction] = float(pred_gt_conf[iAction][iAction]) / colwise[iAction]
    recall[iAction]    = float(pred_gt_conf[iAction][iAction]) / 12.0

# exit()
print()
#
# #Plots
# subjects = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11","S12"]
# actions = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]
# actinput = []
# axlabel = ["groundtruth","predicted"]
# goal_dir = os.getcwd() + "/plots/conf_matrix/"
# figname = "train_test_conf"
#
# #Confusion Mat
# mpt.plot_confusion_matrix(pred_gt_conf, classes=actions, normalize=False, title='confusion matrix', axs=axlabel)
# plt.tight_layout()
# plt.savefig(goal_dir + figname)
# plt.close('all')
# # plt.show()
#
#
# #Precision
# # prec=np.zeros((11,1),dtype=float)
# precisionTrans = np.reshape(precision,(11,1))
# recallTrans = np.reshape(recall,(11,1))
#
# #prec2= np.append(prec,precisionTrans,axis=1)
#
# # fig2="recall"
# # rowlabel=['recall']
# # axlabel2=["recall","actions"]
#
# fig2="precision"
# rowlabel=['precision']
# axlabel2=["precision","actions"]
#
# im, cbar = heatmap(precisionTrans,actions , rowlabel, cmap='jet')
# texts = annotate_heatmap(im, valfmt="{x:.2f} ")
# plt.axes().set_aspect('auto')
# plt.savefig(goal_dir + fig2)
# plt.show()
#

print()