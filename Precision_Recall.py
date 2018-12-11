import numpy as np
import os


conf = np.load('eval_mat.npy')
pred_gt_conf = np.zeros((11, 11), dtype=int) # rows = predictec, cols = ground truth

# majority vote
ncorrect = 0
for iRow in range(0, 132):
    #conf[iRow][iRow] = 1000000 # remove diagonal

    answers = np.full(12, 11, dtype=int) # response per subject

    subjects = range(0, 12)
    subjects = np.setdiff1d(subjects, np.array([iRow % 12]))

    for iSubject in subjects:
        subscripts = range(iSubject, 132, 12)

        answers[iSubject] = np.argmin(conf[iRow][subscripts])

    counts = np.bincount(answers)
    answer = np.argmax(counts)
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

print("Accuracy: ", accuracy)
print()
#exit()

# pred_gt_conf = np.zeros((11, 11), dtype=int) # rows = predictec, cols = ground truth
#
# # minimum of average errors
# ncorrect = 0
# for iRow in range(0, 132):
#     #conf[iRow][iRow] = 1000000 # remove diagonal
#
#     avg_err = np.zeros(11, dtype=float)
#
#     # remove currect subject from comparison
#     subjects = range(0, 12)
#     subjects = np.setdiff1d(subjects, np.array([iRow % 12]))
#
#     for iAction in range(0, 11):
#         for iSubject in subjects:
#             avg_err[iAction] += conf[iRow][iAction * 12 + iSubject]
#
#     answer = np.argmin(avg_err)
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

#exit()
# print()