import numpy as np
import random

# TODO: normalize confusion matrix
confusion_matrix = np.load('eval_mat_new.npy')

nSubjects = 9
nActions = 4

np.fill_diagonal(confusion_matrix, float('inf'))
class_perf = np.zeros((nActions, 3), dtype=float)
heatmap = np.zeros((nActions, nActions))
heatmapBinary = np.zeros(4)


for iter in range(0, 5000):

    indices = [random.randrange(0, 8), random.randrange(9, 17), random.randrange(18, 26), random.randrange(27, 35)]

    for action in range(0, nActions):
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        for row in enumerate(confusion_matrix):

            row_elements = row[1]
            row_index = row[0]

            # Random Elements not include the current row number
            row_rand_elements = row_elements[[indices[0] + (indices[0] == row_index) % nSubjects,
                                            indices[1] + (indices[1] == row_index) % nSubjects,
                                            indices[2] + (indices[2] == row_index) % nSubjects,
                                            indices[3] + (indices[3] == row_index) % nSubjects]]

            running_class = row_index / nSubjects

            predict = np.argmin(row_rand_elements, axis=0)

            # Update Heatmap
            heatmap[running_class][predict] += 1.0

            groundtruth = (action == row_index / nSubjects)# -> true/false (should)
            predict = (predict == action)

            if predict == groundtruth:
                heatmapBinary[action] += predict


            # if groundtruth:
            #     if groundtruth == predict: # (groundtruth: yes, predictor: yes)
            #         tp += 1
            #     else:                      # (groundtruth: yes, predictor: no)
            #         fn += 1
            # else:  # negative
            #     if groundtruth != predict: # (groundtruth: no, predictor: yes)
            #         fp += 1
            #     else:                      # (groundtruth: no, predictor: no)
            #         tn += 1

            if groundtruth:
                if predict:  # (groundtruth: yes, predictor: yes)
                    tp += 1
                else:  # (groundtruth: yes, predictor: no)
                    fn += 1
            else:  # negative
                if predict:  # (groundtruth: no, predictor: yes)
                    fp += 1
                else:  # (groundtruth: no, predictor: no)
                    tn += 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp + 0.001)
        rec = tp / (tp + fn)

        class_perf[action][0] += acc
        class_perf[action][1] += prec
        class_perf[action][2] += rec

# Performance per class
class_perf = class_perf/iter
class_perf_round = np.round(class_perf, decimals=4)
# Average performance
avgPerf = np.mean(class_perf, axis=0)

print "Performance per Class\n"

print "Action1: ", class_perf_round[0, :]
print "Action2: ", class_perf_round[1, :]
print "Action3: ", class_perf_round[2, :]
print "Action4: ", class_perf_round[3, :]

print"\n"
print"Average performance: Acc: %.3f" % avgPerf[0]," Prec: %.3f" % avgPerf[1], "Recall: %.3f" % avgPerf[2]
print
