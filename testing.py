import numpy as np
import matplotlib.pyplot as plt

ws = np.load('best_windowSize.npy')

kx = ws[:,1]

plt.plot(ws[:,1], ws[:,0], linestyle='-', marker='o', color='b', label='recall')
# plt.plot(thresp, precs, linestyle='-', marker='o', color='g', label='precision')
plt.xlabel('wsizes')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.title("Accuracy/wsize: " + "\nwindow size: 5:1000:5")
# plt.savefig(goal_dir + actions[iAction])
plt.show()
plt.close('all')

print()