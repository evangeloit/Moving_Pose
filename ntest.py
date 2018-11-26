import numpy as np
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
import matplotlib.pyplot as plt

actions = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11","S12"]
d = np.load('avg_cscores.npy')
img_view = np.reshape(d,(12,1))

new_arr = np.empty((12, 1), np.dtype(np.float32))

# for ind1 in new_arr
#     for ind2 in new_arr:


# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.imshow(img_view, interpolation='nearest',cmap=None, aspect='auto')
# # cax = ax.matshow(img_view,cmap=None, aspect='auto')
# fig.colorbar(cax, orientation="horizontal")
# # ax.xaxis.tick_top()
# ax.set_yticks(np.arange(img_view.shape[0]))
# ax.set_yticklabels(actions)
# ax.set_xticklabels("")
# ax.set_ylabel('subjects',va='top')
# ax.set_xlabel('average %')
#
# # ax.autoscale(enable=True, axis='both', tight=True)
# plt.tight_layout()
# for (i, j), z in np.ndenumerate(img_view):
#     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
#
# ax.set_title("Average Performance by subject in mhad")
# plt.show()
collabel = ['average']

im, cbar = heatmap(img_view, actions,collabel,
                   cmap='jet')

texts = annotate_heatmap(im, valfmt="{x:.2f} ")
plt.axes().set_aspect('auto')
plt.title('Average Performance for every Subject\n Mhad_dataset')
plt.show()

print()
# subj_name = ['S01']
# ax1 = ['avg', 'subjects']
# # print(type(d))
# # d.reshape()
# mpt.plot_confusion_matrix(img_view, classes=subj_name, normalize=False, title='mhad avg score', axs=ax1)
# plt.show()
# #
# plt.show()
