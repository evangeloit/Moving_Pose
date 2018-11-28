import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
from Moving_Pose_Descriptor import confmat as cfm




def plot_heatmaps(avg_cscore, c_score, subj_name, params=None):

    if params[0] == 1:

        goal_dir = os.path.join(params[1])

        # plot 1 subj vs all
        name1 = '1vsAll'
        axis1 = ['subjects', 'subjects']
        mpt.plot_confusion_matrix(c_score, classes=subj_name, normalize=False, title='mhad class score per subject', axs=axis1)
        plt.savefig(goal_dir + name1)
        plt.close('all')

        # # Plot Average Performance in dataset
        name2 = 'avg_performance_dtset'
        col_label = ['average']
        img_view = np.reshape(avg_cscore, (12, 1))
        im, cbar= heatmap(img_view, subj_name, col_label, cmap='jet')
        texts = annotate_heatmap(im, valfmt="{x:.2f} ")
        plt.axes().set_aspect('auto')
        plt.title('Average Performance for every Subject\n Mhad_dataset')
        plt.savefig(goal_dir + name2)
        plt.close('all')

# 1 vs all / Average dataset performance save figs path
savefig_avg = os.getcwd() + "/plots/conf_matrix/"
params_avg = [1, savefig_avg]

subj_name = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11","S12"]
avg_cscore = np.load('avg_cscores.npy')
c_score = np.load('c_score.npy')

plot_heatmaps(avg_cscore, c_score, subj_name, params=params_avg)
# plt.show()
# plt.show()
