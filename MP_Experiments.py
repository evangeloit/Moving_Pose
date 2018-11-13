import os
from os.path import join
from os import listdir, rmdir
from shutil import move
import glob
import shutil
import re

import Moving_Pose_Descriptor.MP_tools2 as mpt

topf = ['S03']
childf = ['R05']
# top_path = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/'
top_path = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'

mpt.move_dir(top_path,topf,childf=childf)
# os.chdir(top_path)
#
#
# for name in topf:
#     path = top_path + name
#     print(path)
#     subdir_names = childf
#     filelist = os.listdir(path)
#     src = []
#     dstn = []
#
#     # Make source and destenation dirs
#     for f in filelist:
#         dest = os.path.join(path, f)
#         dstn.append(dest)
#         for file in listdir(dest):
#             source = os.path.join(dest, file)
#             for n in subdir_names:
#                 if n in file: src.append(source)
#     # Move Files
#     # print(src)
#     for num in range(0, len(src)):
#         for filename in listdir(join(dstn[num], subdir_names[0])):
#             # pass
#             # print(dstn[num])
#             move(join(src[num], filename), join(dstn[num], ''))
#
#     #Remove Empty dirs
#     for num2 in range(0, len(src)):
#         sub_folders_pathname = src[num2]
#         sub_folders_list = glob.glob(sub_folders_pathname)
#         for sub_folder in sub_folders_list:
#             shutil.rmtree(sub_folder)
#
#     #Rename Parent folders
#     for xt in os.listdir(path):
#         file_name, file_ext = os.path.splitext(xt)
#
#         # match = re.match(r"([A-Z]+)([0-9]+)", file_name, re.I)
#         # if match:
#         #     items = list(match.groups())
#
#         # f_num = items[1]
#         # f_word = items[0]
#
#         dt_name = 'mhad'
#
#         # print('{}_{}_{}{}'.format(dt_name,topf[0].lower(),f_word.lower(),f_num))
#         new_name = '{}_{}_{}'.format(dt_name, name.lower(), file_name.lower())
#         # print(os.getcwd()+'/'+topf[0]+'/'+file_name)
#         os.rename((os.getcwd() + '/' + name + '/' + file_name), (os.getcwd() + '/' + name + '/' + new_name))
#
#
#
# print('files moved to parent folder and parent folder is renamed!')
#
# # os.chdir('/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/')
# # # print(os.getcwd())
# # for xt in os.listdir(did):
# #     file_name, file_ext = os.path.splitext(xt)
# #
# #     match = re.match(r"([A-Z]+)([0-9]+)", file_name, re.I)
# #     if match:
# #         items = list(match.groups())
# #
# #     f_num = items[1]
# #     f_word = items[0]
# #
# #     dt_name = 'mhad'
# #
# #     # print('{}_{}_{}{}'.format(dt_name,topf[0].lower(),f_word.lower(),f_num))
# #     new_name = '{}_{}_{}{}'.format(dt_name, topf[0].lower(), f_word.lower(), f_num)
# #     # print(os.getcwd()+'/'+topf[0]+'/'+file_name)
# #     os.rename((os.getcwd()+'/'+topf[0]+'/'+file_name),(os.getcwd()+'/'+topf[0]+'/'+new_name))
#


