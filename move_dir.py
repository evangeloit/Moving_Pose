import os
from os.path import join
from os import listdir, rmdir
from shutil import move
import glob
import shutil


import re
import sys



# path = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/s01_a01/'
path = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/S009'

subdir_names = ['R01']
filelist = os.listdir(path)
src = []
dstn = []
# # Check for directories with images.
for f in filelist:
    dest = os.path.join(path, f)
    dstn.append(dest)
    for file in listdir(dest):
        source =os.path.join(dest,file)
        for n in subdir_names:
            if n in file: src.append(source)

for num in range(0,len(src)):
    for filename in listdir(join(dstn[num], subdir_names[0])):
        # print(src[num])
        move(join(src[num],filename),join(dstn[num],''))
        # sub_folders_pathname = src[num]
        # sub_folders_list = glob.glob(sub_folder_pathname)
        # for sub_folder in sub_folders_list:
        #     shutil.rmtree(sub_folder)