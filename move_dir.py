import os
from os.path import join
from os import listdir, rmdir
from shutil import move
import glob
import shutil


import re
import sys


path = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/S009'

subdir_names = ['R01']
filelist = os.listdir(path)
src = []
dstn = []

#Make source and destenation dirs
for f in filelist:
    dest = os.path.join(path, f)
    dstn.append(dest)
    for file in listdir(dest):
        source =os.path.join(dest,file)
        for n in subdir_names:
            if n in file: src.append(source)
#Move Files
# print(src)
for num in range(0,len(src)):
    for filename in listdir(join(dstn[num], subdir_names[0])):
        # print(filename)
        move(join(src[num],filename),join(dstn[num],''))

#Remove Empty dirs
for num2 in range(0,len(src)):
    sub_folders_pathname = src[num2]
    sub_folders_list = glob.glob(sub_folders_pathname)
    for sub_folder in sub_folders_list:
    # print(sub_folder)
        shutil.rmtree(sub_folder)