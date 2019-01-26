import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
import json

dtpath = '/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/Mydataset/data/'


def db_json_export(dtpath,export=None):
    subj_name = mpt.AlpNumSorter(os.listdir(dtpath))  # List of Subjects in the directory
    mydataset = []

    for subj in enumerate(subj_name):  # for every subject
        a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj[1]), 'json')
        acts = mpt.AlpNumSorter(a)
        acts_no_ext = mpt.AlpNumSorter(a_no_ext)
        for act in enumerate(acts_no_ext):
            mydataset.append(act[1])

    mydata = mpt.AlpNumSorter(mydataset)

    if export:
        with open('database_out.json', 'w') as outfile:
            json.dump(mydata, outfile)

    print(mydata)

    return mydata


my_db = db_json_export(dtpath, export=False)

print()