# import json
import os
# with open('dat_dict.json') as f:
#     data = json.load(f)
#
# for sets in data['datasets']:
#     print(sets)

def datasets_list(dtpath):
    # dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
    os.chdir(dtpath) # Mhad Dataset directory
    subj_name = os.listdir(os.getcwd()) # List of Subjects in the directory
    # print(subj_name)

    actions = []
    for subj in range(0,len(subj_name)):#for every subject
        acts = os.listdir(os.path.join(os.getcwd(), subj_name[subj]))
        actions.extend(acts)
        for act in range(0,len(acts)): #for every action of a subject
            pass
    return actions
# print(actions)

