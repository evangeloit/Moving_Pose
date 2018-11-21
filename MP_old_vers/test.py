import Moving_Pose_Descriptor.DatasetsCreateJson as dtl
import json
import os
dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
datasets = dtl.datasets_list(dtpath)


# with open(os.path.join(os.environ['mvpd'],"dataset.json")) as f:
#     datasets = list(json.load(f))
#
# k= len(datasets)
#
# mod_name = ["mh_body_male_customquat"]
#
# model_names = mod_name * k

# new_datalist = [ x for x in datasets if "a03" not in x ]

print(len(datasets))
# print(len(model_names))

print(datasets)
# print(model_names)
# print(len(new_datalist))

with open('dataset.json', 'w') as f:
    json.dump(datasets, f)



