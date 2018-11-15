import os
import sys
import itertools

import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as MTR

package_path = os.environ['mvpd']

landmarks_source = ['gt', 'detections', 'openpose', 'json_openpose'][2] # [3]

#Viz Parameters
visualize_params = {'enable':False,
             'client': 'opencv','labels':True, 'depth':True, 'rgb':True,
             'wait_time':0}
assert visualize_params['client'] in ['opencv','blender']


# Objective Params.
objective_params = {
    'enable': True, #pf_params['pf']['n_particles'] > pf_params['pf']['levmar_particles'],
    'objective_weights':{'rendering':1.,
                         'primitives':0.,
                         'collisions':0.
                         },
    'depth_cutoff': 500,
    'bgfg_type': 'depth'
}



#Results Paths
results_path = "Human_tracking/results_normal/"
results_cam_inv = "Human_tracking/results_camera_invariant/"


# 
# model_names = ["mh_body_male_customquat_950", "mh_body_male_customquat", "mh_body_male_customquat",
#                "mh_body_male_customquat_950", "mh_body_male_customquat", "mh_body_male_customquat",
#                "mh_body_male_customquat_950", "mh_body_male_customquat_950", "mh_body_male_customquat_950",
#                "mh_body_male_customquat_950", "mh_body_male_customquat_950", "mh_body_male_customquat_950",
#                "mh_body_male_customquat_950", "mh_body_male_customquat_950", "mh_body_male_customquat_950",
#                "mh_body_male_customquat_950", "mh_body_male_customquat_950", "mh_body_male_customquat_950",
#                "mh_body_male_customquat_950", "mh_body_male_customquat",      "mh_body_male_customquat",
#                "mh_body_male_customquat"]
# datasets = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04',
#             'mhad_s04_a04', 'mhad_s05_a04', 'mhad_s06_a04',
#             'mhad_s07_a04', 'mhad_s08_a04', 'mhad_s09_a01',
#             'mhad_s09_a02', 'mhad_s09_a03', 'mhad_s09_a04',
#             'mhad_s09_a05', 'mhad_s09_a06', 'mhad_s09_a07',
#             'mhad_s09_a08', 'mhad_s09_a09', 'mhad_s09_a10',
#             'mhad_s09_a11', 'mhad_s10_a04', 'mhad_s11_a04',
#             'mhad_s12_a04'
# ]

model_names = ["mh_body_male_customquat","mh_body_male_customquat",
               "mh_body_male_customquat","mh_body_male_customquat",
               "mh_body_male_customquat","mh_body_male_customquat",
               "mh_body_male_customquat"
               ]
# model_names = ["mh_body_male_customquat"]

datasets = [
    'mhad_s03_a04','mhad_s06_a04',
    'mhad_s12_a04','mhad_s11_a04',
    'mhad_s05_a04','mhad_s02_a04',
    'mhad_s10_a04'
]

#model_names = ["mh_body_male_customquat"] * len(datasets)
#model_names = ["mh_body_male_custom"]
#datasets = ["mhad_ammar"]

# Command line parameters.
sel_rep = int(sys.argv[1])
dry_run = int(sys.argv[2])

# Experiment Parameters.
n_iterations = range(1)
dataset_model_pairs = [(d, m) for (d, m) in zip(datasets, model_names)]
# ransac = [[0.0, 0.0]]
ransac = [[0.1, 0.3]]
levmar_particles = [1]
# n_particles = [0]
n_particles = [1]
filter_occluded = [False]
filter_history = [False]
# filter_occluded = [True, False]
# filter_history = [True, False]


# Experiments loop.
rep = 0
for (dataset, model_name), i in \
        itertools.product(dataset_model_pairs, n_iterations):
    # if (fo == True) or (fh == True):
    # print(sel_rep)
    # if rep == 100:
        if rep == sel_rep:
            # if p < lp: p = lp
            # Results Filename
            res_filename = os.path.join(Paths.results, results_path + "/{0}_{1}_it{2}.json")
            res_filename = res_filename.format(dataset, model_name, i)

            new_res = os.path.join(Paths.results, results_cam_inv + "/{0}_{1}_inv_it{2}.json")
            new_res = new_res.format(dataset, model_name, i)
            print '{0} -- {1} -- d: {2}, m:{3}, results:{4}'. \
                format(rep, i, dataset, model_name, res_filename)
            if dry_run:
                pass
            else:
                import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs
                import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
                import PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools as vt

                model3d, model_class = tt.ModelTools.GenModel(model_name)
                params_ds = tt.DatasetTools.Load(dataset)

                # PF Initialization
                hmf_arch_type = "2levels"
                pf_params = pfs.Load(model_name, model_class, hmf_arch_type)
                pf_params['pf']['n_particles'] = 1#p
                pf_params['pf']['init_state'] = tt.DatasetTools.GenInitState(params_ds, model3d)
                pf_params['meta_mult'] = 1
                pf_params['pf_listener_flag'] = False
                pf_params['pf']['enable_smart'] = True
                pf_params['pf']['smart_pf']['smart_particles'] = 1#lp
                pf_params['pf']['smart_pf']['enable_blocks'] = False
                pf_params['pf']['smart_pf']['enable_bounds'] = True
                pf_params['pf']['smart_pf']['ceres_report'] = False
                pf_params['pf']['smart_pf']['max_iterations'] = 50
                # pf_params['pf']['smart_pf']['interpolate_num'] = 3
                pf_params['pf']['smart_pf']['filter_occluded'] = False #fo
                pf_params['pf']['smart_pf']['filter_occluded_params'] = {
                    'thres': 0.2,
                    'cutoff': 100,
                    'sigma': 0.2
                }
                pf_params['pf']['smart_pf']['filter_random'] = True
                pf_params['pf']['smart_pf']['filter_random_ratios'] = [0.1, 0.3] #r
                pf_params['pf']['smart_pf']['filter_history'] =False #fh
                pf_params['pf']['smart_pf']['filter_history_thres'] = 100

                #Performing tracking
                mesh_manager = tt.ObjectiveTools.GenMeshManager(model3d)
                model3dobj, decoder, renderer = tt.ObjectiveTools.GenObjective(mesh_manager, model3d, objective_params)
                visualizer = vt.Visualizer(model3d, mesh_manager, decoder, renderer)
                decoder = visualizer.decoder
                grabbers = tt.DatasetTools.GenGrabbers(params_ds, model3d, landmarks_source)
                pf, rng = tt.ParticleFilterTools.GenPF(pf_params, model3d, decoder)

                results = tt.TrackingLoopTools.loop(params_ds, model3d, grabbers, pf,
                                                    pf_params['pf'], model3dobj, objective_params,
                                                    visualizer, visualize_params)

                # parameters = {}
                # parameters['ransac'] = r
                # parameters['levmar_particles'] = lp
                # parameters['n_particles'] = p
                # parameters['filter_occluded'] = fo
                # parameters['filter_history'] = fh
                # results.parameters = parameters
                # results.save(res_filename)

                if res_filename is not None:
                    results.save(res_filename)

                # Camera Invariant states

                res = MTR.ModelTrackingResults()
                res.load(res_filename)
                states = res.get_model_states(model_name)

                for fr in states:
                    for index in range(0, 7):
                        states[fr][index] = 0
                        if index == 6:
                            states[fr][index] = 1

                    res.add(fr, model_name, states[fr])

                res.save(new_res)

        rep += 1







