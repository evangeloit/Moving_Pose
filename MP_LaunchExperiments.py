import os
import PythonModel3dTracker.Paths as Paths

os.chdir('/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor')
dry_run = 0
experiments_exec = 'MP_Experiments.py'
experiments_path = os.path.join(os.getcwd(),experiments_exec)

print()

for i in range(2):
    command_ = "python {0} {1} {2}".format(experiments_path, i, dry_run)
    print "Calling:", command_
    package = os.environ['mvpd']
    print(package)
    os.system(command_)