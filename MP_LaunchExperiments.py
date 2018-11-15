import os
import PythonModel3dTracker.Paths as Paths

package_path = os.environ['mvpd'] #Moving Pose Descriptor Directory env variable

dry_run = 0
experiments_exec = 'MP_Experiments.py'
experiments_path = os.path.join(package_path,experiments_exec)

print(experiments_path)

for i in range(8):
    command_ = "python {0} {1} {2}".format(experiments_path, i, dry_run)
    print "Calling:", command_
    os.system(command_)