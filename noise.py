import numpy as np

def addnoise(age, age_uncertainty_in_frames):

    noisyAge = np.random.normal() * age_uncertainty_in_frames + age

    # print(abs(age-noisyAge))
    return  np.floor(noisyAge)