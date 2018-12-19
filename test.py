import numpy as np


##### Most common element ###
nlist = [(100,2,2),(200,1,2), (300,2,3), (400,4,3), (500,1,3), (600,2,3), (700,2,3), (800,3,1), (900,4,1)]

def most_often_occurence(nlist):
    actions = []
    for t in nlist:
        actions.append(t[2])

    occurences = np.bincount(actions)
    i = np.argmax(occurences)

    return (i, float(np.amax(occurences)) / len(nlist))

print(most_often_occurence(nlist))

print()