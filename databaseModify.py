import random


# @fullDatabase = array[N][[features...], subject, action]
def reduceDatabase(fullDatabase, trainCount, testCount):
    indices = range(0, len(fullDatabase))
    random.shuffle(indices)

    return (
        fullDatabase[indices[0 : trainCount]],
        fullDatabase[indices[trainCount : trainCount + testCount]]
    )

