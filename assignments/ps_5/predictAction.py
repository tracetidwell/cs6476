import numpy as np
from utils import normalized_euclidean_distance


def predictAction(testMoments, trainMoments, trainLabels):

    min_dist = float('inf')
    predictedLabel = -1
    variances = np.std(trainMoments, axis=0)**2

    for train_example, label in zip(trainMoments, trainLabels):
        dist = normalized_euclidean_distance(train_example.reshape(1, -1), testMoments, variances)

        if dist < min_dist:
            min_dist = dist
            predictedLabel = label

    return predictedLabel[0]
