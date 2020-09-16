import numpy as np
from Config import regular, byzantine

"""
Different Byzantine attacks, include:
same-value attacks, sign-flipping attacks,
sample-duplicating attacks (only conducted in non-iid case)

@:param workerPara : the set of workers' model parameters
"""


def same_value_attack(workerPara):
    for id in byzantine:
        workerPara[id] = 100 * np.ones((10, 784))
    return workerPara, '-sv'


def sign_flipping_attacks(workerPara):
    for id in byzantine:
        workerPara[id] *= -4
    return workerPara, '-sf'


def sample_duplicating_attack(workerPara):
    for id in byzantine:
        workerPara[id] = workerPara[regular[0]]
    return workerPara, '-noniid'
