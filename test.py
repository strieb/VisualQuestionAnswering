from VQAConfig import VQAConfig
from Environment import DATADIR
import pickle
import numpy as np
import random

config = VQAConfig()
gloveFile = '%s/Database/%s.pickle' % (DATADIR, 'glove.6B.100d')
with open(gloveFile, 'rb') as fp:
    gloveIndex = pickle.load(fp)

def nearest(word, values):
    bestWord = ''
    bestDistance = 1000000
    for word2, values2 in gloveIndex.items():
        dist = np.sum(np.square(values-values2))
        if word != word2 and dist < bestDistance:
            bestWord = word2
            bestDistance = dist
    return bestWord, bestDistance


print(nearest('rain',gloveIndex['rain']-gloveIndex['water']+gloveIndex['ice']))