from VQAConfig import VQAConfig
from Environment import DATADIR
import pickle
import numpy as np
import random

config = VQAConfig()
gloveFile = '%s/Database/%s.pickle' % (DATADIR, 'glove.42B.300d')
with open(gloveFile, 'rb') as fp:
    gloveIndex = pickle.load(fp)


for word, values in gloveIndex.items():
    if random.randint(0,100) == 0:
        bestWord = ''
        bestDistance = 1000000
        for word2, values2 in gloveIndex.items():
            dist = np.sum(np.square(values-values2))
            if word != word2 and dist < bestDistance:
                bestWord = word2
                bestDistance = dist
        print(word+" "+bestWord+" "+str(bestDistance))
