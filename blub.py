import pickle
import numpy as np
from collections import Counter

from Environment import DATADIR, GLOVE, GLOVE_SIZE

dataSubType = 'train2014'
databaseFile = '%s/Database/%s.pickle' % (DATADIR, dataSubType)

with open(databaseFile, 'rb') as fp:
    database = pickle.load(fp)

answers = [answer for answers in database['answers'] for answer in answers]
times = [answer for answer in answers if ' feet' in answer or  ' ft' in answer or  ' miles' in answer]
print(len(times)/len(answers))
common = Counter(times).most_common(100)
print(common)