import csv
import numpy as np
import pickle

from Environment import DATADIR
GLOVE = 'glove.6B.100d'
GLOVE_SIZE = 100

with open(DATADIR+'/Database/'+GLOVE+'.txt', encoding='utf8') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=' ',quoting=csv.QUOTE_NONE)
    index = {}
    idx = 0
    for row in csvReader:
        arr = np.ndarray(shape=(GLOVE_SIZE,))
        for i in range(GLOVE_SIZE):
            arr[i] = float(row[i+1])
        index[row[0]] = arr
        idx += 1
        # if idx > 100000:
        #     break

with open(DATADIR+'/Database/'+GLOVE+'.pickle', 'wb') as fp:
    pickle.dump(index, fp)
