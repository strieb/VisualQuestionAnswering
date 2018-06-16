import csv
import numpy as np
import pickle

with open('C:/ml/VQA/Database/glove.6B.100d.txt', encoding='utf8') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=' ',quoting=csv.QUOTE_NONE)
    index = {}
    for row in csvReader:
        arr = np.ndarray(shape=(100,))
        for i in range(100):
            arr[i] = float(row[i+1])
        index[row[0]] = arr

with open('C:/ml/VQA/Database/glove100.pickle', 'wb') as fp:
    pickle.dump(index, fp)
