import json
import re
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle


dataDir		='C:/ml/VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/preprocessed_avg/' %(dataDir, dataSubType)
databaseFile = '%s/Database/%s' % (dataDir, dataSubType)

directory = os.fsencode(imgDir)
i = 0
filelist = os.listdir(directory)
print(len(filelist))
images = np.ndarray(((len(filelist),2048)),dtype=np.float32)
index = {}
for i in range(len(filelist)):
    file = filelist[i]
    filename = os.fsdecode(file)
    idx = filename[:-4]
    image = np.load(imgDir+str(idx)+'.npy')
    images[i,:] = image
    index[idx] = i

np.save(databaseFile+"images",images)
with open(databaseFile+"imageindex.json", 'wb') as fp:
    json.dump(index, fp)

