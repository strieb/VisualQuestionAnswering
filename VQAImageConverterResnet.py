import json
from collections import Counter
import re
from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot as plt
import os
import VQAModel
from keras.applications.xception import decode_predictions, preprocess_input
# from keras.applications.inception_v3 import decode_predictions, preprocess_input
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math

from Environment import DATADIR
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
saveDir = 'resnext'
dataSubType = 'val2014'
imgDir = '%s/Images/%s/' % (DATADIR, dataSubType)


# 363, 555
# 427, 619
size1 = 224*2
size2 = 224*2


model = VQAModel.createModelResNet((size1, size2, 3))
model.summary()

i = 0
directory = os.fsencode(imgDir)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imgPath = os.path.join(imgDir, filename)
        id = int(filename[-16:-4])
        img = load_img(imgPath)
        width, height = img.size
        img = img.resize((size2, size1), resample=Image.BICUBIC)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        # img_array = np.tile(img,(32,1,1,1))
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        pred = predictions[0].reshape(49,2048)
        np.save(imgDir+saveDir+"/"+str(id), pred)
        if i < 1000 and i%100 == 0:
            print(i)
        if i % 1000 == 0:
            print(i)
        i += 1
        