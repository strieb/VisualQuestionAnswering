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
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math

from Environment import dataDir
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'val2014'
annFile = '%s/Annotations/%s%s_%s_annotations.json' % (dataDir, versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (dataDir, versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/' % (dataDir, dataSubType)

i = 0
directory = os.fsencode(imgDir)


model = VQAModel.createModelInception((427, 619, 3))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imgPath = os.path.join(imgDir, filename)
        id = int(filename[-16:-4])
        img = load_img(imgPath)
        width, height = img.size
        if(width >= height):
            img = img.resize((619, 427), resample=Image.BICUBIC)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            # img_array = np.tile(img,(32,1,1,1))
            img_array = np.expand_dims(img_array, axis=0)
            # plt.imshow((img_array[0] + 1)/2)
            # plt.show()
            predictions = model.predict(img_array)
            pred = predictions[0].reshape(24,2048)
            np.save(imgDir+"preprocessed_24/"+str(id), pred)
            if i % 1000 == 0:
                print(predictions[0].shape)
                print(pred.shape)
                print(i)
            i += 1

model = VQAModel.createModelInception((619, 427, 3))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imgPath = os.path.join(imgDir, filename)
        id = int(filename[-16:-4])
        img = load_img(imgPath)
        width, height = img.size
        if(width < height):
            img = img.resize((427, 619), resample=Image.BICUBIC)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            # img_array = np.tile(img,(32,1,1,1))
            img_array = np.expand_dims(img_array, axis=0)
            # plt.imshow((img_array[0] + 1)/2)
            # plt.show()
            predictions = model.predict(img_array)
            pred = predictions[0].reshape(24,2048)
            np.save(imgDir+"preprocessed_24/"+str(id), pred)
            if i % 1000 == 0:
                print(predictions[0].shape)
                print(pred.shape)
                print(i)
            i += 1
