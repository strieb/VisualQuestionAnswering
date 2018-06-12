import json
from collections import Counter
import re
from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import VQAModel
from matplotlib import pyplot as plt
import os
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from PIL import Image, ImageOps


dataDir		='C:/ml/VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='val2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/' %(dataDir, dataSubType)


model = VQAModel.createModelInception()

directory = os.fsencode(imgDir)
i = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imgPath = os.path.join(imgDir, filename)

        img = load_img(imgPath, target_size=(299,299), interpolation='bilinear')
        #img = Image.open(imgPath)
        #img = ImageOps.fit(img,(299,299),Image.ANTIALIAS)

        id = int(filename[-16:-4])
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array,axis=0)
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        np.save(imgDir+"preprocessed_avg/"+str(id),predictions[0])
        if i % 100 == 0:
            print(i)
        i+= 1