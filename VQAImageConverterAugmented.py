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
from keras.applications.inception_resnet_v2 import decode_predictions, preprocess_input
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator

from Environment import DATADIR
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
saveDir = 'augmented_res_24'
annFile = '%s/Annotations/%s%s_%s_annotations.json' % (DATADIR, versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (DATADIR, versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/' % (DATADIR, dataSubType)

i = 0
directory = os.fsencode(imgDir)

# 363, 555
# 427, 619
size1 = 427
size2 = 619

generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=10,
    zoom_range=[0.9,1]
    )
    

# model = VQAModel.createInceptionResNetFull((size1, size2, 3))
# model.summary()
# for file in os.listdir(directory):
#     for k in range(2):
#         filename = os.fsdecode(file)
#         if filename.endswith(".jpg"):
#             imgPath = os.path.join(imgDir, filename)
#             id = int(filename[-16:-4])
#             img = load_img(imgPath)
#             width, height = img.size
#             if(width >= height):
#                 img = img.resize((size2, size1), resample=Image.BICUBIC)
#                 img_array = img_to_array(img)
#                 img_array = preprocess_input(img_array)
#                 transformed = generator.random_transform(img_array)
#                 img_array = np.expand_dims(transformed, axis=0)

#                 # predictions = model.predict(img_array)
#                 # print(str(decode_predictions(predictions,top=5)))

#                 plt.imshow((transformed+1.0)/2.0)
#                 plt.show()
# exit()

def processImage(img_resized):
        img_array = img_to_array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = generator.random_transform(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        pred = predictions[0].reshape(24,1536)
        return pred

for k in range(4,8):
    print("starting batch "+str(k))
    model = VQAModel.createModelInceptionResNet((size1, size2, 3))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            imgPath = os.path.join(imgDir, filename)
            id = int(filename[-16:-4])
            img = load_img(imgPath)
            width, height = img.size
            if(width >= height):
                img_resized = img.resize((size2, size1), resample=Image.BICUBIC)
                pred = processImage(img_resized)
                np.save(imgDir+saveDir+"/"+str(id)+'_'+str(k), pred)
                if i < 1000 and i%100 == 0:
                    print(i)
                if i % 1000 == 0:
                    print(i)
                i += 1

    model = VQAModel.createModelInceptionResNet((size2, size1, 3))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            imgPath = os.path.join(imgDir, filename)
            id = int(filename[-16:-4])
            img = load_img(imgPath)
            width, height = img.size
            if(width < height):
                img_resized = img.resize((size1, size2), resample=Image.BICUBIC)
                pred = processImage(img_resized)
                np.save(imgDir+saveDir+"/"+str(id)+'_'+str(k), pred)
                if i < 1000 and i%100 == 0:
                    print(i)
                if i % 1000 == 0:
                    print(i)
                i += 1