import json
from collections import Counter
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot as plt
import os
from keras.applications.xception import decode_predictions, preprocess_input
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math

from Environment import DATADIR
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
saveDir = 'preprocessed_xcep_24'
annFile = '%s/Annotations/%s%s_%s_annotations.json' % (DATADIR, versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (DATADIR, versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/' % (DATADIR, dataSubType)

i = 0
directory = os.fsencode(imgDir)

size1 = 427
size2 = 619

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imgPath = os.path.join(imgDir, filename)
        id = int(filename[-16:-4])
        img = load_img(imgPath)
        width, height = img.size
        if(width >= height):
            img = img.resize((size2, size1), resample=Image.BICUBIC)
            img_array = img_to_array(img)
            
            img_array = preprocess_input(img_array)
            # img_array = np.tile(img,(32,1,1,1))
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            
            print(str(decode_predictions(predictions,top=5)))
            plt.imshow(img)
            plt.show()
exit()