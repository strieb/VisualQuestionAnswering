import json
from collections import Counter
import re
from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import numpy as np
from VQAGenerator import VQAGenerator
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import VQAModel
from matplotlib import pyplot as plt
import os
from keras.models import load_model
from random import randint

from Environment import DATADIR, TEST_NAME, IMAGE_TYPE, IMAGE_FEATURE_SIZE, IMAGE_FEATURE_STATE_SIZE
model = load_model(DATADIR+'/Results/'+TEST_NAME+'/model.keras')

prediction_generator = VQAGenerator(False, batchSize=512, imageType=IMAGE_TYPE, imageFeatureSize = IMAGE_FEATURE_SIZE, imageFeatureStateSize = IMAGE_FEATURE_STATE_SIZE, predict= True)
explain_model = VQAModel.explainModel(model)

prediction, heat = explain_model.predict_generator(prediction_generator,workers=8, steps=50)
# prediction_generator.evaluate(prediction)

for xx in range(20):
    k = randint(0,prediction.shape[0])
    prediction_generator.print(k,prediction[k],heat[k])

# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)