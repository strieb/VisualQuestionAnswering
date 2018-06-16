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
from VQAGenerator import VQAGenerator
from keras.models import load_model
from random import randint


training_generator = VQAGenerator(True, batchSize=512)
validation_generator = VQAGenerator(False, batchSize=512)

model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength)
# model = load_model('C:/ml/VQA/Database/model.keras')

model.fit_generator(training_generator, epochs=8)
model.save('C:/ml/VQA/Database/model.keras')

prediction_generator = validation_generator
prediction_generator.predict = True
# model = load_model('C:/ml/VQA/Database/model.keras')
# prediction_generator = VQAGenerator(False, batchSize=64, predict=True)

prediction = model.predict_generator(prediction_generator)
prediction_generator.evaluate(prediction)


for xx in range(20):
    k = randint(0,len(prediction_generator.good))
    prediction_generator.print(k,prediction[k])

# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)