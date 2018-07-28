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
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from PIL import Image, ImageOps
from keras.models import load_model
from random import randint

# model = load_model('C:/ml/VQA/Database/model.keras')

training_generator = VQAGenerator(True, batchSize=512, imageType='preprocessed_24')

model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength, training_generator.gloveEncoding())


prediction_generator = VQAGenerator(False, batchSize=512, imageType='preprocessed_24', predict= True)
eval_model = VQAModel.evalModel(model)

for i in range(16):
    model.fit_generator(training_generator, epochs=1,workers=4)
    model.save('C:/ml/VQA/Database/model.keras')
    prediction, heat = eval_model.predict_generator(prediction_generator,workers=4)
    prediction_generator.evaluate(prediction)
del(training_generator)


# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)