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

training_generator = VQAGenerator(True, batchSize=64)
validation_generator = VQAGenerator(False, batchSize=64)
model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength)
#model = load_model('C:/ml/VQA/Database/model.keras')


model.fit_generator(training_generator, epochs=5, validation_data=validation_generator)
model.save('C:/ml/VQA/Database/model.keras')

# prediction = model.predict(validation_generator.single(556))[0]
# top = [(i,prediction[i]) for i in range(len(prediction))]
# top = sorted(top, key=lambda entry: entry[1])
# inv_map = {v: k for k, v in validation_generator.answerEncoding.items()}
# for entry in top[-5:]:
#     print('Result '+str(entry[0])+': '+str(entry[1])+", "+inv_map[entry[0]])