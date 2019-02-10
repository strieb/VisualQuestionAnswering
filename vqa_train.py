import json
from collections import Counter
import re
from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import numpy as np
from VQAGenerator import VQAGenerator

import VQAModel
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageOps
from keras.models import load_model
from random import randint
from Environment import DATADIR, TEST_NAME
from typing import NamedTuple
import time

class Result(NamedTuple):
    epoch: int
    test_accuracy: float
    train_accuracy: float


training_generator = VQAGenerator(True, batchSize=512, imageType=None)


if not os.path.exists(DATADIR+'/Results/'+TEST_NAME):
    os.mkdir(DATADIR+'/Results/'+TEST_NAME)

model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength, training_generator.gloveEncoding())
# model = load_model(DATADIR+'/Results/'+TEST_NAME+'/model.keras')

prediction_generator = VQAGenerator(False, batchSize=512, imageType=None, predict= True)
eval_model = VQAModel.evalModel(model)

results = []

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
for i in range(24):
    model.fit_generator(training_generator, epochs=1,workers=4)
    print("Test set")
    prediction, heat = eval_model.predict_generator(prediction_generator,workers=1)
    test_accuracy = prediction_generator.evaluate(prediction)
    print("Training set")
    training_generator.predict = True
    prediction, heat = eval_model.predict_generator(training_generator,workers=1, steps=128)
    train_accuracy = training_generator.evaluate(prediction)
    training_generator.predict = False
    results.append(Result(epoch=i+1,test_accuracy=test_accuracy,train_accuracy=train_accuracy))

    result_str = ""
    for result in results:
        result_str += "{0:2d}, {1:6.4f}, {2:6.4f}\n".format(result.epoch,result.test_accuracy,result.train_accuracy)

    with open(DATADIR+'/Results/'+TEST_NAME+'/results-'+timestamp+'.txt', "w") as text_file:
        text_file.write(result_str)

    model.save(DATADIR+'/Results/'+TEST_NAME+'/model.keras')

del(training_generator)


# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)