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
from Environment import DATADIR
from typing import NamedTuple
import time
from VQAConfig import VQAConfig

def trainConfig(config: VQAConfig):
    training_generator = VQAGenerator(True,False, config)
    if config.modelIdentifier:
        model = load_model(DATADIR+'/Results/'+config.testName+'/model_'+config.modelIdentifier+'_'+str(config.epoch)+'.keras')
    else:
        model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength, training_generator.gloveEncoding(), config)

    model.get_layer('noise_layer').stddev = config.noise
    prediction_generator = VQAGenerator(False,True, config)
    eval_model = VQAModel.evalModel(model)


    if not os.path.exists(DATADIR+'/Results/'+config.testName):
        os.mkdir(DATADIR+'/Results/'+config.testName)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    result_str = str(config) +'\n\n'
    with open(DATADIR+'/Results/'+config.testName+'/results-'+timestamp+'.txt', "w") as text_file:
        text_file.write(result_str)

    best = 0
    for i in range(config.epoch+1,config.stop):
        model.fit_generator(training_generator, epochs=1,workers=6)
        model.save(DATADIR+'/Results/'+config.testName+'/model_'+timestamp+"_"+str(i)+'.keras')
        print("Test set")
        prediction = eval_model.predict_generator(prediction_generator,workers=6, steps=None if i>6 else 128)
        test_accuracy = prediction_generator.evaluate(prediction)
        print("Training set")
        training_generator.predict = True
        prediction = eval_model.predict_generator(training_generator,workers=6, steps=128)
        train_accuracy = training_generator.evaluate(prediction)
        training_generator.predict = False

        result_str += "{0:2d}, {1:6.4f}, {2:6.4f}\n".format(i,test_accuracy,train_accuracy)

        with open(DATADIR+'/Results/'+config.testName+'/results-'+timestamp+'.txt', "w") as text_file:
            text_file.write(result_str)

        if test_accuracy > best:
            print("best")
            best = test_accuracy

trainConfig(VQAConfig(imageType= 'resnext_24',
    testName='gru_resnext',
    gloveName='glove.42B.300d',
    gloveSize=300,
    dropout=True,
    augmentations=None,
    stop=23,
    gatedTanh=True,
    batchNorm=False,
    embedding='gru',
    imageFeaturemapSize=24,
    imageFeatureChannels=2048,
    noise=0
    )
)




# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)