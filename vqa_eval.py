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

from Environment import DATADIR
from VQAConfig import VQAConfig

def evalConfig(config: VQAConfig):

    #model = load_model(DATADIR+'/Results/'+config.testName+'/model_'+config.modelIdentifier+'_'+str(config.epoch)+'.keras')
  
    prediction_generator = VQAGenerator(False,True, config)
    model = VQAModel.createModel(prediction_generator.questionLength, prediction_generator.answerLength, prediction_generator.gloveEncoding(), config)
    explain_model = VQAModel.explainModel(model)

    prediction,top, heat1,heat2,norm = explain_model.predict_generator(prediction_generator,workers=8, steps=1)
    print(norm)
    acc = prediction_generator.evaluate(top)
    print(acc)
    for xx in range(20):
        k = randint(0,heat1.shape[0])
        prediction_generator.print(k,prediction[k],heat1[k])


evalConfig(VQAConfig(imageType= 'preprocessed_res_24',
    testName='gru_res_24_cnn_nobatch',
    gloveName='glove.42B.300d',
    gloveSize=300,
    dropout=True,
    augmentations=None,
    modelIdentifier='Mar-02-2019_1650',
    epoch=13,
    stop=16,
    batchSize=16
    )
)


# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)