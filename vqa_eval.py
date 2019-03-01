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

    model = load_model(DATADIR+'/Results/'+config.testName+'/model_'+config.modelIdentifier+'_'+str(config.epoch)+'.keras')

    prediction_generator = VQAGenerator(False,True, config)
    explain_model = VQAModel.explainModel(model)

    prediction,top, heat1,heat2 = explain_model.predict_generator(prediction_generator,workers=8, steps=50)
    acc = prediction_generator.evaluate(top)
    print(acc)
    for xx in range(20):
        k = randint(0,heat1.shape[0])
        prediction_generator.print(k,prediction[k],heat2[k])


evalConfig(VQAConfig(imageType= 'preprocessed_res_24',
    testName='gru_res_24_dropout_glove_augmented',
    gloveName='glove.42B.300d',
    gloveSize=300,
    dropout=True,
    augmentations=None,
    modelIdentifier='Mar-01-2019_1137',
    epoch=13,
    stop=16
    )
)


# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)