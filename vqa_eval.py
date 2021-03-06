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
    # model = VQAModel.createModel(prediction_generator.questionLength, prediction_generator.answerLength, prediction_generator.gloveEncoding(), config)
    eval_model = VQAModel.evalModel(model)

    top = eval_model.predict_generator(prediction_generator,workers=8)
    acc, results = prediction_generator.evaluate(top)
    with open(DATADIR+'/Results/'+config.testName+'/results-'+config.modelIdentifier+"_"+str(config.epoch)+'.json', 'w') as fp:
        json.dump(results, fp)
    print("Accuracy: "+ str(acc))


if __name__ == '__main__':
    evalConfig(VQAConfig(imageType= 'preprocessed_res_24',
        testName='inceptionResNet_noise_test',
        gloveName='glove.42B.300d',
        gloveSize=300,
        augmentations=8,
        imageFeaturemapSize=24,
        imageFeatureChannels=2048,
        modelIdentifier='Mar-08-2019_0747',
        trainvaltogether=False,
        epoch=14
        )
    )