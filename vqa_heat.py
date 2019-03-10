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
import numpy as np

def explainConfig(config: VQAConfig):

    model = load_model(DATADIR+'/Results/'+config.testName+'/model_'+config.modelIdentifier+'_'+str(config.epoch)+'.keras')
  
    prediction_generator = VQAGenerator(False,True, config)
    # model = VQAModel.createModel(prediction_generator.questionLength, prediction_generator.answerLength, prediction_generator.gloveEncoding(), config)
    explain_model = VQAModel.explainModel(model)

    #prediction,top, heat1,heat2 = explain_model.predict_generator(prediction_generator,workers=8)
    prediction,top, linear, softmax = explain_model.predict_generator(prediction_generator,workers=8,steps=20)
    acc, results = prediction_generator.evaluate(top)
    print(acc)
    avg = np.average(linear)
    print("avg: "+str(avg))
    r = randint(0,linear.shape[0]-20)
    for k in range(100):
        print(k)
        prediction_generator.print(k+r,prediction[k+r],linear[k+r],softmax[k+r],avg)

if __name__ == '__main__':
    explainConfig(VQAConfig(imageType= 'preprocessed_res_24',
        testName='inceptionResNet_noise_test',
        gloveName='glove.42B.300d',
        gloveSize=300,
        augmentations=8,
        imageFeaturemapSize=24,
        imageFeatureChannels=1536,
        modelIdentifier='Mar-08-2019_0747',
        epoch=14
        )
    )


# model.fit_generator(training_generator, epochs=1, validation_data=validation_generator)