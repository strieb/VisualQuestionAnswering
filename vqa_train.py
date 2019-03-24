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

def trainConfig(config: VQAConfig,recompile=False):
    training_generator = VQAGenerator(True,False, config)
    if config.modelIdentifier:
        model = load_model(DATADIR+'/Results/'+config.testName+'/model_'+config.modelIdentifier+'_'+str(config.epoch)+'.keras')
    else:
        model = VQAModel.createModel(training_generator.questionLength, training_generator.answerLength, training_generator.gloveEncoding(), config)


#    model.get_layer('noise_layer_image').stddev = config.noise
#    model.get_layer('noise_layer_question').stddev = config.noise
    prediction_generator = VQAGenerator(False,True, config)
    eval_model = VQAModel.evalModel(model)

    img = prediction_generator.getImage(0)

    if not os.path.exists(DATADIR+'/Results/'+config.testName):
        os.mkdir(DATADIR+'/Results/'+config.testName)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    result_str =str(timestamp)+"\n\n"+ str(config) +'\n\n'
    with open(DATADIR+'/Results/'+config.testName+'/results-'+timestamp+'.txt', "w") as text_file:
        text_file.write(result_str)

    if recompile:
        model.compile(optimizer=config.optimizer, loss=config.loss)

    best = 0
    for i in range(config.epoch+1,config.stop):
        # if i == 1 and config.loss=='binary_crossentropy':
        #     model.compile(optimizer="adam", loss="categorical_crossentropy")
        # if i == 3 and config.loss=='binary_crossentropy':
        #     model.compile(optimizer=config.optimizer, loss=config.loss)
        model.fit_generator(training_generator, epochs=1,workers=6)
        model.save(DATADIR+'/Results/'+config.testName+'/model_'+timestamp+"_"+str(i)+'.keras')
        print("Test set")
        prediction = eval_model.predict_generator(prediction_generator,workers=6, steps= None if i>6 else 128)
        test_accuracy, results = prediction_generator.evaluate(prediction)
        print("Training set")
        training_generator.predict = True
        prediction = eval_model.predict_generator(training_generator,workers=6, steps=128)
        train_accuracy, _ = training_generator.evaluate(prediction)
        training_generator.predict = False

        result_str += "{0:2d}, {1:6.4f}, {2:6.4f}\n".format(i,test_accuracy,train_accuracy)

        with open(DATADIR+'/Results/'+config.testName+'/results-'+timestamp+'.txt', "w") as text_file:
            text_file.write(result_str)

        with open(DATADIR+'/Results/'+config.testName+'/answers-'+timestamp+"_"+str(i)+'.json', 'w') as fp:
            json.dump(results, fp)

        if test_accuracy > best:
            print("best")
            best = test_accuracy




if __name__ == '__main__':
    

    trainConfig(VQAConfig(
        imageType= 'preprocessed_res_24',
        testName='augmented_tanh',
        gloveName='glove.42B.300d',
        gloveSize=300,
        dropout=True,
        augmentations=None,
        stop=30,
        gatedTanh=True,
        initializer="he_normal",
        batchNorm=False,
        embedding='gru',
        imageFeaturemapSize=24,
        imageFeatureChannels=1536,
        predictNormalizer='sigmoid',
        loss='categorical_crossentropy',
        optimizer='adam',
        modelIdentifier='Mar-16-2019_0617',
        epoch=15,

        )
    )

    
    # trainConfig(VQAConfig(
    #     imageType= 'preprocessed_res_24',
    #     testName='incres_nothing',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=False,
    #     augmentations=None,
    #     stop=30,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     batchNorm=False,
    #     embedding='gru',
    #     imageFeaturemapSize=24,
    #     imageFeatureChannels=1536,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam'
    #     )
    # )

    # trainConfig(VQAConfig(
    #     imageType= 'preprocessed_res_24',
    #     testName='incres_batchnorm',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=True,
    #     augmentations=None,
    #     stop=30,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     normalizeImage=True,
    #     batchNorm=True,
    #     embedding='gru',
    #     imageFeaturemapSize=24,
    #     imageFeatureChannels=1536,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam'
    #     )
    # )

    # trainConfig(VQAConfig(
    #     imageType= 'preprocessed_res_24',
    #     testName='incres_augment',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=True,
    #     augmentations=8,
    #     stop=20,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     batchNorm=True,
    #     embedding='gru',
    #     imageFeaturemapSize=24,
    #     imageFeatureChannels=1536,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam'
    #     )
    # )
    
    # trainConfig(VQAConfig(
    #     imageType= 'preprocessed_res_24',
    #     testName='incres_augment',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=True,
    #     questionDropout=True,
    #     augmentations=8,
    #     stop=30,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     batchNorm=False,
    #     embedding='gru',
    #     imageFeaturemapSize=24,
    #     imageFeatureChannels=1536,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam'
    #     )
    # )



    # trainConfig(VQAConfig(
    #     imageType= 'preprocessed_res_24',
    #     testName='augmented_tanh',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=True,
    #     questionDropout=True,
    #     augmentations=8,
    #     stop=30,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     batchNorm=False,
    #     embedding='gru',
    #     imageFeaturemapSize=24,
    #     imageFeatureChannels=1536,
    #     trainvaltogether= False,
    #     normalizeImage=False,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     scoreMultiplier=0.1,
    #     )
    # )
    
    # trainConfig(VQAConfig(
    #     imageType= 'rcnn',
    #     testName='botup_init_relu',
    #     gloveName='glove.42B.300d',
    #     gloveSize=300,
    #     dropout=True,
    #     augmentations=None,
    #     stop=20,
    #     gatedTanh=False,
    #     initializer="he_normal",
    #     batchNorm=False,
    #     embedding='gru',
    #     imageFeaturemapSize=36,
    #     imageFeatureChannels=2048,
    #     trainvaltogether= True,
    #     normalizeImage=False,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     scoreMultiplier=0.1
    #     )
    # )

    # trainConfig(VQAConfig(
    #     imageType= 'rcnn',
    #     testName='cnn_tests',
    #     gloveName='glove.6B.100d',
    #     gloveSize=100,
    #     dropout=True,
    #     augmentations=None,
    #     stop=25,
    #     gatedTanh=True,
    #     initializer="he_normal",
    #     batchNorm=True,
    #     embedding='cnn',
    #     imageFeaturemapSize=36,
    #     imageFeatureChannels=2048,
    #     trainvaltogether= True,
    #     normalizeImage=True,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     scoreMultiplier=0.1
    #     )
    # )



    # trainConfig(VQAConfig(
    #     imageType= 'rcnn',
    #     testName='cnn_tests',
    #     gloveName='glove.6B.100d',
    #     gloveSize=100,
    #     dropout=True,
    #     augmentations=None,
    #     stop=25,
    #     gatedTanh=False,
    #     initializer="he_normal",
    #     batchNorm=False,
    #     embedding='cnn',
    #     imageFeaturemapSize=36,
    #     imageFeatureChannels=2048,
    #     trainvaltogether= True,
    #     normalizeImage=True,
    #     predictNormalizer='sigmoid',
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     scoreMultiplier=0.1
    #     )
    # )