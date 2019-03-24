from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnext import ResNeXt101
from keras.layers import BatchNormalization, Permute, Masking, Add, Conv1D, Input, Dense, Concatenate, GRU,LSTM, Multiply, Reshape,GlobalAveragePooling1D, GlobalMaxPool1D,RepeatVector, Activation, Dot, Lambda, Embedding, AveragePooling2D, Dropout,GaussianDropout,GaussianNoise
from keras import regularizers
from keras.models import Model
from VQAConfig import VQAConfig

import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config2.gpu_options.allow_growth = True
sess = tf.Session(config=config2)
set_session(sess)

def createResNetFull(size):
    return ResNeXt101(weights='imagenet',input_shape=size, include_top=True)

    
def createModelResNet(size):
    inception = ResNeXt101(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(inception.output)
    return Model(inputs=inception.input,outputs=pool)


def createModelXception(size):
    inception = Xception(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(inception.output)
    return Model(inputs=inception.input,outputs=pool)

def createXceptionFull(size):
    return Xception(weights='imagenet',input_shape=size)

def createModelInceptionResNet(size):
    inception = InceptionResNetV2(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(inception.output)
    return Model(inputs=inception.input,outputs=pool)

def createInceptionResNetFull(size):
    return InceptionResNetV2(weights='imagenet',input_shape=size)


def createModelInception(size):
    inception = InceptionV3(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(inception.output)
    return Model(inputs=inception.input,outputs=pool)

    
def createModelNasNet(size):
    nas = NASNetLarge(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in nas.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(nas.output)
    return Model(inputs=nas.input,outputs=pool)

def createNasNetFull(size):
    return NASNetLarge(weights='imagenet',input_shape=size)

def createModelInceptionFull(size):
    inception = InceptionV3(weights='imagenet',input_shape=size)
    return inception

def mult(ip):
    x = ip[0]
    y = ip[1]
    return K.batch_dot(x,y)

    
def constant(batch):
    batch_size = K.shape(batch)[0]
    k_constants = K.variable(np.reshape(np.identity(64),(1,64,64)))
    k_constants = K.tile(k_constants, (batch_size, 1, 1))
    return k_constants


def evalModel(model):
    argmax = Lambda(lambda x: K.argmax(x,-1), name='eval_lambda')(model.output)
    return Model(inputs=model.input, outputs=[argmax])

def explainModel(model):
    argmax = Lambda(lambda x: K.argmax(x,-1), name='eval_lambda')(model.output)
    #return Model(inputs=model.input, outputs=[model.output, argmax, model.get_layer('reshape_10').output,model.get_layer('activation_3').output])
    return Model(inputs=model.input, outputs=[model.output, argmax, model.get_layer('linear_attention').output,model.get_layer('softmax_attention').output])

def createGLU(units,config: VQAConfig, input):
    conv_1 = Conv1D(units,5,padding="same",kernel_initializer=config.initializer, activation=None)(input)
    conv_2 = Conv1D(units,5,padding="same",kernel_initializer=config.initializer, activation="sigmoid")(input)
    mult = Multiply()([conv_1,conv_2])  
    if config.batchNorm:
        return  BatchNormalization()(mult)
    else:
        return mult

def createGatedBlock(units,config: VQAConfig, input):
    glu_1 = createGLU(units, config, input)
    glu_2 = createGLU(units, config, glu_1)
    return Add()([input,glu_2])

def createMask(input,mask):
    return Lambda(lambda x: x[0] * x[1])([mask,input])

def createDenseLayer(units, config: VQAConfig, input, regularize = True):
    if config.gatedTanh:
        tanh_layer = Dense(units,activation='tanh', kernel_initializer=config.initializer)(input)
        sigmoid_layer = Dense(units,activation='sigmoid',kernel_initializer=config.initializer)(input)
        mult =  Multiply()([tanh_layer,sigmoid_layer])
        if config.batchNorm:
            return  BatchNormalization()(mult)
        else:
            return mult
    else:
        regularizer = regularizers.l2(config.regularization) if config.regularization and regularize else None
        dense = Dense(units, activation='relu', kernel_initializer=config.initializer, kernel_regularizer=regularizer)(input) 
        if config.batchNorm:
            return  BatchNormalization()(dense)
        else:
            return dense
        
    
def createModel(words, answers, glove_encoding,config: VQAConfig):
    K.clear_session()
    question = Input(shape=(14,))
    question_mask = Reshape(target_shape=(14,1))(Lambda(lambda x: K.cast(x>0, K.floatx()))(question))

    embedding_layer = Embedding(words+4, glove_encoding.shape[1], input_length=14, trainable=True,mask_zero=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([glove_encoding])

    question_embedded = embedding_layer(question) # shape = (encodingSize,14)
    if config.noise > 0:
        question_embedded = GaussianNoise(config.noise, name='noise_layer_question')(question_embedded)
    if config.questionDropout:
        question_embedded = Dropout(config.dropoutRate)(question_embedded)

    question_embedded_masked = createMask(question_embedded,question_mask)

    glu_1 = createGLU(512,config, question_embedded_masked)
    gated_1 = createGatedBlock(512,config, glu_1)
    gated_2 = createGatedBlock(512,config, gated_1)
    glu_avgpool = GlobalAveragePooling1D()(gated_2)

    
    question_avgpool = GlobalAveragePooling1D()(question_embedded)
    question_dense = Dense(512, activation=None)(question_avgpool)

    gru = GRU(512)(question_embedded)

    if config.embedding == 'gru':
        question_layer = gru
    elif config.embedding == 'cnn':
        question_layer = glu_avgpool
    else:
        question_layer = question_dense
    
    image = Input(shape=(config.imageFeaturemapSize, config.imageFeatureChannels))
    if config.normalizeImage:
        image_norm = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(image)
    else:
        image_norm = image
    if config.imageDropout:
        image_dropout = Dropout(config.dropoutRate)(image_norm)
    else:
        image_dropout = image_norm

    image_attention = createAttentionLayers(image_dropout,question_layer, config)

    fusion = createFusionLayers(image_attention, question_layer, config)
    
    regularizer = regularizers.l2(config.regularization) if config.regularization else None
    predictions = Dense(answers, activation=config.predictNormalizer,kernel_initializer=config.initializer, kernel_regularizer=regularizer )(fusion)
    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer=config.optimizer, loss=config.loss)
    model.summary()
    return model


def createAttentionLayers(image_features, question_features, config: VQAConfig):
    question_repeat = RepeatVector(config.imageFeaturemapSize)(question_features)
    concat = Concatenate()([question_repeat,image_features])
    dense_concat = createDenseLayer(512,config,concat, False)
    if config.attentionDropout:
        dense_concat = Dropout(config.dropoutRate)(dense_concat)
    dense_linear = Dense(1, activation='linear')(dense_concat)
    dense_reshape =  Reshape(target_shape=(config.imageFeaturemapSize,),name="linear_attention")(dense_linear)
    softmax = Activation(activation='softmax',name="softmax_attention")(dense_reshape)
    softmax_reshape =  Reshape(target_shape=(1, config.imageFeaturemapSize))(softmax)
    image_attention = Lambda(mult)([softmax_reshape,image_features])
    image_attention_reshape = Reshape(target_shape=(config.imageFeatureChannels,))(image_attention)
    return image_attention_reshape


def createFusionLayers(image_features, question_features, config: VQAConfig):
    dense_question = createDenseLayer(512,config, question_features)
    dense_image = createDenseLayer(512,config, image_features)
    both_mult = Multiply()([dense_question,dense_image])
    dense_both = createDenseLayer(1024,config, both_mult)

    if config.dropout:
        dense_both_dropout = Dropout(rate=config.dropoutRate,name='dropout_layer')(dense_both)
        return dense_both_dropout
    else:
        return dense_both