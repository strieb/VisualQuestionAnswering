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
# from VQAConfig import VQAConfig

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
    argmax = Lambda(lambda x: K.argmax(x,-1))(model.output)
    return Model(inputs=model.input, outputs=[model.output, argmax, model.get_layer('reshape_2').output,model.get_layer('activation_1').output])

def createGLU(units,config: VQAConfig, input):
    conv_1 = Conv1D(units,5,padding="same",kernel_initializer=config.initializer, activation=None)(input)
    conv_2 = Conv1D(units,5,padding="same",kernel_initializer=config.initializer, activation="sigmoid")(input)
    mult = Multiply()([conv_1,conv_2])  
    if config.batchNorm:
        return  BatchNormalization()(mult)
    else:
        return mult

def createMask(input,mask):
    return Lambda(lambda x: x[0] * x[1])([mask,input])

def createGatedTanh(units, input):
    tanh_layer = Dense(units,activation='tanh',kernel_initializer='he_normal')(input)
    sigmoid_layer = Dense(units,activation='sigmoid',kernel_initializer='he_normal')(input)
    return Multiply()([tanh_layer,sigmoid_layer])

def createGatedTanhBatchNorm(units, input):
    tanh_layer = Dense(units,activation='tanh')(input)
    sigmoid_layer = Dense(units,activation='sigmoid')(input)
    mult =  Multiply()([tanh_layer,sigmoid_layer])
    norm = BatchNormalization()(mult)
    return norm

def createDenseLayer(units, config: VQAConfig, input):
    if config.gatedTanh:
        tanh_layer = Dense(units,activation='tanh', kernel_initializer=config.initializer)(input)
        sigmoid_layer = Dense(units,activation='sigmoid',kernel_initializer=config.initializer)(input)
        mult =  Multiply()([tanh_layer,sigmoid_layer])
        if config.batchNorm:
            return  BatchNormalization()(mult)
        else:
            return mult
    else:
        return Dense(units, activation='relu', kernel_initializer=config.initializer)(input) 
    
def createModel(words, answers, glove_encoding,config: VQAConfig):
    question = Input(shape=(14,))
    question_mask = Reshape(target_shape=(14,1))(Lambda(lambda x: K.cast(x>0, K.floatx()))(question))
    #question_mask_2 = RepeatVector(512)(question_mask_1)
    #question_mask = Permute((2,1))(question_mask_2)
    embedding_layer = Embedding(words+4, glove_encoding.shape[1], input_length=14, trainable=True,mask_zero=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([glove_encoding])

    question_embedded = embedding_layer(question) # shape = (encodingSize,14)
    question_embedded_noise = GaussianNoise(config.noise, name='noise_layer')(question_embedded)

    question_embedded_masked = createMask(question_embedded,question_mask)
    glu_1 = createGLU(512,config, question_embedded_masked)
    glu_2 = createGLU(512,config, glu_1)
    glu_3 = createGLU(512,config, glu_2)
    glu_add = Add()([glu_1,glu_3])
    glu_gated = createGatedTanhBatchNorm(512,glu_add)
    glu_avgpool = GlobalAveragePooling1D()(glu_gated)

    # question_trained = Dense(100, activation='linear')(question_embedded)

    
    question_dense = Dense(512, activation='relu')(question_embedded)
    question_dense_mask = Lambda(lambda x: x[0] * x[1])([question_mask,question_dense])
    #question_dense_mask = Multiply()([question_mask,question_dense])
    question_maxpool = GlobalAveragePooling1D()(question_dense_mask)

    gru = GRU(512)(question_embedded_noise)

    if config.embedding == 'gru':
        question_layer = gru
    elif config.embedding == 'cnn':
        question_layer = glu_avgpool
    else:
        question_layer = question_maxpool
    
    image = Input(shape=(config.imageFeaturemapSize, config.imageFeatureChannels))
    image_noise = GaussianNoise(config.noise, name='noise_layer')(image)
    image_norm = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(image)

    image_attention = createAttentionLayers(image_norm,question_layer, config)

    fusion = createFusionLayers(image_attention, question_layer, config)
    predictions = Dense(answers, activation='sigmoid',kernel_initializer='he_normal' )(fusion)
    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    return model


def createAttentionLayers(image_features, question_features, config: VQAConfig):
    question_repeat = RepeatVector(config.imageFeaturemapSize)(question_features)
    concat = Concatenate()([question_repeat,image_features])
    dense_concat = createDenseLayer(512,config,concat)
    dense_linear = Dense(1, activation='linear')(dense_concat)
    dense_reshape =  Reshape(target_shape=(config.imageFeaturemapSize,))(dense_linear)
    # divide = Lambda(lambda x: x/5.0)(dense_reshape)
    softmax = Activation(activation='softmax')(dense_reshape)
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
        dense_both_dropout = Dropout(rate=0.5,name='dropout_layer')(dense_both)
        return dense_both_dropout
    else:
        return dense_both