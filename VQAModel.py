from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnext import ResNeXt101
from keras.layers import BatchNormalization, Permute, Masking, Add, Conv1D, Input, Dense, Concatenate, GRU,LSTM, Multiply, Reshape,GlobalAveragePooling1D, GlobalMaxPool1D,RepeatVector, Activation, Dot, Lambda, Embedding, AveragePooling2D, Dropout,GaussianDropout
from keras import regularizers
from keras.models import Model

import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def createResNetFull(size):
    return Xception(weights='imagenet',input_shape=size, include_top=True)

    
def createModelResNet(size):
    inception = Xception(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(2,2))(inception.output)
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
    argmax = Lambda(lambda x: K.argmax(x,-1))(model.output)
    return Model(inputs=model.input, outputs=[argmax])

def explainModel(model):
    return Model(inputs=model.input, outputs=[model.output, model.get_layer('activation_1').output])

def createGLU(units, input):
    conv_1 = Conv1D(units,5,padding="same", activation=None)(input)
    conv_2 = Conv1D(units,5,padding="same", activation="sigmoid")(input)
    mult = Multiply()([conv_1,conv_2])
    norm = BatchNormalization()(mult)
    return norm

def createMask(input,mask):
    return Lambda(lambda x: x[0] * x[1])([mask,input])

def createGatedTanh(units, input):
    tanh_layer = Dense(units,activation='tanh')(input)
    sigmoid_layer = Dense(units,activation='sigmoid')(input)
    return Multiply()([tanh_layer,sigmoid_layer])

def createGatedTanhBatchNorm(units, input):
    tanh_layer = Dense(units,activation='tanh')(input)
    sigmoid_layer = Dense(units,activation='sigmoid')(input)
    mult =  Multiply()([tanh_layer,sigmoid_layer])
    norm = BatchNormalization()(mult)
    return norm
    
def createModel(words, answers, glove_encoding, feature_state_size, feature_size):

    question = Input(shape=(14,))
    question_mask = Reshape(target_shape=(14,1))(Lambda(lambda x: K.cast(x>0, K.floatx()))(question))
    #question_mask_2 = RepeatVector(512)(question_mask_1)
    #question_mask = Permute((2,1))(question_mask_2)
    embedding_layer = Embedding(words+4, glove_encoding.shape[1], input_length=14, trainable=True,mask_zero=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([glove_encoding])

    question_embedded = embedding_layer(question) # shape = (encodingSize,14)
    question_embedded_masked = createMask(question_embedded,question_mask)
    glu_1 = createGLU(512, question_embedded_masked)
    glu_2 = createGLU(512, glu_1)
    glu_3 = createGLU(512, glu_2)
    glu_add = Add()([glu_1,glu_3])
    glu_gated = createGatedTanhBatchNorm(512,glu_add)
    glu_maxpool = GlobalAveragePooling1D()(glu_gated)

    # question_trained = Dense(100, activation='linear')(question_embedded)

    
    question_dense = Dense(512, activation='relu')(question_embedded)
    question_dense_mask = Lambda(lambda x: x[0] * x[1])([question_mask,question_dense])
    #question_dense_mask = Multiply()([question_mask,question_dense])
    question_maxpool = GlobalAveragePooling1D()(question_dense_mask)

    gru = GRU(512)(question_embedded)
    question_layer = gru
    
    image = Input(shape=(feature_size, feature_state_size))

    image_attention = createAttentionLayers(image,question_layer,feature_state_size, feature_size)

    fusion = createFusionLayers(image_attention, question_layer, False)
    predictions = Dense(answers, activation='sigmoid')(fusion)

    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy')
    model.summary()
    return model


def createAttentionLayers(image_features, question_features, feature_state_size, feature_size):
    question_repeat = RepeatVector(feature_size)(question_features)
    concat = Concatenate()([question_repeat,image_features])
    dense_concat = Dense(512, activation='relu')(concat)
    dense_linear = Dense(1, activation='linear')(dense_concat)
    dense_reshape =  Reshape(target_shape=(feature_size,))(dense_linear)
    # divide = Lambda(lambda x: x/5.0)(dense_reshape)
    softmax = Activation(activation='softmax')(dense_reshape)
    softmax_reshape =  Reshape(target_shape=(1, feature_size))(softmax)
    image_attention = Lambda(mult)([softmax_reshape,image_features])
    image_attention_reshape = Reshape(target_shape=(feature_state_size,))(image_attention)
    return image_attention_reshape


def createFusionLayers(image_features, question_features, gatedTanh = False):
    if gatedTanh:
        dense_question = createGatedTanhBatchNorm(512, question_features)
        dense_image = createGatedTanhBatchNorm(512, image_features)
        both_mult = Multiply()([dense_question,dense_image])
        dense_both = createGatedTanhBatchNorm(1024, both_mult)
        dense_both_dropout = Dropout(rate=0.5)(dense_both)
        return dense_both_dropout

    else:
        # ,kernel_regularizer=regularizers.l2(0.01)
        dense_question = Dense(512, activation='relu')(question_features)
        dense_image = Dense(512, activation='relu')(image_features)
        both_mult = Multiply()([dense_question,dense_image])
        dense_both = Dense(1024, activation='relu')(both_mult)
        # dense_both_dropout = Dropout(rate=0.5)(dense_both)
        return dense_both