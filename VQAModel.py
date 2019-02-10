from keras.applications.inception_v3 import InceptionV3
from keras.layers import Permute, Masking, Add, Conv1D, Input, Dense, Concatenate, GRU,LSTM, Multiply, Reshape,GlobalAveragePooling1D, GlobalMaxPool1D,RepeatVector, Activation, Dot, Lambda, Embedding, AveragePooling2D, Dropout,GaussianDropout
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


def createModelInception(size):
    inception = InceptionV3(weights='imagenet', include_top=False, pooling=None,input_shape=size )
    for layer in inception.layers:
        layer.trainable = False
    pool = AveragePooling2D(pool_size=(3,3))(inception.output)
    return Model(inputs=inception.input,outputs=pool)


def createModelInceptionFull():
    inception = InceptionV3(weights='imagenet')
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
    return Model(inputs=model.input, outputs=[argmax, argmax])
    #return Model(inputs=model.input, outputs=[argmax, model.get_layer('activation_1').output])

def createGLU(units, input):
    conv_1 = Conv1D(units,5,padding="same", activation=None)(input)
    conv_2 = Conv1D(units,5,padding="same", activation="sigmoid")(input)
    return Multiply()([conv_1,conv_2])


def createGatedTanh(units, input):
    tanh_layer = Dense(units,activation='tanh')(input)
    sigmoid_layer = Dense(units,activation='sigmoid')(input)
    return Multiply()([tanh_layer,sigmoid_layer])
    
def createModel(words, answers, glove_encoding):
    feature_size = 1

    question = Input(shape=(14,))
    question_mask = Reshape(target_shape=(14,1))(Lambda(lambda x: K.cast(x>0, K.floatx()))(question))
    #question_mask_2 = RepeatVector(512)(question_mask_1)
    #question_mask = Permute((2,1))(question_mask_2)
    embedding_layer = Embedding(words+4, glove_encoding.shape[1], input_length=14, trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([glove_encoding])

    question_embedded = embedding_layer(question) # shape = (encodingSize,14)
    glu_1 = createGLU(512, question_embedded)
    glu_2 = createGLU(512, glu_1)
    glu_3 = createGLU(512, glu_2)
    glu_add = Add()([glu_1,glu_3])
    glu_gated = createGatedTanh(512,glu_add)
    glu_maxpool = GlobalAveragePooling1D()(glu_gated)

    # question_trained = Dense(100, activation='linear')(question_embedded)

    
    question_dense = Dense(512, activation='relu')(question_embedded)
    question_dense_mask = Lambda(lambda x: x[0] * x[1])([question_mask,question_dense])
    #question_dense_mask = Multiply()([question_mask,question_dense])
    question_maxpool = GlobalAveragePooling1D()(question_dense_mask)

    # gru = GRU(512)(question_embedded)
    question_layer = question_maxpool

    question_repeat = RepeatVector(feature_size)(question_layer)
    
    image = Input(shape=(feature_size, 2048))

    # image_reshape = Reshape(target_shape=(64,2048))(image)

    # fixed_input = Lambda(constant)(image)
    # image_reshape = Concatenate()([image_reshape,fixed_input])

    concat = Concatenate()([question_repeat,image])
    dense_concat = Dense(512, activation='relu')(concat)
    dense_linear = Dense(1, activation='linear')(dense_concat)
    dense_reshape =  Reshape(target_shape=(feature_size,))(dense_linear)
    softmax = Activation(activation='softmax')(dense_reshape)
    # softmax_dropout = Dropout(rate=0.4)(softmax)
    softmax_reshape =  Reshape(target_shape=(1, feature_size))(softmax)
    image_attention = Lambda(mult)([softmax_reshape,image])
    image_attention_reshape = Reshape(target_shape=(2048,))(image_attention)
    
    image_reshape = Reshape(target_shape=(2048,))(image)

    dense_question = Dense(512, activation='relu')(question_layer)
    #dense_question = createGatedTanh(512, question_layer)

    dense_image = Dense(512, activation='relu')(image_reshape)
    #dense_image = createGatedTanh(512, image_reshape)

    both_mult = Multiply()([dense_question,dense_image])

    dense_both = Dense(1024, activation='relu')(both_mult)
    #dense_both = createGatedTanh(1024, both_mult)
    dense_both_dropout = Dropout(rate=0.5)(dense_both)
    predictions = Dense(answers, activation='sigmoid')(dense_both_dropout)

    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    return model
