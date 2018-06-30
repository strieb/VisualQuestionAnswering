from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Concatenate, GRU, Multiply, Reshape, GlobalAveragePooling1D,RepeatVector, Activation, Dot, Lambda, Embedding, AveragePooling2D
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
import numpy as np

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

    
def konstant(batch):
    batch_size = K.shape(batch)[0]
    k_constants = K.variable(np.reshape(np.identity(64),(1,64,64)))
    k_constants = K.tile(k_constants, (batch_size, 1, 1))
    return k_constants


def evalModel(model):
    return Model(inputs=model.input, outputs=[model.output, model.get_layer('activation_1').output])

def createModel(words, answers, glove_encoding):

    question = Input(shape=(14,))

    embedding_layer = Embedding(words+2, 100, input_length=14, trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([glove_encoding])

    question_embedded = embedding_layer(question)


    # question_special = Input(shape=(words,))
    # dense_question_special = Dense(200, activation='relu')(question)
    gru = GRU(512)(question_embedded)
    # concat = Concatenate()([dense_question_special,gru])

    question_repeat = RepeatVector(24)(gru)
    
    image = Input(shape=(24, 2048))

   

    # image_reshape = Reshape(target_shape=(64,2048))(image)

    # fixed_input = Lambda(konstant)(image)
    # image_reshape = Concatenate()([image_reshape,fixed_input])

    concat = Concatenate()([question_repeat,image])
    dense_concat = Dense(512, activation='relu')(concat)
    dense_linear = Dense(1, activation='linear')(dense_concat)
    dense_reshape =  Reshape(target_shape=(24,))(dense_linear)
    softmax = Activation(activation='softmax')(dense_reshape)
    softmax_reshape =  Reshape(target_shape=(1, 24))(softmax)
    image_attention = Lambda(mult)([softmax_reshape,image])
    image_attention_reshape = Reshape(target_shape=(2048,))(image_attention)

    dense_question = Dense(512, activation='relu')(gru)
    # dense_question_mult = Dense(512, activation='relu')(dense_question)
    # dense_question_concat = Dense(256, activation='relu')(dense_question)

    dense_image = Dense(512, activation='relu')(image_attention_reshape)
    # dense_image_mult = Dense(512, activation='relu')(dense_image)
    # dense_image_concat = Dense(256, activation='relu')(dense_image)

    both_mult = Multiply()([dense_question,dense_image])
    # both_concat = Concatenate()([both_mult, dense_question_concat, dense_image_concat])
    dense_both = Dense(2048, activation='relu')(both_mult)
    predictions = Dense(answers, activation='sigmoid')(dense_both)

    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    return model
