from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Concatenate
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def createModelInception():
    inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    for layer in inception.layers:
        layer.trainable = False
    return inception


def createModelInceptionFull():
    inception = InceptionV3(weights='imagenet')
    return inception


def createModel(words, answers):

    question = Input(shape=(words,))
    dense_question = Dense(512, activation='tanh')(question)

    image = Input(shape=(2048,))
    dense_image = Dense(512, activation='tanh')(image)

    both = Concatenate()([dense_image, dense_question])
    dense_both = Dense(1024, activation='tanh')(both)
    dense_both = Dense(1024, activation='tanh')(dense_both)
    predictions = Dense(answers, activation='softmax')(dense_both)

    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model
