from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Concatenate, GRU, Multiply
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

    question = Input(shape=(14,100))
    # question_special = Input(shape=(words,))
    # dense_question_special = Dense(512, activation='relu')(question_special)
    gru = GRU(512+256)(question)
    # concat = Concatenate()([dense_question_special,gru])
    dense_question = Dense(512+256, activation='relu')(gru)

    image = Input(shape=(2048,))
    dense_image = Dense(512+256, activation='relu')(image)

    both = Multiply()([dense_image, dense_question])
    dense_both = Dense(2048, activation='relu')(both)
    predictions = Dense(answers, activation='softmax')(dense_both)

    model = Model(inputs=[question, image], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    #model.summary()

    return model
