from keras.layers import Input, Embedding, GRU, RepeatVector
from keras.layers import Reshape, Concatenate, Activation
from keras.layers import Lambda, Multiply, Dense, Dropout
import keras.backend as Backend
from keras.models import Model

TOKEN_LENGTH = 3000
IMAGE_REGIONS = 24
IMAGE_CHANNELS = 2048
ANSWERS = 2000

# Inputs
image = Input(shape=(IMAGE_REGIONS, IMAGE_CHANNELS))
question = Input(shape=(14,))

# Question Embedding
question_embedded = Embedding(input_dim=TOKEN_LENGTH, output_dim=100,
    input_length=14, trainable=True, mask_zero=True)(question)
question_layer = GRU(512)(question_embedded)

# Attention
question_repeat = RepeatVector(IMAGE_REGIONS)(question_layer)
attention_concat = Concatenate()([question_repeat, image])
attention_dense = Dense(512, activation='relu')(attention_concat)
attention_linear = Dense(1, activation='linear')(attention_dense)
attention_linear_reshape = Reshape(
    target_shape=(IMAGE_REGIONS,))(attention_linear)
attention_softmax = Activation(
    activation='softmax')(attention_linear_reshape)
attention_weights = Reshape(target_shape=(
    1, IMAGE_REGIONS))(attention_softmax)
image_attention = Lambda(lambda x:  Backend.batch_dot(
    x[0], x[1]))([attention_weights, image])
image_attention_reshape = Reshape(
    target_shape=(IMAGE_CHANNELS,))(image_attention)

# Multimodal Fusion
question_dense = Dense(512, activation='relu')(question_layer)
image_dense = Dense(512, activation='relu')(
    image_attention_reshape)
fusion_layer = Multiply()([question_dense, image_dense])
fusion_dense = Dense(1024, activation='relu')(fusion_layer)
fustion_dropout = Dropout(rate=0.5)(fusion_dense)

# Classifier
predictions = Dense(
    ANSWERS, activation='softmax')(fustion_dropout)

model = Model(inputs=[question, image], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy')
