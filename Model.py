import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPool2D, concatenate, Input, Dropout, BatchNormalization, ReLU

def conv_block(filter, input):
    conv = Conv2D(filter, kernel_size=(3, 3), padding='same')(input)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    conv = Conv2D(filter, (3, 3), padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    return conv

def UNET(input_shape, dropout, filter):

    input_layer = Input(input_shape)

    conv1 = conv_block(filter, input_layer)
    encoder = MaxPool2D((2, 2), strides=2)(conv1)
    encoder = Dropout(dropout)(encoder)

    conv2 = conv_block(filter * 2, encoder)
    encoder = MaxPool2D((2, 2), strides=2)(conv2)
    encoder = Dropout(dropout)(encoder)

    conv3 = conv_block(filter * 4, encoder)
    encoder = MaxPool2D((2, 2), strides=2)(conv3)
    encoder = Dropout(dropout)(encoder)

    conv4 = conv_block(filter * 8, encoder)
    encoder = MaxPool2D((2, 2), strides=2)(conv4)
    encoder = Dropout(dropout)(encoder)

    conv5 = conv_block(filter * 16, encoder)
    encoder = MaxPool2D((2, 2), strides=2)(conv5)
    encoder = Dropout(dropout)(encoder)

    conv6 = conv_block(filter * 32, encoder)

    # now the decoder
    decoder = Conv2DTranspose(filter * 16, (3, 3), strides=(2, 2), padding='same')(conv6)
    decoder = concatenate([decoder, conv5])
    decoder = Dropout(dropout)(decoder)
    conv6 = conv_block(filter * 16, decoder)

    decoder = Conv2DTranspose(filter * 8, (3, 3), strides=(2, 2), padding='same')(conv6)
    decoder = concatenate([decoder, conv4])
    decoder = Dropout(dropout)(decoder)
    conv7 = conv_block(filter * 8, decoder)

    decoder = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv7)
    decoder = concatenate([decoder, conv3])
    decoder = Dropout(dropout)(decoder)
    conv8 = conv_block(filter * 4, decoder)

    decoder = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv8)
    decoder = concatenate([decoder, conv2])
    decoder = Dropout(dropout)(decoder)
    conv9 = conv_block(filter * 2, decoder)

    decoder = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv9)
    decoder = concatenate([decoder, conv1])
    decoder = Dropout(dropout)(decoder)
    conv10 = conv_block(filter, decoder)

    # sigmoid, since it's either tumor or no tumor
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    model = Model(inputs=[input_layer], outputs=[output])

    return model
