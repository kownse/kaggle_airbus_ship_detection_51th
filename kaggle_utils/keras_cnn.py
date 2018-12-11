from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input,ZeroPadding2D,Conv2D,MaxPooling2D, \
    BatchNormalization, Flatten, Dropout, Dense, Activation, Add, \
    SpatialDropout2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.merge import concatenate
from keras import backend as K

def dense_set(inp_layer, n, activation, drop_rate=0.):
    if drop_rate > 0.:
        inp_layer = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(inp_layer)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation = activation)(bn)
    return act

def conv_layer(inp_layer, feature_size, kernel_size=(3, 3),strides=(1,1), 
               padding='same', activited = True):
    conv = Conv2D(filters=feature_size, kernel_size=kernel_size, strides=strides, 
                  padding=padding, kernel_initializer='he_normal')(inp_layer)
    bn = BatchNormalization(axis=-1)(conv)

    if activited:
        return ELU()(bn)#LeakyReLU(1/10)(bn)
    else:
        return bn

def bottonneck_conv_layer(inp_layer, feature_size, kernel_size=(3,3), strides=(1,1),
                          padding='same', activited = True, botton_size=16):
    bottonneck = Conv2D(filters=botton_size, kernel_size=(1,1), strides=(1,1), 
                        padding='same', kernel_initializer='he_normal')(inp_layer)
    
    conv = Conv2D(filters=feature_size, kernel_size=kernel_size, strides=strides, 
                  padding=padding, kernel_initializer='he_normal')(bottonneck)
    bn = BatchNormalization(axis=-1)(conv)

    if activited:
        return ELU()(bn)# LeakyReLU(1/10)(bn)
    else:
        return bn


def residual_block(inp_layer, feature_size, kernal_size=(3, 3), strides=(1,1), 
                   dropout = 0., bottonneck = False):
    shortcut = inp_layer

    y = inp_layer
    if bottonneck:
        y = bottonneck_conv_layer(y, feature_size, kernal_size, strides, 
                       padding='same', activited = True)
        
        if dropout > 0.:
            y = SpatialDropout2D(dropout)(y)
        
        y = bottonneck_conv_layer(y, feature_size, kernal_size, strides, 
                       padding='same', activited = False)

        shortcut = bottonneck_conv_layer(shortcut, feature_size, kernel_size=(1,1), strides=strides,
                            padding='same', activited=False)
    else:
        y = conv_layer(y, feature_size, kernal_size, strides, 
                       padding='same', activited = True)
        
        if dropout > 0.:
            y = SpatialDropout2D(dropout)(y)
            
        y = conv_layer(y, feature_size, kernal_size, strides, 
                       padding='same', activited = False)

        shortcut = conv_layer(shortcut, feature_size, kernel_size=(1,1), strides=strides,
                            padding='same', activited=False)

    y = Add()([y,shortcut])
    y = BatchNormalization(axis=-1)(y)
    y = ELU()(y)#LeakyReLU(1/10)(y)
    
    return y

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def UNET_256(inputsize = 256, dropout_val=0.2, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, inputsize, inputsize))
        axis = 1
    else:
        inputs = Input((inputsize, inputsize, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, 0, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, 0, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model