"""
The VGG network architecture was introduced by Simonyan and Zisserman
in their 2014 paper.

The paper can be accessed in https://arxiv.org/abs/1409.1556.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from warnings import filterwarnings
filterwarnings('ignore')

from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras import backend as K

from .BatchNormalization import BatchNormalization

def get_last_conv_layer_name():
    return 'block5_conv4'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)

def conv_block(units, conv_name, bn_name, activation='relu',
               padding='same', bn=False, axis=3):
    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding=padding, name=conv_name)(inp)
        if bn:
            x = BatchNormalization(axis=axis, name=bn_name)(x)
        x = Activation(activation)(x)
        return x
    
    return layer_wrapper

def classifier(base_layers, nb_classes=2, trainable=False):
    input_shape = (None, 7, 7, 512)

    out = Flatten(name='flatten')(base_layers)
    out = Dense(4096, activation='relu', name='fc1')(out)
    out = Dropout(0.5)(out)
    out = Dense(4096, activation='relu', name='fc2')(out)
    out = Dropout(0.5)(out)

    out_class = Dense(nb_classes, activation='softmax', kernel_initializer='zero',
                                    name='dense_class_{}'.format(nb_classes))(out)

    return out_class

def nn_base(input_tensor=None, trainable=False, batch_normalization=False):

    # determine input shape. As the code will only support
    # tensorflow backend, there is only one option for input
    # shape
    input_shape = (None, None, 3) # 3 because the model will receive RGB images

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # because the backend is tensorflow,
    # K.image_dim_ordering() == 'tf
    bn_axis = 3

    # Block 1
    x = conv_block(64, 'block1_conv1', 'block1_bn1', bn=batch_normalization, axis=bn_axis)(img_input)
    x = conv_block(64, 'block1_conv2', 'block1_bn2', bn=batch_normalization, axis=bn_axis)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_block(128, 'block2_conv1', 'block2_bn1', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(128, 'block2_conv2', 'block2_bn2', bn=batch_normalization, axis=bn_axis)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_block(256, 'block3_conv1', 'block3_bn1', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(256, 'block3_conv2', 'block3_bn2', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(256, 'block3_conv3', 'block3_bn3', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(256, 'block3_conv4', 'block3_bn4', bn=batch_normalization, axis=bn_axis)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_block(512, 'block4_conv1', 'block4_bn1', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block4_conv2', 'block4_bn2', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block4_conv3', 'block4_bn3', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block4_conv4', 'block4_bn4', bn=batch_normalization, axis=bn_axis)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_block(512, 'block5_conv1', 'block5_bn1', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block5_conv2', 'block5_bn2', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block5_conv3', 'block5_bn3', bn=batch_normalization, axis=bn_axis)(x)
    x = conv_block(512, 'block5_conv4', 'block5_bn4', bn=batch_normalization, axis=bn_axis)(x)

    return x

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
activation (Activation)      (None, 224, 224, 64)      0         
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
activation_1 (Activation)    (None, 224, 224, 64)      0         
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
activation_2 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
activation_3 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
activation_4 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
activation_5 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
activation_6 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
activation_7 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
activation_8 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
activation_9 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
activation_10 (Activation)   (None, 28, 28, 512)       0         
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
activation_11 (Activation)   (None, 28, 28, 512)       0         
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
activation_12 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
activation_13 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
activation_14 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
activation_15 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
flatten (Flatten)            (None, 100352)            0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              411045888 
_________________________________________________________________
dropout (Dropout)            (None, 4096)              0         
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_class_2 (Dense)        (None, 2)                 8194      
=================================================================
Total params: 447,859,778
Trainable params: 447,859,778
Non-trainable params: 0
_________________________________________________________________
"""