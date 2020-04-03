"""
The EfficientNet-B0 network architecture was introduced by Tan and Le
in their 2019 paper.

The paper can be accessed in https://arxiv.org/abs/1905.11946.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from warnings import filterwarnings
import string
import json

filterwarnings('ignore')

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras.layers import Dropout, Activation, DepthwiseConv2D, Reshape
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .BatchNormalization import BatchNormalization

from ..utils import DENSE_KERNEL_INITIALIZER, \
                    CONV_KERNEL_INITIALIZER, \
                    round_filters, \
                    get_swish

from ..config import Struct

def get_last_conv_layer_name():
    return 'block5_conv4'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)

def mb_conv_block(block, prefix='', activation='relu',
               padding='same', axis=3, drop_rate=None):

    filters = block.input_filters * block.expand_ratio
    target_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)
    num_reduced_filters = max(1, int(
            block.input_filters * block.se_ratio
        ))

    def expansion_phase(_inp):

        if block.expand_ratio != 1:
            _x = Conv2D(
                filters, 1,
                padding=padding,
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=prefix + 'expand_conv')(_inp)
            _x = BatchNormalization(axis=axis, name=prefix + 'expand_bn')(_x)
            _x = Activation(activation, name=prefix + 'expand_activation')(_x)
        else:
            _x = _inp
        
        return _x
    
    def depthwise_convolution(_inp):

        _x = DepthwiseConv2D(
            block.kernel_size,
            strides=block.strides,
            padding=padding,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + 'dw_conv')(_inp)
        _x = BatchNormalization(axis=axis, name=prefix + 'bn')(_x)
        _x = Activation(activation, name=prefix + 'activation')(_x)

        return _x

    def squeeze_and_excitation(_inp):
        se_tensor = GlobalAveragePooling2D(name=prefix + 'se_squeeze')(_inp)
        se_tensor = Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = Conv2D(
            num_reduced_filters, 1,
            activation=activation,
            padding=padding,
            use_bias=True,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + 'se_reduce')(se_tensor)
        se_tensor = Conv2D(filters, 1,
            activation='sigmoid',
            padding=padding,
            use_bias=True,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + 'se_expand')(se_tensor)
        _x = layers.multiply([_inp, se_tensor], name=prefix + 'se_excite')

        return _x

    def output_phase(_x, _input):
        _x = Conv2D(
            block.output_filters, 1,
            padding=padding,
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + 'project_conv')(_x)
        _x = BatchNormalization(axis=axis, name=prefix + 'project_bn')(x)

        if block.id_skip and all(s == 1 for s in block.strides) and block.input_filters == block.output_filters:
            if drop_rate and drop_rate > 0:
                _x = Dropout(
                    drop_rate,
                    noise_shape=(None, 1, 1, 1),
                    name=prefix + 'drop')(_x)
            _x = layers.add([_x, _input])
        
        return _x

    def layer_wrapper(inp):

        # expansion phase
        x = expansion_phase(inp)

        # depthwise convolution
        x = depthwise_convolution(x)

        # squeeze and excitation phase
        x = squeeze_and_excitation(x, inp)

        return x

    return layer_wrapper

def classifier(base_layers, nb_classes=2, trainable=False):

    out = Flatten(name='flatten')(base_layers)
    out = Dense(4096, activation='relu', name='fc1')(out)
    out = Dropout(0.5)(out)
    out = Dense(4096, activation='relu', name='fc2')(out)
    out = Dropout(0.5)(out)
    out = Dense(8192, activation='relu', name='fc3')(out)
    out = Dropout(0.5)(out)
    out = Dense(8192, activation='relu', name='fc4')(out)
    out = Dropout(0.5)(out)
    out = Dense(8192, activation='relu', name='fc5')(out)
    out = Dropout(0.5)(out)
    out = Dense(16384, activation='relu', name='fc6')(out)
    out = Dropout(0.3)(out)

    out_class = Dense(nb_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER,
                      name='dense_class_{}'.format(nb_classes))(out)

    return out_class

def nn_base(width_coefficient, depth_divisor, input_tensor=None, trainable=False, dropout_rate=0.2, drop_connect_rate=0.2):

    # open json contaning information about blocks
    with open('./block_args.json', 'r') as f:
        blocks_json = json.load(f)

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

    # get swish activation function
    activation = get_swish()

    # build stem
    x = Conv2D(
        round_filters(32, width_coefficient, depth_divisor), 3,
        padding='same', use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='stem_conv')(img_input)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = Activation(activation, name='stem_activation')(x)

    # build blocks
    total_num_blocks = sum(blocks_json[block]['num_repeat'] for block in blocks_json)
    block_num = 0

    # go through every big block defined in block_args.json
    for idx, block in enumerate(blocks_json):

        # transform this specific block in an object
        block = Struct(**blocks_json[block])

        # ensure num_repeat is bigger than zero
        assert block.num_repeat > 0

        # update block input and output filters based on depth multiplier.
        block.input_filters = round_filters(block.input_filters, width_coefficient, depth_divisor)

        # the first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / total_num_blocks
        x = mb_conv_block(
            block,
            drop_rate=drop_rate,
            activation=activation,
            prefix='block{}a_'.format(idx + 1))(x)

        block_num += 1
        if block.num_repeat > 1:
            block.input_filters = block.output_filters
            block.strides = [1, 1]

            for bidx in range(block.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / total_num_blocks
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )

                x = mb_conv_block(
                    block,
                    prefix=block_prefix,
                    drop_rate=drop_rate,
                    activation=activation)
                block_num += 1

    # build top
    x = Conv2D(
        round_filters(1280, width_coefficient, depth_divisor), 1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='top_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = Activation(activation, name='top_activation')(x)

    return x