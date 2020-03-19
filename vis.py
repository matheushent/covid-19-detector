from optparse import OptionParser
import numpy as np
import json
import cv2
import os

from explain.core.occlusion_sensitivity import OcclusionSensitivity
from explain.core.grad_cam import GradCAM

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from src.config import Struct

import tensorflow as tf

parser = OptionParser()

parser.add_option("-p", dest="path", help="Path to the image.")
parser.add_option("-w", dest="weights", help="Path to the weights file (hdf5 or h5).")
parser.add_option("-c", dest="config", help="Path to the config file.")
parser.add_option("-g", dest="gpu", help="Use GPU or not.", action="store_false", default=True)
(options, args) = parser.parse_args()

if not options.path:
    parser.error("Pass -p argument")

if not options.weights:
    parser.error("Pass -w argument")

if not options.config:
    parser.error("Pass -c argument")

if tf.__version__ == '2.1.0':
    physical_devices = tf.config.list_physical_devices('GPU')
    if options.gpu:
        assert len(physical_devices) > 0, "You set to use GPU but none is available. Make sure you have the correct drivers installed."

        device = '/GPU:0'
        try: 
            tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        except: 
            # Invalid device or cannot modify virtual devices once initialized.
            # Probably an error will raise
            pass
    else:
        device = '/CPU:0'
elif tf.__version__ == '2.0.0':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if options.gpu:
        assert len(physical_devices) > 0, "You set to use GPU but none is available. Make sure you have the correct drivers installed."

        device = '/GPU:0'
        try: 
            tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        except: 
            # Invalid device or cannot modify virtual devices once initialized.
            # Probably an error will raise
            pass
    else:
        device = '/CPU:0'
else:
    raise ImportError("Tensorflow version must be 2.1.0 or 2.0.0")

print('Loading config file...')
with open(options.config, 'rb') as f:
    C = json.load(f)
# Construct an object
C = Struct(**C)

if C.network == 'vgg16':
    from src.architectures import vgg16 as nn
elif C.network == 'vgg19':
    from src.architectures import vgg19 as nn
elif C.network == 'resnet50':
    from src.architectures import resnet50 as nn
elif C.network == 'resnet152':
    from src.architectures import resnet152 as nn

with tf.device(device):
    print('Loading weights from {}'.format(options.weights))
    # Create our model, load weights and then
    # compile it
    input_shape_img = (224, 224, 3)

    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(img_input)
    classifier = nn.classifier(base_layers, trainable=False)

    optimizer = Adam(learning_rate=0.001)

    model =  Model(inputs=img_input, outputs=classifier)
    model.load_weights(options.weights)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

with tf.device('/CPU:0'):
    print('Loading image...')
    image = cv2.imread(options.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data = (np.array([image]), None)

with tf.device(device):
    explainer = GradCAM()
    # Compute Grad-CAM
    print('Computing Grad-CAM')
    arrays = [explainer.explain(data, model, class_index=i, _grid=False, layer_name=nn.get_last_conv_layer_name())[0] for i in range(2)]
    for i, array in enumerate(arrays):
        explainer.save(array, C.common_path, 'grad_cam_class_{}.png'.format(i))

    explainer = OcclusionSensitivity()
    # Compute OcclusionSensitivity
    print('Computing occlusion sensitivity')
    arrays = [explainer.explain(data, model, class_index=i, _grid=False, patch_size=15)[0] for i in range(2)]
    for i, array in enumerate(arrays):
        explainer.save(array, C.common_path, 'occlusion_sensitivity_class_{}.png'.format(i))

print('Exiting...')