from tensorflow.keras.applications import ResNet152, ResNet50, \
                                          VGG16, VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from optparse import OptionParser
from src.config import Struct
import tensorflow as tf
import numpy as np
import pickle
import time
import json
import cv2
import os

binarizer = None
device = None
model = None
C = None

parser = OptionParser()

parser.add_option("-w", dest="weights", help="Path to model weights file (hdf5 or h5)")
parser.add_option("-c", dest="config", help="Path to the config file.")
parser.add_option("-p", dest="path", help="Path to folder containing images to be classified")
parser.add_option("-g", dest="gpu", help="Use GPU or not.", action="store_false", default=True)

(options, args) = parser.parse_args()

if not options.weights:
    parser.error("Pass -w argument")
if not options.config:
    parser.error("Pass -c argument")
if not options.path:
    parser.error("Pass -p argument")

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

input_shape_img = (224, 224, 3)

img_input = Input(shape=input_shape_img)

print('Loading pre-trained weights...')
if C.network == 'vgg16':
    from src.architectures import vgg16 as nn

    # define input of our network
    input_shape_img = (224, 224, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)

elif C.network == 'vgg19':
    from src.architectures import vgg19 as nn

    # define input of our network
    input_shape_img = (224, 224, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = VGG19(weights='imagenet', include_top=False, input_tensor=img_input)
    
elif C.network == 'resnet50':
    from src.architectures import resnet50 as nn

    # define input of our network
    input_shape_img = (224, 224, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = ResNet50(weights='imagenet', include_top=False, input_tensor=img_input)
    
elif C.network == 'resnet152':
    from src.architectures import resnet152 as nn

    # define input of our network
    input_shape_img = (224, 224, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = ResNet152(weights='imagenet', include_top=False, input_tensor=img_input)
    
elif C.network == 'efficientnet-b0':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (224, 224, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.0, 1.0, input_tensor=img_input, dropout_rate=0.2)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b0_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b1':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (240, 240, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.0, 1.1, input_tensor=img_input, dropout_rate=0.2)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b1_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b2':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (260, 260, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.1, 1.2, input_tensor=img_input, dropout_rate=0.3)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b2_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b3':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (300, 300, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.2, 1.4, input_tensor=img_input, dropout_rate=0.3)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b3_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b4':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (380, 380, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.4, 1.8, input_tensor=img_input, dropout_rate=0.4)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b4_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b5':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (456, 456, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.6, 2,2, input_tensor=img_input, dropout_rate=0.4)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b5_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b6':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (528, 528, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(1.8, 2.6, input_tensor=img_input, dropout_rate=0.5)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b6_noisy-student_notop.h5")
    
elif C.network == 'efficientnet-b7':
    from src.architectures import efficientnet as nn

    # define input of our network
    input_shape_img = (600, 600, 3)
    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(2.0, 3.1, input_tensor=img_input, dropout_rate=0.5)
    base_layers = Model(img_input, base_layers)
    base_layers.load_weights("https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b7_noisy-student_notop.h5")
    

with tf.device(device):

    print('Loading weights from {}'.format(options.weights))
    classifier = nn.classifier(base_layers.output, trainable=False)

    optimizer = Adam(learning_rate=0.0001)

    model =  Model(inputs=base_layers.input, outputs=classifier)
    model.load_weights(options.weights)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

def load_binarizer(path):
    global binarizer
    with open(path, 'rb') as f:
        binarizer = pickle.load(f)
    
def prepare_images(images, target):
    """
    Args:
        image:  list containing images paths
        target: image shape
    Return:
        numpy.ndarray of shape (batch_size, width, length, channels)
    """
	# if the image mode is not RGB, convert it
    # read, change channels and resize image
    _images = []
    for image in images:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target)
        _images.append(image)
	# return the processed image
    return np.stack(_images, axis=0)

def decode_predictions(preds):
    """
    Args:
        preds: model predictions
    Returns:
        A numpy.nparray [(decoded prediction, probability), (decoded prediction, probability)...]
    """
    global binarizer
    indices = np.argmax(preds, axis=1)
    decoded_predictions = binarizer.inverse_transform(indices)
    probs = [preds[i][index] for i, index in enumerate(indices)]

    array = []
    for i in range(len(probs)):
        array.append((probs[i], decoded_predictions[i]))
    return np.array(array)

print("Loading binarizer...")
load_binarizer(os.path.join(C.common_path, 'label_binarizer.pickle'))

images_path = [os.path.join(options.path, image_name) for image_name in os.listdir(options.path)]

images = prepare_images(images_path, target=(224, 224))

data = {"success": False}

print("Predicting...")
with tf.device(device):
    preds = model.predict(images)
results = decode_predictions(preds)

data["predictions"] = []

for result in results:
    r = {"label": result[1], "probability": float(result[0])}
    data["predictions"].append(r)
    
data["success"] = True

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(images.shape[0]):
    text = data["predictions"][i]["label"]
    prob = str(round(data["predictions"][i]["probability"] * 100, 3)) + ' %'

    if text == 'covid':
        color = (255, 0, 0)
    else:
        color = (0, 255, 0)

    cv2.putText(images[i], text, (10, 25), font, .5, color, 1, cv2.LINE_AA)
    cv2.putText(images[i], prob, (10, 15), font, .5, color, 1, cv2.LINE_AA)

    cv2.imshow('Image', images[i])
    cv2.waitKey(0);

    print('Saving image...')
    cv2.imwrite(os.path.join('predicted', os.listdir(options.path)[i]), images[i])
    cv2.destroyAllWindows()