"""
Code based on https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
"""

# USAGE
# Start the server:
# 	python server.py -m path/to/model_file
# Submita a request via Python:
#	python request.py

# import the necessary packages
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from optparse import OptionParser
from src.config import Struct
import tensorflow as tf
import numpy as np
import pickle
import flask
import json
import cv2
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
binarizer = None
device = None
model = None

if tf.__version__ == '2.1.0':
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
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

    if len(physical_devices) > 0:
        device = '/GPU:0'
        try: 
            tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        except: 
            # Invalid device or cannot modify virtual devices once initialized.
            # Probably an error will raise
            pass
    else:
        device = '/CPU:0'

def check_gpu_availability():
    global device

    if tf.__version__ == '2.1.0':
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0:
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

        if len(physical_devices) > 0:
            device = '/GPU:0'
            try: 
                tf.config.experimental.set_memory_growth(physical_devices[0], True) 
            except: 
                # Invalid device or cannot modify virtual devices once initialized.
                # Probably an error will raise
                pass
        else:
            device = '/CPU:0'

def get_nn(config):
    """
    Args:
        config: Path to the confi file generated during training
    Returns:
        Model instance
    """
    # Loads the model configurations from config file
    # generated during training
    with open(config, 'r') as f:
        C = json.load(f)
    
    C = Struct(**C)

    # Load correct network
    if C.network == 'resnet50':
        from src.architectures import resnet50 as nn
    elif C.network == 'resnet152':
        from src.architectures import resnet152 as nn
    elif C.network == 'vgg16':
        from src.architectures import vgg16 as nn
    else:
        from src.architectures import vgg19 as nn
    
    # Create our model, load weights and then
    # compile it
    input_shape_img = (224, 224, 3)

    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(img_input)
    classifier = nn.classifier(base_layers, trainable=False)

    return Model(inputs=img_input, outputs=classifier)

def load_model(path, config):
    """
    Args:
        path: Path to the model weights
        config: Path to the config file generated during training
    Returns:
        Compiled model ready for predicting
    """
    global model
    model = get_nn(config)

    print('Loading weights from {}'.format(path))
    model.load_weights(path)

    return model

def load_binarizer(path):
    global binarizer
    with open(path, 'rb') as f:
        binarizer = pickle.load(f)

def prepare_image(image, target):
    """
    Args:
        image:  path to image
        target: image shape
    Return:
        A python list of processed images
    """
	# if the image mode is not RGB, convert it
    # read, change channels and resize image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target)
    image = np.expand_dims(image, axis=0)

	# return the processed image
    return image

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
    return np.array([(probs[i], decoded_predictions[i])] for i in range(len(probs)))

@app.route("/predict", methods=["POST"])
def predict():
    global device
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read path
            image = flask.request.files["image"].read().decode("utf-8")

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with tf.device(device):
                preds = model.predict(image)
            results = decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for result in results:
                r = {"label": result[0], "probability": float(result[1])}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-m", dest="model", help="Path to model weights file (hdf5 or h5)")
    parser.add_option("-c", dest="config", help="Path to the config file.")

    (options, args) = parser.parse_args()

    if not options.model:
        parser.error("Pass -m argument")
    if not options.config:
        parser.error("Pass -c argument")

    print("* Loading Keras model and Flask starting server...")
    print("Please wait until server has fully started")
    load_model(options.model, options.config)
    print('Server running')
    app.run()