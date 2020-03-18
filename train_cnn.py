from __future__ import division

from warnings import filterwarnings
filterwarnings('ignore')

from optparse import OptionParser
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import numpy as np
import pickle
import random
import pprint
import json
import time
import sys
import cv2
import os

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, \
                                       EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, \
                                    AveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, \
                            confusion_matrix

from src.config import Config, Struct

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", dest="path", help="Path to training data.")
parser.add_option("-g", dest="gpu_option", help="If use or not gpu in case it is possible. (Default = True)", action="store_false", default=True)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg16, vgg19, resnet50 and resnet152", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=True).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=True).", action="store_false", default=True)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=True).", action="store_false", default=True)
parser.add_option("--bn", dest="batch_normalization", help="If use or not batch normalization. Available only for both vgg models", action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).", default="config.txt")
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights for classifier model.")
parser.add_option("--mn", dest="model_name", help="Name of the model")

(options, args) = parser.parse_args()

if not options.config_filename:
    parser.error("Pass --config_filename argument")
if not options.model_name:
    parser.error("Pass --mn argument")
if not options.path:
    parser.error("Pass -p argument")

if tf.__version__ == '2.1.0':
    physical_devices = tf.config.list_physical_devices('GPU')
    if options.gpu_option:
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
    if options.gpu_option:
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

with tf.device('/CPU:0'):
    common_path = os.path.join('logs', options.model_name, str(time.time()))
    if not os.path.exists(common_path):
        os.makedirs(common_path)

    C = Config()
    C.model_name = options.model_name
    C.use_horizontal_flips = bool(options.horizontal_flips)
    C.use_vertical_flips = bool(options.vertical_flips)
    C.rot_90 = bool(options.rot_90)
    C.common_path = common_path

    if options.network == 'vgg16':
        from src.architectures import vgg16 as nn
        C.network = 'vgg16'
    elif options.network == 'vgg19':
        from src.architectures import vgg19 as nn
        C.network = 'vgg19'
    elif options.network == 'resnet50':
        from src.architectures import resnet50 as nn
        C.network = 'resnet50'
    elif options.network == 'resnet152':
        from src.architectures import resnet152 as nn
        C.network = 'resnet152'
    else:
        raise ValueError("Not a valid model was passed")

    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path

    if not options.config_filename.endswith('.txt'):
        C.config_filename = options.config_filename + '.txt'

    config_output_filename = os.path.join(common_path, options.config_filename)
    with open(config_output_filename, 'w') as config_f:
        json.dump(C.__dict__, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

    # training params
    batch_size = 8
    learning_rate = 1e-3

    print("Loading images...")
    imagePaths = list(paths.list_images(options.path))
    data = []
    labels = []

    for imagePath in imagePaths:
    	# extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)

    data = np.stack(data, axis=0) / 255.0
    labels = np.array(labels)

    print('Dumping binarizer into pickle file...')
    binarizer = LabelBinarizer()
    binarizer.fit(labels)
    with open(os.path.join(common_path, 'label_binarizer.pickle'), 'wb') as f:
        pickle.dump(binarizer, f)
    labels = binarizer.transform(labels)
    labels = to_categorical(labels)

    print('Splitting')
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.20,
                                                        stratify=labels,
                                                        random_state=24)

    data_gen = ImageDataGenerator(
        rotation_range=24,
        fill_mode='nearest',
        horizontal_flip=C.use_horizontal_flips,
        vertical_flip=C.use_vertical_flips,
        data_format=K.image_data_format()
    )

    model_path = os.path.join(common_path, 'model.hdf5')

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            period=5
        ),
        TensorBoard(
            common_path
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1
        )
    ]

    # as backend is tf
    input_shape_img = (224, 224, 3)

    img_input = Input(shape=input_shape_img)

    base_layers = nn.nn_base(img_input)
    classifier = nn.classifier(base_layers, trainable=True)

    model = Model(inputs=img_input, outputs=classifier)

    #for layer in base_layers.layers:
    #    layer.trainable = False

    print('Compiling model...')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    model.summary()
    
    try:
        model.load_weights(C.base_net_weights)
        print('Weights loaded from {}'.format(C.base_net_weights))
    except Exception as e:
        print(e)
        print('Not possible to load weights from {}'.format(C.base_net_weights))
        pass

# if GPU is available, train on GPU
with tf.device(device):
    print('Training...')
    history = model.fit_generator(
        data_gen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=(len(x_train) // batch_size),
        validation_data=(x_test, y_test),
        validation_steps=(len(x_test) // batch_size),
        verbose=1,
        epochs=options.num_epochs,
        callbacks=callbacks
    )
    print('Saving model in {}'.format(model_path))
    model.save(model_path)
    model.save_weights(os.path.join(common_path, 'model_weights.h5'))

    print('Evaluating the model...')
    preds = model.predict(x_test, batch_size=batch_size)
    preds = np.argmax(preds, axis=1)

    report = classification_report(
        y_test.argmax(axis=1), preds,
        target_names=binarizer.classes_,
        output_dict=True
    )
    print()
    print(report)
    print()
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(common_path, 'classification_report.csv'), index=False, encoding='utf-8')

with tf.device('/CPU:0'):
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(y_test.argmax(axis=1), preds)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    # plot the training loss and accuracy
    k = len(history.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, k), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, k), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, k), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, k), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(common_path, 'metrics_plot.png'))