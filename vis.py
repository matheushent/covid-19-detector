from optparse import OptionParser
import numpy as np
import json
import cv2
import os

from explain.core.occlusion_sensitivity import OcclusionSensitivity
from explain.core.grad_cam import GradCAM

from src.config import Struct

import tensorflow as tf

parser = OptionParser()

parser.add_option("-p", dest="path", help="Path to the image.")
parser.add_option("-m", dest="model_path", help="Path to the model file (hdf5).")
parser.add_option("-c", dest="config", help="Path to the config file.")
(options, args) = parser.parse_args()

if not options.path:
    parser.error("Pass -p argument")

if not options.model_path:
    parser.error("Pass -m argument")

if not options.config:
    parser.error("Pass -c argument")

print('Loading config file...')
with open(options.config, 'rb') as f:
    C = json.load(f)
# Construct an object
C = Struct(**C)

print('Loading model from {}'.format(options.model_path))
model = tf.keras.models.load_model(options.model_path)

print('Loading image...')
image = cv2.imread(options.path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

data = (np.array([image]), None)

explainer = GradCAM()
# Compute Grad-CAM
print('Computing Grad-CAM')
arrays = [explainer.explain(data, model, class_index=i, _grid=False)[0] for i in range(2)]
for i, array in enumerate(arrays):
    explainer.save(array, C.common_path, 'grad_cam_class_{}.png'.format(i))

explainer = OcclusionSensitivity()
# Compute OcclusionSensitivity
print('Computing occlusion sensitivity')
arrays = [explainer.explain(data, model, class_index=i, _grid=False)[0] for i in range(2)]
for i, array in enumerate(arrays):
    explainer.save(array, C.common_path, 'occlusion_sensitivity_class_{}.png'.format(i))

print('Exiting...')