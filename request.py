"""
Code based on Adrian's repo (https://github.com/jrosebr1/simple-keras-rest-api)
"""

# USAGE
# python request.py -p path/to/folder/containing_images

# import the necessary packages
from optparse import OptionParser
import requests
import os

parser = OptionParser()
parser.add_option("-p", dest="path", help="Path to the folder containing images to be classified")
(options, args) = parser.parse_args()

if not options.path:
    parser.error("Pass -p argument")

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"

# list comprehension where each element of the list is the path
# of an image that will be classified
images = [os.path.join(options.path, path) for path in os.listdir(options.path)]

for image in images:
	payload = {"image": image}

	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()

	# ensure the request was sucessful
	if r["success"]:
		# loop over the predictions and display them
		for (i, result) in enumerate(r["predictions"]):
			print("{}. {}: {:.4f}".format(i + 1, result["label"],
				result["probability"]))

	# otherwise, the request failed
	else:
		print("Request failed")