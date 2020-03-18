"# covid-19-detector"

## Pre-requisites

We assume you have an GPU available and python 3.6+ installed. Make sure you have installed all the correct drivers for gpu training. Check out ```https://www.tensorflow.org/install/gpu``` for information about GPU setup.

Assuming you're ok, run ```pip install -r requirements.txt``` to install all necessary packages.

## Training

Set the following options to train:

* ```-p```: Path to the dataset (dataset/)
* ```-g```: If use or not gpu in case it is possible. (Default = True)
* ```--network```: Base network to use. Supports vgg16, vgg19, resnet50 and resnet152 (Default = resnet50)
* ```--hf```: Augment with horizontal flips in training. (Default=True)
* ```--vf```: Augment with vertical flips in training. (Default=True)
* ```--rot```: Augment with 90 degree rotations in training. (Default=True)
* ```--bn```: If use or not batch normalization. Available only for both vgg models
* ```--num_epochs```: Number of epochs. (Default = 100)
* ```--config_filename```: Name of txt file that stores all the metadata related to the training (to be used when testing)
* ```--input_weight_path```: Input path for weights for classifier model
* ```--mn```: Name of the model

Example of training command line:

```python train_cnn.py -p dataset/ -g True --network vgg19 --mn vgg19_1```

## Pattern Visualization

Set the following options to run the vis script:

* ```-p```: Path to the image
* ```-m```: Path to the model file (hdf5)
* ```-c```: Path to the config file
* ```-g```: Use GPU or not (Default = True)

## Server

To run the server just enter the following line on cmd:

```python server.py -m path/to/model/weights/file```

The weights file were generated after training

After, to make an http request run request.py:

```python request.py -p path/to/folder/containing/images/to/be/classified```

## Credits

* Many thanks to Joseph Paul Cohen who makes the dataset available on https://github.com/ieee8023/covid-chestxray-dataset
* Many thanks to Adrian Rosebrock for providing some interesting code (https://github.com/jrosebr1)