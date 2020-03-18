"# covid-19-detector" 

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

## Pattern Visualization

Set the following options to run the vis script:

* ```-p```: Path to the image
* ```-m```: Path to the model file (hdf5)
* ```-c```: Path to the config file
* ```-g```: Use GPU or not (Default = True)

## Credits

* Many thanks to Joseph Paul Cohen who makes the dataset available on https://github.com/ieee8023/covid-chestxray-dataset
* Many thanks to Adrian Rosebrock for providing some interesting code (https://github.com/jrosebr1)