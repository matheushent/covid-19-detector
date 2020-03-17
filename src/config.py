from tensorflow.keras import backend as K

class Config:

    def __init__(self):

        self.verbose = True

        self.network = 'resnet50'

        # setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        self.model_path = None

        self.model_name = None

        self.config_filename = None

        self.base_net_weights = None

class Struct:

    def __init__(self, **entries):

        self.__dict__.update(entries)