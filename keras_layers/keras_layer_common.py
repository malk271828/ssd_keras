import keras
import keras.backend as K
from packaging import version

def isChannelsLast():
    if version.parse(keras.__version__) <= version.parse("2.2.4"):
        ret = (K.image_dim_ordering() == "tf")
    else:
        ret = (K.image_data_format() == "channels_last")
    return ret
