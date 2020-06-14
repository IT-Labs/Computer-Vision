from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class MiniVGGNet:
    """
    MiniVGG network is a substantially smaller architecture than the original VGGNet
    VGG networks are characterized by two key components
    1. All CONV layers are using only 3x3 filters
    2. Stacks up multiple CONV -> RELU layer sets before applying a POOL layer
    """
    @staticmethod
    def build(width, height, depth, classes):
        """
        The method that builds up the CNN architecture
        :param width: width of the input images
        :param height: height of the input images
        :param depth: depth (channels in the image, 3 for RGB images (red, green, blue channels))
        :param classes: number of classes to predict
        :return:
        """
        model = Sequential()
        inputShape = (height, width, depth)
        # Variable chanDim is set to -1 if the order of the inputShape is (height, width, depth)
        # meaning the depth of the channel comes last in the triple
        chanDim = -1

        if K.image_data_format == "channel_first":
            inputShape = (depth, height, width)
            # if the channel is first in the triple (depth, height, width) we set chanDim to 1
            # Batch normalization layers use the channel dimension in the process, that is why we specficy the order
            chanDim = 1

        # The first set of CONV -> RELU where after each we apply BN layers to avoid overfitting
        # and a POOL -> DO that also help in reducing overfitting and increase the classification accuracy
        # First set of CONV -> RELU -> BN use 32 filters each with 3x3 shape
        # The consecutive CONV -> RELU -> BN layers allow the network to learn more rich features, which
        # is a common practice when training deeper CNNs, before applying POOL layer to reduce the spatial dimensions
        # of the input image
        # Then we apply POOL layer with a size of 2x2, and since we do not provide explicitly stride, keras asumes 2x2 S
        # Finally, a DROPOUT layer with a probabliy of 25%
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # The second set of CONV -> RELU -> BN layers now learn 64 filters with 3x3 shape
        # It is common to increase the number of filters as the spatial input size decreases deeper in the network.
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # We add flatten layer to flatten the output of the previous layer
        # Then we add the only FC layer (512 nodes) with a RELU activation and a BN
        # Further applying a DO layer with p = 0.5
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Finally a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model