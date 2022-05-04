from multiprocessing import pool
from pyexpat import model
from statistics import mode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras import backend as K

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # 1st block: CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu", input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd block: CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd block: CONV => ELU => CONV => ELU => POOL 
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th block: FC => ELU
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 5th block: FC => ELU
        model.add(Dense(64, kernel_initializer="he_normal", activation="elu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 6th block
        model.add(Dense(classes, kernel_initializer="he_normal", activation="softmax"))

        return model
        