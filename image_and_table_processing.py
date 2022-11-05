# import necessary packages for data loading and processing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

import pandas as pd
import numpy as np

import cv2
import os


def load_table(inputPath): # load_table function accepts the path to the input dataset

    df = pd.read_csv(inputPath, delimiter = "\t", header = None)
    #df = pd.read_csv(inputPath, delimiter = " ", header = None)

    return df


def image_data_extract(imagePaths): # load_table function accepts the path to the input images


    images = []


    for (i, imagepath) in enumerate(imagePaths):

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (224,224),interpolation=cv2.INTER_AREA)  # width followed by height

        image_to_append = image

        # add the resized images to the images list on which the network will be trained
        images.append(image_to_append)
        

#     # # return the set of images as an array
    return np.array(images)

        




