
# numpy and matplotlib
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
# os
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import graphviz
# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, LearningRateScheduler
from keras.utils import plot_model
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
#from keras.optimizers import schedules, Adam
import random
import math
import datetime

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# you will need to download the numpy binaries from sharepoint and
# put them in the same directory as this file to run this
x_train_path = 'x_train.npy'
y_train_path = 'y_train.npy'

x_test_path = 'x_test.npy'
y_test_path = 'y_test.npy'

batch_size = 64
num_of_test_samples = 624

test_datagen = ImageDataGenerator(rescale=1. / 255)

model = load_model('best_model.04-0.41-0.88.h5')
# you will need to download the numpy binaries from sharepoint and
# put them in the same directory as this file to run this
x_train = np.asarray(np.load(x_train_path))
y_train = np.asarray(np.load(y_train_path))



x_test = np.asarray(np.load(x_test_path))
y_test = np.asarray(np.load(y_test_path))

y_test[y_test == 'normal'] = 0
y_test[y_test == 'pneumonia'] = 1
y_test = y_test.astype(int)
x_train = np.expand_dims(x_train, axis=3) # expand dims to what model.fit expects
x_test = np.expand_dims(x_test, axis=3) # expand dims to what model.fit expects
print(x_train.shape) # check the shape


#Confution Matrix and Classification Report

y_pred = []
init=0


p=model.predict(x_test)
y_pred = np.rint(p)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Normal', 'Pneumonia']
print(classification_report(y_test, y_pred, target_names=target_names))

np.set_printoptions(precision=2)


df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), ['Normal', 'Pneumonia'], ['Normal', 'Pneumonia'])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='d') # font size
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


