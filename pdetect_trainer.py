
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
from sklearn.metrics import classification_report, confusion_matrix
#from keras.optimizers import schedules, Adam
import random
import math
import datetime

# set up a fixed random seed for repeatability
random.seed(a=None, version=2)


# the step decay implementation is inspired by 
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))

# the function that performs the step decay
# epoch - the current epoch number 
def lr_decay(epoch):
   initial_lrate = 0.0001
   reduce_by = 0.5
   epochs_count = 10.0
   new_rate = initial_lrate * math.pow(reduce_by,  
           math.floor((1+epoch)/epochs_count))
   return new_rate

## Callback function to get batch info to print for each batch
# batch - the batch
# logs - the log data
def batchOutput(batch, logs):
    print("Finished batch: " + str(batch))
    print(logs)

# set up the batchOutput callback for reference
batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

# Sets up the optimizer
def get_optimizer():
  return keras.optimizers.Adam(lr_schedule)  

# set up the LearningRateScheduler
lrate = LearningRateScheduler(lr_decay)

# you will need to download the numpy binaries from sharepoint and
# put them in the same directory

# paths to NumPy binaries for training set
x_train_path = 'x_train.npy'
y_train_path = 'y_train.npy'

# paths to NumPy binaries for test set
x_test_path = 'x_test.npy'
y_test_path = 'y_test.npy'

loss_history = LossHistory()

# Initialize the optimizer
keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

BATCH_SIZE = 64 
EPOCHS = 100

# the neural network model
input_data = Input(shape=(256,256,1))
output_1 = Conv2D(128, (3,3), activation='relu', padding='same')(input_data)
output_1 = MaxPooling2D(pool_size=(4, 4))(output_1)
output_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = BatchNormalization()(output_1)
output_1 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = BatchNormalization()(output_1)
output_1 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = Flatten()(output_1)
output_1 = Dense(512, activation = 'relu')(output_1)
output_1 = Dropout(0.8)(output_1)
output_1 = Dense(64, activation = 'relu')(output_1)
output_1 = Dropout(0.8)(output_1)
pred = Dense(1, activation='sigmoid', name='prediction')(output_1)

model = Model(inputs=input_data, outputs=pred)

# print the model summary
print(model.summary())

# plot the current model and save it to disk for later reference
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# compile the model
model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])


# the callback list
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',
        monitor='val_acc', save_best_only=True, mode='max'), 
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000001),
    keras.callbacks.EarlyStopping(monitor='val_acc', patience=20),
    batchLogCallback, loss_history, lrate
]

# load the training and test binaries
x_train = np.asarray(np.load(x_train_path))
y_train = np.asarray(np.load(y_train_path))
x_test = np.asarray(np.load(x_test_path))
y_test = np.asarray(np.load(y_test_path))

# expand dims to what model.fit expects
x_train = np.expand_dims(x_train, axis=3) 
x_test = np.expand_dims(x_test, axis=3) 

print(x_train.shape) # check the shape

history = model.fit(x_train, # x-train contains data, y-train contains labels
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_data=(x_test,y_test),
                      verbose=2)


###
# Plot training & validation accuracy values
# after training is complete
###

plt.ioff()

figure_prediction = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Prediction (Train)', 'Prediction (Test)'], loc='upper left')
plt.savefig('prediction_loss_'+str(datetime.datetime.now())+'.png')
plt.close(figure_prediction)

figure_prediction = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Prediction (Train)', 'Prediction (Test)'], loc='upper left')
plt.savefig('prediction_acc_' +str(datetime.datetime.now())+'.png')
plt.close(figure_prediction)


