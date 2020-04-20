
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
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.models import load_model, Model
import random

random.seed(a=None, version=2)
x_train_path = 'x_train.npy'
y_train_path = 'y_train.npy'

x_test_path = 'x_test.npy'
y_test_path = 'y_test.npy'


## Callback function to get batch info to print for each batch
# @param batch - the batch
# @param logs - the log data
def batchOutput(batch, logs):
    print("Finished batch: " + str(batch))
    print(logs)

batchLogCallback = LambdaCallback(on_batch_end=batchOutput)




BATCH_SIZE = 64 # if you reduce batch size to 32 or 16 make sure to adjust epochs as necessary
EPOCHS = 25

# the teff branch
input_data = Input(shape=(256,256,1))
output_1 = Conv2D(32, (13,13), activation='relu', padding='same')(input_data)
output_1 = MaxPooling2D(pool_size=(6, 6))(output_1)
output_1 = Conv2D(32, (5, 5), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(4, 4))(output_1)
output_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(4, 4))(output_1)
output_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(output_1)
output_1 = MaxPooling2D(pool_size=(2, 2))(output_1)
output_1 = Flatten()(output_1)
output_1 = Dense(64, activation = 'relu')(output_1)
output_1 = Dropout(0.4)(output_1)
pred = Dense(1, activation='sigmoid', name='prediction')(output_1)

model = Model(inputs=input_data, outputs=pred)

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

# you can tweak the patience values as needed
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',
        monitor='val_acc', save_best_only=True, mode='max'), 
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001),
    keras.callbacks.EarlyStopping(monitor='val_acc', patience=20),
    batchLogCallback
]

x_train = np.asarray(np.load(x_train_path))
y_train = np.asarray(np.load(y_train_path))

x_test = np.asarray(np.load(x_test_path))
y_test = np.asarray(np.load(y_test_path))

x_train = np.expand_dims(x_train, axis=3) # expand dims to what model.fit expects
x_test = np.expand_dims(x_test, axis=3) # expand dims to what model.fit expects
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
###

plt.ioff()

figure_prediction = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Prediction (Train)', 'Prediction (Test)'], loc='upper left')
plt.savefig('prediction_loss')
plt.close(figure_prediction)

figure_prediction = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Prediction (Train)', 'Prediction (Test)'], loc='upper left')
plt.savefig('prediction_acc')
plt.close(figure_prediction)

