import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.models import load_model
import keras
import cv2

# This file was used for a live demonstration

# Path to the demo images
imageA = '/Volumes/Storage/chest_xray/resized_person1671_virus_2887.jpeg'
imageB = '/Volumes/Storage/chest_xray/resized_NORMAL2-IM-0279-0001.jpeg'

def getDemoPrediction(file, model):
    model = load_model(model)
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1,256,256,1)
    result = model.predict(image)
    result = np.rint(result)

    if (result): print ('The patient has pneumonia (probably).')
    else: print ('The patient does not have pneumonia (probably).')

# First result (imageA) has pneumonia
getDemoPrediction(imageA,'best_model.04-0.41-0.88.h5')

# Second result (imageB) does not have pneumonia
getDemoPrediction(imageB,'best_model.04-0.41-0.88.h5')