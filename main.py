from flask import *  
from keras.utils import np_utils
import numpy as np
from glob import glob

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.backend import set_session, get_session
 
from keras import backend as K
from tqdm import tqdm
import pickle
import cv2
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json, Sequential, Model, load_model

from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Lambda, ELU, Cropping2D, Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
#import matplotlib.pyplot as plt

from random import randint

import json
dog_names=[]
with open('data/dog_names.json') as json_file:
    dog_names = json.load(json_file)


global sess
global graph
sess = tf.Session()
graph = tf.get_default_graph()

with graph.as_default():
    set_session(sess)
    ResNet50_model_for_dog_breed = ResNet50(weights='imagenet')
    graph = tf.get_default_graph()

with graph.as_default():
    set_session(sess)
    Res_model_for_adjusting_shape = ResNet50(weights='imagenet', include_top=False)
    graph = tf.get_default_graph()

with graph.as_default():
    set_session(sess)
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_Resnet = bottleneck_features['train']
    valid_Resnet = bottleneck_features['valid']
    test_Resnet = bottleneck_features['test']
    Resnet_Model = Sequential()
    Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
    Resnet_Model.add(Dense(133,activation='softmax'))
    Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')
    graph = tf.get_default_graph()
    

def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

#define generic function for pre-processing images into 4d tensor as input for CNN
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


#predicts the dog breed based on the pretrained ResNet50 models with weights from imagenet
def ResNet50_predict_labels(img_path):
    with graph.as_default():
        # returns prediction vector for image located at img_path
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model_for_dog_breed.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    print(111111111111111111111)
    prediction = ResNet50_predict_labels(img_path)
    print(prediction)
    return ((prediction <= 268) & (prediction >= 151))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    print(len(faces))
    return len(faces) > 0

def Resnet_predict_breed(img_path):
    # extract bottleneck features
    #bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    y = path_to_tensor(img_path)
    print(y.shape)
    y = preprocess_input(y)
    print(y.shape)
    x = Res_model_for_adjusting_shape.predict(y)
    #bottleneck_feature = Res_model_for_adjusting_shape.predict(y)
    print(x.shape)
    # obtain predicted vector
    predicted_vector = Resnet_Model.predict(x)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def get_correct_prenom(word, vowels):
    if word[0].lower() in vowels:
            return "an"
    else:
        return "a"


def predict_image(img_path):
    vowels=["a","e","i","o","u"]
    #if a dog is detected in the image, return the predicted breed.
    with graph.as_default():
        set_session(sess)
        if dog_detector(img_path)==True:
            print(222222222222222)
            predicted_breed=Resnet_predict_breed(img_path).rsplit('.',1)[1].replace("_", " ")
            prenom=get_correct_prenom(predicted_breed,vowels)
            return "The predicted dog breed is " + prenom + " "+ str(predicted_breed) + "."
        #if a human is detected in the image, return the resembling dog breed.
        if face_detector(img_path)==True:
            predicted_breed=Resnet_predict_breed(img_path).rsplit('.',1)[1].replace("_", " ")
            prenom=get_correct_prenom(predicted_breed,vowels)
            return "This photo looks like " + prenom + " "+ str(predicted_breed) + "."
        #if neither is detected in the image, provide output that indicates an error.
        else:
            return "No human or dog could be detected, please provide another picture."

'''
def instantiate_model():
    set_session(sess)
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_Resnet = bottleneck_features['train']
    valid_Resnet = bottleneck_features['valid']
    test_Resnet = bottleneck_features['test']
    Resnet_Model = Sequential()
    Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
    Resnet_Model.add(Dense(133,activation='softmax'))
    Resnet_Model.load_weights("weights.best.Resnet.hdf5")
    graph = tf.get_default_graph()
    Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    model = Resnet_Model
    graph = tf.get_default_graph()
    return model


height = 224
width = 224
dim = (width, height)

'''

IMAGE_FOLDER = 'static/'
#PROCESSED_FOLDER = 'processed/'

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
#app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/')  
def upload():
	return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
		image_ext = cv2.imread(full_filename)
		img_path = full_filename
		#print(image_ext.shape)
		with graph.as_default():
			set_session(sess)
			txt = predict_image(img_path)
			#result = predict_image(img_path, model)
			#txt = result
		final_text = 'Results after Detecting Dog Breed in Input Image'
		return render_template("success.html", name = final_text, img = full_filename, out_1 = txt)
		

@app.route('/info', methods = ['POST'])  
def info():
	return render_template("info.html")  


if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)  






