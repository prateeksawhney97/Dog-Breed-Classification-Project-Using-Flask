from flask import *  
from extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.python.keras.backend import set_session, get_session
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input
from keras import backend as K
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Lambda, ELU, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import pickle
import cv2
import glob
import tensorflow as tf
from keras.models import load_model
from random import randint

ResNet50_model = ResNet50(weights='imagenet')

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
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def Resnet_predict_breed(Resnet_model, img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def get_correct_prenom(word, vowels):
    if word[0].lower() in vowels:
            return "an"
    else:
        return "a"

def predict_image(img_path, model):
    vowels=["a","e","i","o","u"]
    #if a dog is detected in the image, return the predicted breed.
    if dog_detector(img_path)==True:
        predicted_breed=Resnet_predict_breed(model, img_path).rsplit('.',1)[1].replace("_", " ")
        prenom=get_correct_prenom(predicted_breed,vowels)
        return "The predicted dog breed is " + prenom + " "+ str(predicted_breed) + "."
    #if a human is detected in the image, return the resembling dog breed.
    if face_detector(img_path)==True:
        predicted_breed=Resnet_predict_breed(model, img_path).rsplit('.',1)[1].replace("_", " ")
        prenom=get_correct_prenom(predicted_breed,vowels)
        return "This photo looks like " + prenom + " "+ str(predicted_breed) + "."
    #if neither is detected in the image, provide output that indicates an error.
    else:
        return "No human or dog could be detected, please provide another picture."


def instantiate_model():
    #build model
    global model
    global graph
    global sess
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_Resnet = bottleneck_features['train']
    valid_Resnet = bottleneck_features['valid']
    test_Resnet = bottleneck_features['test']
    Resnet_Model = Sequential()
    Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
    Resnet_Model.add(Dense(133,activation='softmax'))
    Resnet_Model.load_weights("weights.best.Resnet.hdf5")
    Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model = Resnet_Model



height = 224
width = 224
dim = (width, height)

instantiate_model()

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
		
		with graph.as_default():
			result = predict_image(img_path, model)
		final_text = 'Results after Detecting Dog Breed in Input Image'
		return render_template("success.html", name = final_text, img = full_filename, out_1 = txt)
		

@app.route('/info', methods = ['POST'])  
def info():
	return render_template("info.html")  


if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)  






