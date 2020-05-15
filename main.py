from flask import *  
from extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import pickle
import cv2
import glob
#import tensorflow as tf
from keras.models import load_model
from random import randint

ResNet50_model = load_model("ResNet50_model.h5")

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def get_correct_prenom(word, vowels):
    if word[0].lower() in vowels:
            return "an"
    else:
        return "a"

Resnet_Model = load_model("weights.best.Resnet.hdf5")

def Resnet_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet_Model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

height = 224
width = 224
dim = (width, height)


IMAGE_FOLDER = 'static/'
PROCESSED_FOLDER = 'processed/'

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

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
		initial_image = np.copy(image_ext)
		imag = cv2.resize(initial_image, dim, interpolation = cv2.INTER_AREA)
		after_resizing = "processed_imag.jpg"
		cv2.imwrite(os.path.join(PROCESSED_FOLDER, after_resizing), imag)
		#cv2.imwrite(after_resizing, imag)
		#model = load_model("weights.best.Resnet.hdf5")
		full_filename_after_resizing = os.path.join(app.config['PROCESSED_FOLDER'], after_resizing)
		img_path = full_filename_after_resizing
		#imag = np.expand_dims(imag, axis=0)
		#pred = model.predict(imag)
		vowels=["a","e","i","o","u"]
    		#show_img(img_path)
		#if a dog is detected in the image, return the predicted breed.
		if dog_detector(img_path)==True:
			predicted_breed=Resnet_predict_breed(img_path).rsplit('.',1)[1].replace("_", " ")
			prenom=get_correct_prenom(predicted_breed,vowels)
			txt="The predicted dog breed is " + prenom + " "+ str(predicted_breed) + "."
			#if a human is detected in the image, return the resembling dog breed.
		if face_detector(img_path)==True:
			predicted_breed=Resnet_predict_breed(img_path).rsplit('.',1)[1].replace("_", " ")
			prenom=get_correct_prenom(predicted_breed,vowels)
			txt="This photo looks like " + prenom + " "+ str(predicted_breed) + "."
		#if neither is detected in the image, provide output that indicates an error.
		else:
			txt="No human or dog could be detected, please provide another picture."

		final_text = 'Results after Detecting Dog Breed in Input Image'
		return render_template("success.html", name = final_text, img = full_filename, out_1 = txt)
		

@app.route('/info', methods = ['POST'])  
def info():
	return render_template("info.html")  


if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)  






