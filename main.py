from flask import *  
from extract_bottleneck_features import *

import pandas as pd
import numpy as np
import os

import pickle
import cv2
import glob
#import tensorflow as tf
from keras.models import load_model
from random import randint

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






