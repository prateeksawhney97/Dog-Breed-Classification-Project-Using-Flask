from flask import *  

import pandas as pd
import numpy as np
import os

import pickle
import cv2
import glob
#import tensorflow as tf
from keras.models import load_model
from random import randint

height = 224
width = 224
dim = (width, height)


IMAGE_FOLDER = 'static/'

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

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

		model = load_model("model.h5")
		imag = np.expand_dims(imag, axis=0)
		pred = model.predict(imag)
		pred_0 = pred[0][0]
		pred_1 = pred[0][1]
		pred_2 = pred[0][2]
		pred_3 = pred[0][3]
		pred_4 = pred[0][4]
		value = max(pred_0, pred_1, pred_2, pred_3, pred_4)
		if(value==pred_0):
			txt = "Ajoba Temple, Goa"
			desc = "Located at the entrance of Keri (or Querim) beach in Goa. It is a small temple but quite interesting on the sands of the beach. Keri beach is the northernmost beach of Goa. Thereafter Goa ends and Maharastra starts."
			made = "A beautiful temple painted in turquoise and orange, located on the shores of Keri beach in Pernem. The much-famed jatra of Ajoba devasthan at Keri-Pernem happens in the first week of February. Devotees from Goa and neighbouring states, throng to the temple to witness this."
		
		

			
		final_text = 'Results after Detecting Monument in Input Image'
		return render_template("success.html", name = final_text, img = full_filename, out_1 = txt)
		

@app.route('/info', methods = ['POST'])  
def info():
	return render_template("info.html")  


if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)  






