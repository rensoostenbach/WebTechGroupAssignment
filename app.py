from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_login import LoginManager, UserMixin, \
                                login_required, login_user, logout_user 
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import os.path
import base64
sys.path.append(os.path.abspath('./model'))
from load import *

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])

#init flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
login_manager = LoginManager()
global model, graph
model, graph = init()

#def convertImage(imgData1):
#	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#	#print(imgstr)
#	with open('output.png','wb') as output:
#		output.write(base64.b64decode(imgstr))
		
@app.route('/')
def index():
	return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return predict(filename)
    return render_template('upload.html')
	
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
	
@app.route('/predict/', methods=['GET', 'POST'])
def predict(file):
	#imgData = request.get_data()
	#convertImage(imgData)
	#print "debug"
	x = imread(os.path.join(app.config['UPLOAD_FOLDER'], file), mode = 'RGB')
	#x = np.invert(x)
	x = imresize(x,(128, 128))
	x = x.reshape(1, 128, 128, 3)
	#print "debug2"
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#print "debug3"
		#convert the response to a string
		response = np.array_str(out) + '\n' + np.array_str(np.argmax(out,axis=1))
		return response	

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True)
	