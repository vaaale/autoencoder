import os

import numpy as np
from flask import Flask, request, redirect, url_for, flash
from flask import send_from_directory
from scipy.misc import imread, imsave
from scipy.misc import imresize
from werkzeug.utils import secure_filename

from models import build_model, DeepAuto

UPLOAD_FOLDER = '/tmp/uploads/input'
OUTPUT_FOLDER = '/tmp/uploads/output'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

autoencoder = build_model('model', DeepAuto, (None, None, 3))
autoencoder.summary()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
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

            print('Predicting....')
            image = (imresize(imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), mode='RGB'), (480, 640)) / 255.).astype(np.float32)

            output = autoencoder.predict(image.astype(np.float32).reshape((1, *image.shape)))
            output = (output.reshape(image.shape) * 255).astype(np.uint8)
            imsave(os.path.join(app.config['OUTPUT_FOLDER'], filename), output)
            return redirect(url_for('reconstructed_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/input/<filename>')
def uploaded_file(filename):
    print('Serving imate ' + os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/output/<filename>')
def reconstructed_file(filename):
    print('Servinng image ' + os.path.join(app.config['OUTPUT_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.secret_key = 'alex'
    app.run(threaded=False)
