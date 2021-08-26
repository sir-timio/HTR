import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from validators import url as urlval
from core import model_predict
from flask_cors import CORS, cross_origin
import requests
import string
import random

UPLOAD_FOLDER = os.path.join('/home', 'mts', 'mysite', 'tmp')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def id_generator(ext, size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size)) + '.' + ext


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"prediction": "no file, upload only one"})
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(filename)
                filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                prediction = model_predict(filename)
                os.remove(filename)
                return jsonify({'prediction': prediction})
            else:
                return jsonify({'prediction': 'bad file, upload another'})

        elif 'imgurl' in request.form:
            imgurl = request.form['imgurl']
            extension = os.path.splitext(imgurl)[1]
            if not urlval(imgurl) and extension not in ALLOWED_EXTENSIONS:
                return jsonify({'prediction': 'bad url, try another'})
            r = requests.get(imgurl, allow_redirects=True)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], id_generator(extension))
            with open(filename, 'wb') as f:
                f.write(r.content)
            prediction = model_predict(filename)
            os.remove(filename)
            return jsonify({'prediction': prediction})

        else:
            return jsonify({"prediction": "no file or url"})
    else:
        return jsonify({"prediction": "please, post file or url"})


if __name__ == "__main__":
    app.run()
