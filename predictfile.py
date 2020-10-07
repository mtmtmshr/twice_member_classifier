import os
from flask import Flask, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import keras,sys
from keras.models import Sequential, load_model


UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

image_size = 50
classes = ["sana","momo","mina"]
num_classes = len(classes)


def allowed_file(filename):
    print("." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"]
        if not file or not file.filename:
            flash("ファイルがありません")
            return redirect(request.url)
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            model = load_model("./twice_cnn_aug.h5")
            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)/255
            X = []
            X.append(data)
            X = np.array(X)
            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)
            prediction = classes[predicted] + " " + str(percentage) + " %"
            return render_template("index.html", image="./uploads/{}".format(filename), predict = prediction)
            
    return """
    <!DOCTYPE HTML>
    <html><head>

    <title>ファイルをアップロードして判定する</title></head>
    <body>
    <h1>ファイルをアップロードして判定する<h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    """

from flask import send_from_directory

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
