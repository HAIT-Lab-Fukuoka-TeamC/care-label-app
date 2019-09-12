from flask import Flask, redirect, request, jsonify, render_template, url_for, send_from_directory, session
from keras import models
from PIL import Image
from keras.models import load_model
from flask_cors import CORS
from PIL import ImageFile
from keras.backend import tensorflow_backend as backend
import keras
import numpy as np
import sys, os, io
import glob
import tensorflow as tf
from keras.models import model_from_json
from werkzeug import secure_filename

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


CORS(app)

# http://127.0.0.1:5000/にアクセスしたら、一番最初に読み込まれるページ
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

# 画像を選択し、アップロードボタンを押したら呼ばれる処理
@app.route('/predict', methods=['GET','POST'])
def predict():
    # リクエストがPOSTの時にのみ、条件文の中のコードが実行されます。
    if request.method == "POST":
        if 'file' not in request.files:
            print("ファイルがありません")
        else:
            img = request.files["file"]
            filename = secure_filename(img.filename)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = '/uploads/' + filename

            graph = tf.get_default_graph()
            backend.clear_session() # 2回以上連続してpredictするために必要な処理


            # モデルの読み込み
            model = model_from_json(open('and_1.json', 'r').read())

            # 重みの読み込み
            model.load_weights('and_1_weight.hdf5')


            image_size = 50

            image = Image.open(img)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = np.array(data)
            X = X.astype('float32')
            X = X / 255.0
            X = X[None, ...]

            # 受け取った画像をnumpyのアレイに変換
            prd = model.predict(X)
            # 配列の最大要素のインデックスを返しprelabelに代入します
            prelabel = np.argmax(prd, axis=1)
            probability = max(prd[0])


            if prelabel == 0:
                name = "40_weak"
            elif prelabel == 1:
                name = "donot_tumble_dry"
            elif prelabel == 2:
                name = "ironing_upto150"
            elif prelabel == 3:
                name = "not_bleachable"



            return render_template('index.html',name=name, img_url=img_url, probability=probability)
    else:
        # ターミナル及びコマンドプロンプトに出力するメッセージ
        print("get request")

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
