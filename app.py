# -*- coding: utf-8 -*-

from flask import Flask, redirect, make_response, request, jsonify, render_template, url_for, send_from_directory, session
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
import base64

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


CORS(app)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            print("ファイルがありません")
        else:
            img = request.files["file"]
            filename = secure_filename(img.filename)

            root, ext = os.path.splitext(filename)
            ext = ext.lower()

            gazouketori = set([".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm",
                      ".pxm", ".pnm",  ".sr",  ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic", ".dib"])
            if ext not in gazouketori:
                return render_template('index.html',massege = "対応してない拡張子です",color = "red")
            print("success")
            try:

                # img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # img_url = '/uploads/' + filename

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

                prd = model.predict(X)
                other_labels = np.argsort(prd)[0][::-1][:3]
                other_pros = [prd[0][other_labels[0]], prd[0][other_labels[1]], prd[0][other_labels[2]]]

                details = [
                       '水温95℃を限度に、洗濯機で洗えます。',
                       '水温50℃を限度に、洗濯機で洗えます。',
                       'ハンガー等を使って、つり干しします。',
                       '漂白できません。',
                       '弱い操作による、ウエットクリーニングができます。',
                       '水温60℃を限度に、洗濯機で洗えます。',
                       'ウエットクリーニングができます。',
                       'ドライクリーニングはできません。',
                       '非常に弱い操作による、ウエットクリーニングができます。',
                       '水温30℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                       '水温40℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                       '200℃を限度に、アイロンが使えます。',
                       'ご家庭では洗えません。',
                       'ウエットクリーニングはできません。',
                       '石油系溶剤による、ドライクリーニングができます。',
                       '塩素系・酸素系漂白剤で、漂白できます。',
                       '日陰で、平干しします。',
                       '石油系溶剤による、弱いドライクリーニングができます。',
                       '脱水せずぬれたまま、平干しします。',
                       'パークロロエチレン及び石油系溶剤による、弱いドライクリーニングができます。',
                       '排気温度60℃を上限に、タンブル乾燥できます。',
                       '150℃を限度に、アイロンが使えます。',
                       'タンブル乾燥禁止です。',
                       '水温30℃を限度に、洗濯機で洗えます。',
                       '脱水せずぬれたまま、つり干しします。',
                       '日陰で、脱水せずぬれたまま平干しします。',
                       '水温40℃を限度に、洗濯機で弱い洗濯ができます。',
                       'アイロンは使えません。',
                       '平干しします。',
                       '水温30℃を限度に、洗濯機で弱い洗濯ができます。',
                       '日陰で、つり干しします。',
                       '酸素系漂白剤で、漂白できます。塩素系ではできません。',
                       '水温30℃を限度に、洗濯機で弱い洗濯ができます。',
                       'パークロロエチレン及び石油系溶剤による、ドライクリーニングができます。',
                       '水温40℃を限度に、手洗いできます。',
                       '水温70℃を限度に、洗濯機で洗えます。',
                       '日陰で、脱水せずぬれたままつり干しします。',
                       '排気温度80℃を上限に、タンブル乾燥できます。',
                       '水温40℃を限度に、洗濯機で洗えます',
                       '水温50℃を限度に、洗濯機で弱い洗濯ができます。',
                       '110℃を限度に、スチームなしでアイロンが使えます。']

                icons = ['https://shopping.geocities.jp/ecoloco/images/sentaku-images/95c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/50c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/turiboshi.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/ensosanso-hyouhaku-3.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/cwetw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/60c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/cwet.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/dry-ng.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/cwetww.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/30cww.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/40cww.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/airon200c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/ng.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/cwet-ng.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/fdry.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/ensosanso-hyouhaku-2.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/hiraboshi-hikage.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/fdryw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/hiraboshi-hikage.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/pfdryw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/tanb60c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/airon150c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/tanb-ng.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/30c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/turiboshi-nure.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/hiraboshi-hikagenure.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/40cw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/airon-ng.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/hiraboshi.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/60cw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/turiboshi-hikage.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/sanso-hyouhaku-2.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/30cw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/pfdry.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/hand40c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/70c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/turiboshi-nurehikage.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/tanb80c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/40c.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/50cw.gif',
                        'https://shopping.geocities.jp/ecoloco/images/sentaku-images/airon110c.gif']


                pre1_img_url = icons[other_labels[0]]
                pre1_detail = details[other_labels[0]]
                pre1_pro = str(round(other_pros[0] * 100)) + '%'

                pre2_img_url = icons[other_labels[1]]
                pre2_detail = details[other_labels[1]]
                pre2_pro = str(round(other_pros[1] * 100)) + '%'

                pre3_img_url = icons[other_labels[2]]
                pre3_detail = details[other_labels[2]]
                pre3_pro = str(round(other_pros[2] * 100)) + '%'

            except:
                return render_template('index.html',massege = "解析出来ませんでした",color = "red")

            buf = io.BytesIO()
            image = Image.open(img)
            image.save(buf, 'png')
            qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
            qr_b64data = "data:image/png;base64,{}".format(qr_b64str)

            return render_template('index.html', img=qr_b64data, pre1_img_url=pre1_img_url, pre1_detail=pre1_detail, pre1_pro=pre1_pro, pre2_img_url=pre2_img_url, pre2_detail=pre2_detail, pre2_pro=pre2_pro, pre3_img_url=pre3_img_url, pre3_detail=pre3_detail, pre3_pro=pre3_pro)
    else:
        print("get request")

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
@app.errorhandler(413)
def oversize(error):
    return render_template('index.html',massege = "画像サイズが大きすぎます",color = "red")
@app.errorhandler(400)
def nosubmit(error):
    return render_template('index.html',massege = "画像を送信してください",color = "red")
@app.errorhandler(503)
def all_error_handler(error):
     return 'InternalServerError\n', 503

if __name__ == '__main__':
    app.run()
