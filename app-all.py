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
SEEDS_FOLDER = './seeds'
app.config['SEEDS_FOLDER'] = SEEDS_FOLDER
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
            model = model_from_json(open('and.json', 'r').read())

            # 重みの読み込み
            model.load_weights('and_weight.hdf5')


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

            other_labels = np.argsort(prd)[0][::-1][:3]
            other_pros = [prd[0][other_labels[0]], prd[0][other_labels[1]], prd[0][other_labels[2]]]

            names = ['95',
                    '50',
                    'hanging_dry',
                    'not_bleachable',
                    'wetcleaning_weak',
                    '60',
                    'wetcleaning_ok',
                    'donot_drycleaning',
                    'weetcleaning_very_weak',
                    '30_very_weak',
                    '40_very_weak',
                    'ironing_upto200',
                    'not_washable',
                    'donot_wetcleaning',
                    'drycleaning_F',
                    'bleachable',
                    'flat_dry_shade',
                    'drycleaning_F_weak',
                    'flat_dry_wet',
                    'drycleaning_P_weak',
                    'tumble_dry_upto60',
                    'ironing_upto150',
                    'donot_tumble_dry',
                    '30',
                    'hanging_dry_wet',
                    'flat_dry_wetshade',
                    '40_weak',
                    'donot_ironing',
                    'flat_dry',
                    '60_weak',
                    'hanging_dry_shade',
                    'bleachable_oxygen',
                    '30_weak',
                    'drycleaning_P',
                    'hand-wash',
                    '70',
                    'hanging_dry_wetshade',
                    'tumble_dry_upto80',
                    '40',
                    '50_weak',
                    'ironing_upto110']

            details = [
                   '水温90℃を限度に、洗濯機で洗えます。',
                   '水温℃を限度に、洗濯機で洗えます。',
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
                   '水温30℃を限度に、洗濯機で洗えます',
                   '水温50℃を限度に、洗濯機で弱い洗濯ができます。',
                   '110℃を限度に、スチームなしでアイロンが使えます。']


            name1 = names[other_labels[0]]
            pre1_img_url = '/seeds/' + name1 + '.png'
            pre1_detail = details[other_labels[0]]
            pre1_pro = str(round(other_pros[0] * 100)) + '%'


            name2 = names[other_labels[1]]
            pre2_img_url = '/seeds/' + name2 + '.png'
            pre2_detail = details[other_labels[1]]
            pre2_pro = str(round(other_pros[1] * 100)) + '%'


            name3 = names[other_labels[2]]
            pre3_img_url = '/seeds/' + name3 + '.png'
            pre3_detail = details[other_labels[2]]
            pre3_pro = str(round(other_pros[2] * 100)) + '%'

            return render_template('index.html', img_url=img_url, pre1_img_url=pre1_img_url, pre1_detail=pre1_detail, pre1_pro=pre1_pro, pre2_img_url=pre2_img_url, pre2_detail=pre2_detail, pre2_pro=pre2_pro, pre3_img_url=pre3_img_url, pre3_detail=pre3_detail, pre3_pro=pre3_pro)
    else:
        # ターミナル及びコマンドプロンプトに出力するメッセージ
        print("get request")

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/seeds/<filename>')
def seed_file(filename):
    return send_from_directory(app.config['SEEDS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
