# coding:utf-8

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

folder = ['95',
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

image_size = 50

X = []
Y = []
for index, name in enumerate(folder):
    dir = "./pre-all/train/" + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X = X / 255.0

print(X)

Y = np_utils.to_categorical(Y, len(folder))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(folder)))
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

#訓練
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

#評価 & 評価結果出力
print(model.evaluate(X_test, y_test))

# [0.011538714563043074, 0.9979025710419486]

# モデルの保存
open('and_1.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('and_1_weight.hdf5')

import matplotlib.pyplot as plt


def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("model_accuracy.png")

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("model_loss.png")

plot_history(history)

import pickle
with open("history.pickle", mode='wb') as f:
    pickle.dump(history.history, f)
