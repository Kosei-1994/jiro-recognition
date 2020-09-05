# coding: utf-8
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection
from keras.preprocessing.image import ImageDataGenerator

# 店舗名を保存
classes = []
num_classes = len(classes)
# VGGで転移学習するため、(224,224,3)に変換する
image_size = 224
dl_path = './npdata/'
dl_filename = 'jiro_scaled.npy'

# 画像の読み込み（水増しはしない）
X = []
Y = []
for index, classlabel in enumerate(classes):
    photos_dir = "./pics/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
xy = (X_train, X_test, y_train, y_test)
np.save(dl_path+dl_filename, xy)