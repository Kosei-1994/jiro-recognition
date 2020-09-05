from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64
graph = tf.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224
    MODEL_FILE_PATH = './ramen/ml_models/ramen_VGG16_light.h5'
    # classes = ['ラーメン二郎 ひばりヶ丘駅前','ラーメン二郎 茨城守谷','ラーメン二郎 横浜関内','ラーメン二郎 荻窪','ラーメン二郎 歌舞伎町',    'ラーメン二郎 会津若松駅前',    'ラーメン二郎 環七一之江','ラーメン二郎 環七新代田',    'ラーメン二郎 亀戸',    'ラーメン二郎 京急川崎',    'ラーメン二郎 京成大久保',    'ラーメン二郎 桜台駅前',    'ラーメン二郎 札幌',    'ラーメン二郎 三田本',    'ラーメン二郎 小岩','ラーメン二郎 松戸駅前',    'ラーメン二郎 湘南藤沢',    'ラーメン二郎 上野毛',    'ラーメン二郎 新潟',    'ラーメン二郎 新宿小滝橋通り',    'ラーメン二郎 神田神保町',    'ラーメン二郎 西台駅前',  'ラーメン二郎 仙台','ラーメン二郎 千住大橋駅前',    'ラーメン二郎 相模大野',    'ラーメン二郎 池袋東口',    'ラーメン二郎 中山駅前',    'ラーメン二郎 栃木街道典',    'ラーメン二郎 八王子野猿街道2',    'ラーメン二郎 品川',    'ラーメン二郎 府中','ラーメン二郎 目黒',    'ラーメン二郎 立川']
    classes = [
      'めじろ台法政大学前',
      '亀戸',
      '京急川崎',
      '会津若松駅前',
      '横浜関内',
      '歌舞伎町',
      '環七一之江',
      '環七新代田',
      '茨城守谷',
      '荻窪',
      '京成大久保',
      '桜台駅前',
      '札幌',
      '三田本',
      '小岩',
      '松戸駅前',
      '湘南藤沢',
      '上野毛',
      '新潟',
      '新宿小滝橋通り'
      ]
    num_classes = len(classes)


    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert('RGB')
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image)/255.0
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            # print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage