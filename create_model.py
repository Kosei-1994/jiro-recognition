from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from numpy.random import seed

from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# 店舗名を保存
classes = []
num_classes = len(classes)
image_size = 224

# numpyデータのロード
X_train, X_test, y_train, y_test = np.load('./npdata/jiro_scaled.npy', allow_pickle = True)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# コールバックを定義
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=2, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      #  factor=0.1,
                                      #  min_lr=1e-5,
                                       factor=0.1,
                                       min_lr=1e-6,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

# モデルを定義
def vgg(X,y):
  model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
  top_model = Sequential()
  top_model.add(Flatten(input_shape=model.output_shape[1:]))
  top_model.add(Dense(256,activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(num_classes,activation='softmax'))

  model = Model(inputs=model.input, outputs=top_model(model.output))
  for layer in model.layers[:15]:
      layer.trainable = True
  
  model.summary()
  return model

model = vgg(X_train, y_train)

# モデルのコンパイル、トレーニング
n_batch = 32
n_epoch = 100

opt = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=n_batch, epochs=n_epoch, callbacks=callbacks, validation_split=0.1).history

# 精度を確認
score = model.evaluate(X_test, y_test, batch_size=n_batch)
print(score)

fig, ax = plt.subplots(figsize=(8,6), dpi=80)
ax.plot(history['loss'], 'b', label='Training Loss', linewidth=4)
ax.plot(history['val_loss'], 'r', label='Validation Loss', linewidth=4)
ax.plot(history['accuracy'], 'g', label='Training Accuracy', linewidth=2)
ax.plot(history['val_accuracy'], 'black', label='Validation Accuracy', linewidth=2)
plt.legend()
plt.show()

# モデルの保存
model.save("./aiapps/ramen/ml_models/ramen_VGG16_light.h5")