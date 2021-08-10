import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pprint

colab_root = '/content/drive/MyDrive/colab_root/'

# 画像を読み込む。
img = image.load_img(colab_root+'ramen_1.jpg',  target_size=(100,100))
img = np.array(img)

#plt.imshow(img)
#plt.show()

# flow に渡すために配列を四次元化
img = img[None, ...]

# 画像変形のジェネレータを作成
datagen = image.ImageDataGenerator(rotation_range=150)
gen = datagen.flow(img, batch_size = 1)

# バッチの実行
batches = next(gen)
g_img = batches[0]/255
g_img = g_img[None, ...]

# modelの読み込み
model = load_model(colab_root + 'ramen_hiyashi_acc0.9675.h5')
# 正解ラベルの定義
label=['ramen', 'hiyashi']

# 判別
pred = model.predict(g_img, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print('name:',pred_label)
print('score:',score)
