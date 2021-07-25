# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# ファッションMNISTのダウンロード
# train_images, train_labels はモデルの訓練に利用される訓練用データセット
# test_images, test_labels はテスト用データセット
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 画像は28px * 28px * モノクロ256色。labelは衣料品の種類を表す0から9までの整数
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
'''

# モノクロ256階調を0か1にする
train_images = train_images / 255.0
test_images = test_images / 255.0

# 層の定義
# keras.layers.Flatten: 28*28*2の二次元配列を28*28=784pxの1次元配列に変換する。画像の中に詰まれているピクセルの行を取り崩し、横に並べる。データフォーマットの変換だけ。
# keras.layers.Dense(relu): 密結合あるいは全結合されたニューロンの層。128個のノードを持つ。活性化関数はReLU
# keras.layers.Dense(softmax): 合計が1になる10個の確率の層。10クラスのどれであるかを示す
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
# optimizer: モデルが見ているデータと損失関数の値から、どのようにモデルを更新するかを決定する
# loss: 損失関数。訓練中のモデルが不正確であるほど大きな値となる関数。この関数の値を最小化することで、訓練中のモデルを正しい方向に向かわせようとする
# metrics: 訓練とテストのステップを監視するために使用。画像が正しく分類された比率を表す
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
# 画像とラベルの対応関係を学習する
model.fit(train_images, train_labels, epochs=5)

# 正解率の評価
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 予測する
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

# 訓練済みモデルを使って1枚の画像を判定する
# テスト用データセットから画像を1枚取り出す
img = test_images[0]
# 画像を1枚だけのバッチのメンバーにする
# kerasはリストに対して予測を行うように作られているので1枚でもリスト化する必要がある
img = (np.expand_dims(img,0))

predictions_single = model.predict(img)
print(predictions_single)