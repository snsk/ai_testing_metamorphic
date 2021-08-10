# keras memo

## 基本

* https://www.tensorflow.org/tutorials/keras/classification?hl=ja
* TensorFlowのモデルを構築し訓練するためのハイレベルのAPIである tf.kerasを使用
* kerasはデータセットの取得にも対応している
    * https://keras.io/ja/datasets/

* ニューラルネットワークを構築するには、まずモデルの階層を定義し、その後モデルをコンパイルする
    * チュートリアルで利用中間レイヤは Dence（全結合ニューラルネットワークレイヤ）と ReLU
    * そのあと回答を出す出力レイヤは、DenceとSoftMax

## 画像の行列変換

```python
img = img_to_array(load_img(colab_root + 'ramen_2.jpg', target_size=(100,100)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]
```

* load_img: 画像をPIL形式に変換する
* PIL：Python Image Library。Pillowというライブラリでサポートされる。
* img_to_array: PIL形式の画像をnp_array形式に変換する
    * img_to_array 直後
        ```
        array([[[183.,  93., 105.],
                [180.,  99., 105.],
                [178.,  98., 101.],
                ...,
                [140.,  36.,  47.],
                [142.,  42.,  52.],
                [136.,  41.,  49.]],
        ```
    * img_to_array(img)/255 した直後
        ```
        array([[[0.7176471 , 0.3647059 , 0.4117647 ],
                [0.7058824 , 0.3882353 , 0.4117647 ],
                [0.69803923, 0.38431373, 0.39607844],
                ...,
                [0.54901963, 0.14117648, 0.18431373],
                [0.5568628 , 0.16470589, 0.20392157],
                [0.53333336, 0.16078432, 0.19215687]],
        ```
    * なぜ255で割るのか
        * 各ピクセルの値を0以上1以下に正規化する
    * img_nad = img_nad[None, ...]
        * img を4次元（tensor）に変換
        * [None,]は np.newaxis と等価。省略記号（Ellipsis）で残りの次元を省略している
        * img_nad = img_nad[None, ...]した直後

        ```
        array([[[[0.7176471 , 0.3647059 , 0.4117647 ],
         [0.7058824 , 0.3882353 , 0.4117647 ],
         [0.69803923, 0.38431373, 0.39607844],
         ...,
         [0.54901963, 0.14117648, 0.18431373],
         [0.5568628 , 0.16470589, 0.20392157],
         [0.53333336, 0.16078432, 0.19215687]],
        ```
        * 次元が一つ増えている

## modelの保存と読み込み
    
```python
from tensorflow.python.keras.models import load_model
```

```python
model.fit(train_images, train_labels, epochs=3)
model.save('fashion_mnist.h5')
```

保存はmodel.fit()のあとに、model.save()をファイル名付きで呼び出すだけ

```python
model = load_model('fashion_mnist.h5')
model.summary()
```

読み出しは load_model() でファイル名を指定。
model.summary でどのようなモデルかを表示してくれる

## modelの可視化

* modelのビジュアル可視化。要install and PATH setting GraphViz, pydot
    * functional API を利用した複雑なモデルの時はこちら

```python
plot_model(model, to_file='./model.png', show_shapes=True, expand_nested=True)
```

* modelのテキスト可視化。Sequentialモデルならこれでも十分、らしい

```python
print(model.summary())
```

## 訓練と評価の履歴をグラフで出す

* model.fit() の引数で validation_data=(X_test, Y_test) などとして、評価用データセットを与えておくと、result.history[] が取れる

```python
#モデルの訓練
epochs_num=30
result = model.fit(X_train, Y_train, epochs=epochs_num, validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt
 
plt.plot(range(1, epochs_num+1), result.history['acc'], label="training")
plt.plot(range(1, epochs_num+1), result.history['val_acc'], label="validation")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```