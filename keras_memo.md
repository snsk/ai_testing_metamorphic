# keras memo

## 基本

* https://www.tensorflow.org/tutorials/keras/classification?hl=ja
* TensorFlowのモデルを構築し訓練するためのハイレベルのAPIである tf.kerasを使用
* kerasはデータセットの取得にも対応している
    * https://keras.io/ja/datasets/

* ニューラルネットワークを構築するには、まずモデルの階層を定義し、その後モデルをコンパイルする
    * チュートリアルで利用中間レイヤは Dence（全結合ニューラルネットワークレイヤ）と ReLU
    * そのあと回答を出す出力レイヤは、DenceとSoftMax

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