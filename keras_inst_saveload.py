import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model

# ヘルパーライブラリのインポート
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = load_model('fashion_mnist.h5')
model.summary()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 予測する
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
