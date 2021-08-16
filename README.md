# ai_testing_metamorphic

メタモルフィックテスティングの練習

## お題

* ラーメンと冷やし中華を識別するCNNモデルを作成する
    * このモデルに対してメタモルフィック関係を設定する
        * 画像に回転やノイズを足して X→Y, X'→Y となることを確認する

## 方針

* 冷やし中華の画像に対して
    * 画像を10度*36回転させて判定を確認する
    * 画像を右にNピクセル移動させて判定を確認する
* 「それぞれの画像変換（X→X'）に対して判定や正確度が変わらない」ことをメタモルフィック関係と定義する
* 元の画像の正確度は99.9996
* 有意な正確度の低下や、ラーメンとの誤判定があれば「失敗」とする

* 実装の様子（＋各種準備）と結果は以下のファイルで確認できる
    * metamorphic_testing.ipynb

## 実装

* ラーメンと冷やし中華をある程度高精度に識別するモデルを作る
    * ramen_hiyashi_acc0.9675.h5（テストデータの正確度が96.75%）
    * モデルの実装と訓練のコードは metamorphic_testing.ipynb の "2種類の画像を識別するモデル" を参照
* opencvで、上記モデルにおいて99.99％冷やし中華と判断されるデータを加工する
    * 回転の場合は10度刻みで360度＝36ファイル
    * 並進は10px刻みで1000pxまで右に=100ファイル
        * kerasにも画像を加工する機能はあるが、ランダムに加工されるので使いにくい
* load_modelであらかじめ作っておいたモデルを読み込む
    * 毎回の識別の結果を、パラメータと一緒にplot用の配列に入れておく
* 結果配列をplotしてテスト結果をレポートする


## 構成

* Google Colaboratory
    * python 3.7.11
    * tensorflow 2.5.0
    * keras 2.5.0

## note

* 画像データは crawler.py で収集
