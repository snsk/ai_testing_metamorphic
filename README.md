# ai_testing_metamorphic

メタモルフィックテスティングの練習

## お題

* ラーメンと冷やし中華を識別するCNNモデルを作成する
    * このモデルに対してメタモルフィック関係を設定する
        * 画像に回転やノイズを足して X→Y, X'→Y となることを確認する

## 実装

* 1枚の画像を回転させて判定を確認する
    * metamorphic_testing_onefile.py

## 構成

* Google Colaboratory
    * python 3.7.11
    * tensorflow 2.5.0
    * keras 2.5.0

## note

* 画像データは crawler.py で収集
