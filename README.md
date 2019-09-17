# care-label-app
画像認識による洗濯表示の識別

## Description
画像から洗濯表示を識別し、意味を表示する

>洗濯表示とは？  
衣服についたタグのマークのこと。  
洗濯や乾燥の方法、アイロンのかけ方やクリーニングの方法などが示されている。  
全41種類。世界共通。

### WEBアプリケーション
- TOP
- 予測結果表示

## Features
- 洗濯表示の識別
- 意味出力

## Requirement
画像の水増し python 3.6~（f文字列）

WERアプリ
[requirements.txt](/requirements.txt)

## Usage
### 画像の水増し
1枚から約1500枚に水増し。明るさ、ノイズ、回転、移動（上下左右斜め）の加工を行う。

1. [pre-all/train-seeds](/pre-all/train-seeds/)下の該当するフォルダに画像をおく
2. [all-ver2.ipynb](/all-ver2.ipynb)を実行する  
(windowsの場合は、[all-ver2-win.ipynb](all-ver2-win.ipynb))
3. [pre-all/train](/pre-all/train/)下の該当するフォルダに水増しされた画像が出力される


上記の方法は、一括で水増しする場合。
1枚1枚指定して水増しする場合は、[all-file.ipynb](/all-file.ipynb)を実行する。

### 学習
[predict.py](/predict.py)を実行する。

モデル、重み、学習履歴、学習過程の図が保存される。

### WEBアプリケーション
[app.py](/app.py)を実行する。  
localhost:5000から確認できる。  
