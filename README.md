# Egaroucid5
Strong Othello AI based on Egaroucid4, which got 1st place in the world

**You can use the application in Japanese or use Python tkinter version**



## アプリの使い方

**The main application is only in Japanese.**

### 対局

#### 基本操作

最初に手番を```人間先手/人間後手/人間同士/AI同士```から選びます。この項目は対局中に変更ができません。

ヒント表示や評価値表示、読み手数、book誤差は対局中に変更できます。

設定が完了したら```対局開始```ボタンを押すと対局が開始されます。



#### 読み手数

読み手数は```中盤N手読み/終盤N空読み```のスライドバーで設定できます。

読み手数には、中盤の読み手数と終盤に完全読みする空きマス数の2種類あります。

##### 中盤読み手数

中盤の読み手数はその名前の通り、中盤に何手読むかを決める値です。

前向きな枝刈り(Multi Prob Cut)の確証(どれくらいの確証で読みを早く打ち切るか)は読み手数によって以下のように決まります。

| 読み手数 | 確証   |
| -------- | ------ |
| [1, 10]  | 100%   |
| [11, 14] | 95.50% |
| [15, 16] | 93.30% |
| [17, 18] | 90.30% |
| [19, 20] | 84.10% |
| [21, 22] | 75.80% |
| [23, 24] | 65.50% |
| [25, 60] | 61.70% |

##### 終盤空読み

終盤に完全読みに入る残り空きマス数を決める値です。

前向きな枝刈り(Multi Prob Cut)の確証(どれくらいの確証で読みを早く打ち切るか)は空きマス数によって以下のように決まります。

| 読み手数 | 確証   |
| -------- | ------ |
| [1, 18]  | 100%   |
| [19, 20] | 99.20% |
| [21, 22] | 97.70% |
| [23, 24] | 95.50% |
| [25, 26] | 93.30% |
| [27, 28] | 90.30% |
| [29, 30] | 86.40% |
| [31, 32] | 81.60% |
| [33, 60] | 72.60% |



#### ヒント表示

ヒント表示は局面に存在するすべての合法手に対して行われます。

book登録局面の場合は以下のように評価値と```book```という表示がされます。bookに登録されていない局面はその場で何手か読んで、評価値と読み手数を表示します。ヒント表示の読み手数は```ヒント中盤N手読み/ヒント終盤N空読み```のスライドバーで設定できます。読み手数の内部仕様は上記と同じです。



#### book誤差







### book

#### bookの手動修正・追加

修正したい局面にセットして、修正したいマスを右クリックします。そうするとbook修正・追加モードに入ります。

数字キーまたはテンキー(テンキーは動作未確認)とマイナス符号```-```とバックスペースキーを使って修正した評価値を入力してください。同じマスを再び右クリックすると新しい評価値がbookに登録されます。

入力された文字列が整数に変換できなかった場合は「形式エラー」と表示されます。

誤ってbook修正・追加モードに入ってしまった場合は何も入力せずに(入力欄が空の状態で)同じマスを右クリックすると抜けられます。

誤ってbookを登録してしまった場合は手動で戻すか、resourcesフォルダの```book_backup.txt```を```book.txt```にリネームしてください。

#### bookの自動追加

**外部ツールを使う予定。後で作る**



### 定石

定石データはこちらのものを使っています: http://evaccaneer.livedoor.blog/archives/11101657.html



## Abstract

Egaroucid5 is an Othello AI.

**You can [play light version of this AI on the Web](https://www.egaroucid.nyanyan.dev/).**

There are former versions:

https://github.com/Nyanyan/Reversi

https://github.com/Nyanyan/Egaroucid

https://github.com/Nyanyan/Egaroucid3

https://github.com/Nyanyan/Egaroucid4



## Requirements

### Languages

* Python3
* C++

### Additional Python libraries

* subprocess
* tkinter

### Additional C++ libraries

None



## How to use with tkinter

First, you have to clone this repository. For example,

```
$ git clone git@github.com:Nyanyan/Egaroucid5.git
```

Then move to the ```src``` directory

```
$ cd Egaroucid5/src
```

Compile ```egaroucid5.cpp```

```
$ g++ egaroucid5.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -lpthread -Wall -o egaroucid5.out
```

Execute ```main.py```

```
$ python3 main.py
```

Then choose which color AI play. 0 for black, 1 for white

```
AI moves (0: black 1: white): 
```

Press ```Start``` button to play!

