# Egaroucid5
Strong Othello AI based on Egaroucid4, which got 1st place in the world

**You can use the application in Japanese or use Python tkinter version**



## アプリの使い方

**The main application is only in Japanese.**

### book

#### bookの手動修正・追加

修正したい局面にセットして、修正したいマスを右クリックします。そうするとbook修正・追加モードに入ります。

数字キーまたはテンキー(テンキーは動作未確認)とマイナス符号```-```と小数点```.```とバックスペースキーを使って修正した評価値を入力してください。このとき、小数点を使って実数を登録できます。同じマスを再び右クリックすると新しい評価値がbookに登録されます。

実数に変換できなかった場合は「形式エラー」と表示されます。

誤ってbook修正・追加モードに入ってしまった場合は何も入力せずに同じマスを右クリックすると抜けられます。

誤ってbookを登録してしまった場合は手動で戻すか、resourcesフォルダの```book_backup.txt```を```book.txt```にリネームしてください。



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

