# Egaroucid5
Strong Othello AI Application based on Egaroucid4, which got 1st place in the world    

**You can use the [application in Japanese](#application_version) or use Python [tkinter version](#tkinter_version)**

**You can [play light version of this AI on the Web](https://www.egaroucid.nyanyan.dev/).**



<a id="application_version"></a>

## アプリケーション

特設サイトをご覧ください。

https://www.egaroucid-app.nyanyan.dev/



<a id="tkinter_version"></a>

## Abstract

Egaroucid5 is an Othello AI.

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

* CTPL thread pool library
  * https://github.com/vit-vit/CTPL
* RapidJSON
  * https://rapidjson.org/




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
$ g++ egaroucid5.cpp -O3 -fexcess-precision=fast -funroll-loops -flto -mtune=native -lpthread -Wall -o egaroucid5.out
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

