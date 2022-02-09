# Egaroucid5
Strong Othello AI Application.

Light version of this othello AI got 1st place in the world ([CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1/leaderboard))

**You can use the [application in Japanese or English](#application_version) or use Python [tkinter version](#tkinter_version)**

**You can [play light version of this AI on the Web in Japanese](https://www.egaroucid.nyanyan.dev/).**



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

If you don't have boost library, compilation will finish with some errors. Then please 1. install boost libraries or 2. use STL version of thread pool.

If you want to use STL thread pool, Please edit thread_pool.hpp like:

```
// from https://github.com/vit-vit/CTPL
#include "CTPL/ctpl.h"
// #include "CTPL/ctpl_stl.h" // Please use this if you don't have boost
```

to

```
// from https://github.com/vit-vit/CTPL
// #include "CTPL/ctpl.h"
#include "CTPL/ctpl_stl.h" // Please use this if you don't have boost
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

