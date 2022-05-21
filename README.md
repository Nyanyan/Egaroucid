# Egaroucid5
Strong Othello AI Application.

Light version of this othello AI got 1st place in the world ([CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1/leaderboard))

**You can use the [application in Japanese or English](https://www.egaroucid-app.nyanyan.dev/) or use Python [tkinter version](#tkinter_version)**

**You can [play light version of this AI on the Web in Japanese](https://www.egaroucid.nyanyan.dev/).**



<a id="application_version"></a>

## Application

Please see: https://www.egaroucid-app.nyanyan.dev/



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
* Boost
  * https://www.boost.org/





## How to use with tkinter

First, you have to clone this repository. For example,

```
$ git clone git@github.com:Nyanyan/Egaroucid5.git
```

Then move to the ```src``` directory

```
$ cd Egaroucid5/src/test
```

Compile ```egaroucid5.cpp```

```
$ g++ -O3 -fexcess-precision=fast -funroll-loops -flto -march=native -lpthread -Wall ai.cpp -o a.exe
```

Execute ```play.py```

```
$ python3 play.py
```

Then choose which color AI play. 0 for black, 1 for white

```
AI moves (0: black 1: white): 
```

Press ```Start``` button to play!

