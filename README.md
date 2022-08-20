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
$ cd Egaroucid5/src
```

Edit ```setting.hpp```

```
#define USE_BUILTIN_POPCOUNT true
```

to

```
#define USE_BUILTIN_POPCOUNT false
```

Compile ```ai.cpp```

```
$ cd test
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



## Egaroucid5 on Python

First, you have to clone this repository. For example,

```
$ git clone git@github.com:Nyanyan/Egaroucid5.git
```

Then move to the ```src``` directory

```
$ cd Egaroucid5/src
```

Modify ```Python.h``` location

```
// egaroucid5module.cpp

// INCLUDE MUST BE MODIFIED
#include <Python.h> // < this should be modified
// example
//#include "C:/Users/username/AppData/Local/Programs/Python/Python39/include/Python.h"
```

Setup Egaroucid5

```
$ python setup.py install
```

Execute example ```python_egaroucid5.py```

```
$ python python_egaroucid5.py
```

### Usage

#### egaroucid5.init(eval_file, book_file)

Returns True if initialized, False if failed.

example:

```
egaroucid5.init('test/resources/eval.egev', 'test/resources/book.egbk')
```

#### egaroucid5.ai(board, level)

```0/B/b/X/x/*``` for black, ```1/W/w/O/o``` for white, ```./-``` for empty.

board format:

```
[board as 64 characters] [player]
```

Spaces will be ignored.

Returns tuple ```(Score as int, coord as str)```

example (FFO endgame test #40):

```
egaroucid5.ai('1..11110.1111110110011101101110011111100...11110....1..0........0', 21)
```

