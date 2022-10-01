# Egaroucid
Strong Othello AI Application.

Light version of this othello AI got 1st place in the world ([CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1/leaderboard))

**You can use the [application in Japanese or English](https://www.egaroucid-app.nyanyan.dev/)**

**You can [play light version of this AI on the Web in Japanese](https://www.egaroucid.nyanyan.dev/).**



<a id="application_version"></a>

## Application

Please see: https://www.egaroucid-app.nyanyan.dev/



<a id="tkinter_version"></a>

## Abstract

Egaroucid is an Othello AI.

There are former versions:

https://github.com/Nyanyan/Reversi

https://github.com/Nyanyan/Egaroucid_early

https://github.com/Nyanyan/Egaroucid3

https://github.com/Nyanyan/Egaroucid4



## Requirements

### Languages

* Python3
* C++

### Additional Python libraries

* subprocess

### Additional C++ libraries

* None



## How to use with console

First, you have to clone this repository. For example,

```
$ git clone git@github.com:Nyanyan/Egaroucid.git
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
$ g++ -O3 -fexcess-precision=fast -funroll-loops -flto -march=native -lpthread -Wall Egaroucid6_test.cpp -o egaroucid.exe
```

Execute ```egaroucid.exe```

```
$ egaroucid.exe
```

You should input:

```
Player number(0: Black 1: White)
Board Row 1
Board Row 2
Board Row 3
Board Row 4
Board Row 5
Board Row 6
Board Row 7
Board Row 8
```

For example,

```
0
........
........
........
...10...
...100..
...1....
........
........
```

Outputting format is:

```
{evaluation value} {coordinate}
```

For example,

```
0 c3
```

