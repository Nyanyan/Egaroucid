## How to use Egaroucid

[日本語](https://www.egaroucid-app.nyanyan.dev/usage/ja/)

If you have some requests, please let me know through my Twitter ([@Nyanyan_Cube](https://twitter.com/Nyanyan_Cube) or [@takuto_yamana](https://twitter.com/takuto_yamana)), GitHub issue, or pull requests.



**Functions below are in Study mode. Some functions are disabled in other modes.**



### Index

* [Directory](#folder)
* [Lookahead](#lookahead)
* [Evaluation Values](#hint)
* [Analyze](#analyze)
* [History](#history)
* [Input & Output](#inout)
* [Book](#book)
* [Joseki](#joseki)
* [Use AI without GUI](#engine)





<a id="folder"></a>

### Directory

Check if the all things are in these directories before you start the application.

```
Egaroucid5.exe
resources directory
	book.egbk
	eval.egev
	joseki.txt
	settings.txt
	img directory
		checked.png
		icon.png
		logo.png
	languages directory
		english.json
		japanese.json
		languages.json
		languages.txt
records directory
```

Execute ```Egaroucid5.exe``` to start the application.



<a id="lookahead"></a>

### Lookahead

Lookahead depth is defined by the "Levels". These levels are defined as this image.

![lookahead](https://raw.githubusercontent.com/Nyanyan/Egaroucid5/main/img/lookahead_en.png)

You have to be careful when raising the level, because some high levels are too heavy to use in your computer.



<a id="hint"></a>

### Evaluation Values

Evaluation values are shown in each legal cells.

If the board is in your book, ```book``` will be shown. If not, the AI will search at the ```Hint Level``` and show the the level or ```100%``` when exact search is done. The best moves are shown in blue letters.

#### Human Sense Values

This application has another evaluation values. Human Sense Values are defined as putting together these values:

* Prediction of disc difference
* Each player's bifurcation

The idea came from (in Japanese): https://othelloq.com/tweet/quantifing-human-difficulty

Checking```Human Sense Value``` to enable this function. The values are shown in upper-right corner. The greater the value is, the better the move is.

Human Sense Values are calculated as:

1. Calculate the all recent N moves bifurcation
2. Calculate the disc difference prediction in each end of bifurcation
3. Calculate the number of good and bad bifurcations

#### Umigame's values

Umigame's values are the numbers of the bifurcations of the best moves to memorize in your book.

You can see this value with checking ```Umigame's value```.

Black letters show black player's number, White letters show white player's number.

Umigame's values are announced here in Japanese: http://blog.livedoor.jp/umigame_oth/archives/1075469317.html



<a id="analyze"></a>

### Analyze

Clicking ```Analyze``` button, the evaluation values are re-calculated and plotted on the graph in ```AI Level```.



<a id="history"></a>

### History

You can move across the boards with dragging the graph or pushing right/left or A/D keys. You can play another line with putting a stone.



<a id="inout"></a>

### Input & Output

#### Inporting a Record

You can add a record in your clipboard. The record must be written in F5D6 or f5d6 format.

If importing succeed, you can see the board in your app.

#### Inporting a Board

You can add a board in your clipboard. The board must be written in one line with these characters.

* Black Stones: ```0/B/b/X/x/*```
* White Stones: ```1/W/w/O/o```
* Empty Squares: ```./-```

The color to play must be put in the 65th character.

The sample input is:

```
...........................10......100.....1....................0
```

If importing succeed, you can see the board in your app.

#### Save Game

You can save the game with clicking ```Save Game``` button. You can add white/black player names and memo.

A text file will be created in ```records``` directory.

The filename is:

```
yyyy_mm_dd_hh_mm_ss.txt

yyyy: Year
mm: Month
dd: Day
hh: Hour
mm: Minute
ss: Second
```

The contents: 

```
f5d6 record
Black's Score
yyyy_mm_dd_hh_mm_ss
AI Level
Playing Mode(0:AI plays second 1:AI plays second 2:AI vs AI 3:Human vs Human)
Black player's name
White player's name
Memo
```

If the game is not finished, the score will be ```?```.



#### Copy Record

You can copy the record of the board shown with clicking ```Copy Record``` button.



<a id="book"></a>

### Book

#### How Book is used

If AI found at least one move of the legal moves, the AI do not search and put a stone from registered moves. That means that if you register bad moves only in your book, the AI will put on bad squares. You must register better moves first.

#### Book Error

When some legal moves from a board are registered in your book with different evaluation values, AI puts a disc with book error.

Let X as the value of the best move in the legal positions and let Y as the book error, the policy is selected from legal moves in book which has (X-Y) or higher values.

If the registered moves are +1, -2, -10 respectively and book error is 0, to 2, The +1 move only selected. If book error is 3 to 10, one move is selected from +1 or -2 moves randomly. if book error is over 10, a move is selected from all moves.

#### Modifying and Register a Value to Book manually

Set any board and right click the legal cell. You can now set a new value with keys number keys, ten keys, and ```-``` key. You see what you typed on lower left of the app.

If you right click a legal cell by mistake, do nothing and click the cell again. You can escape.

When closing the app, the book is automatically saved.

#### Register Values to Book Automatically

First, you have to set a board which will be the root of a book.

Then you have to set  ```AI Level``` now because this cannot be changed after starting learning the book.

Also You can set the depth of the book and disc difference acceptance now.

Then press the ```Start Learning``` to start.

You can stop it anytime by ```Stop Learning```.

##### How books are created

Books are created with the algorithm below.

The algorithm uses priority queue.

1. Push the root board to the queue.
2. Loop until the queue is empty.
3. Pop a better and early board from the queue.
4. Do these for all legal moves
5. Put a disc and calculate a score, then push it to the book. If the value may be less than the acceptance, the value is not calculated.
6. push the board to the queue

#### Import (Join) a Book

Any book created with this application or created by Edax can be imported on this app.

Press ```Import``` button in the ```Book``` tab, then you can import it with drag-and-drop.

#### Modify the book automatically

Set any board as the root of the modification, then press ```Auto Modification``` to modify the values after the board.

You can see that the latter value has better accuracy than the early one. This function uses latter values to modify the early values.

Best moves must be in your book. If not, the values after modification will be different from what you expected.

#### Default Book

This application has a default book. This book is created as:

* Modified Zebra's book
* AI Level 21
* Modified with a Edax's book [here](http://infinitelyinfinite.blog.fc2.com/blog-entry-64.html).
* I modified manually

I got a permission to use Zebra's book.



<a id="joseki"></a>

### Joseki

Josekis are shown in Japanese. If you created a Joseki data, then I'm very pleased if you send me to pack it in the downloaded zip file.

I used the joseki data here: http://evaccaneer.livedoor.blog/archives/11101657.html



<a id="engine"></a>

### Use AI without GUI

This Application is released under GPL3.0 license. you can use freely under this license.

[Here](https://github.com/Nyanyan/Egaroucid5) you can see codes. [Here](https://github.com/Nyanyan/Egaroucid5/blob/main/src/egaroucid5.cpp) is a code with which you can use AI without GUI. Please compile it to use it.

Inputting format is:

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
{Row of the next move (0-7)} {Column of the next move (0-7)} {evaluation value}
```

