This page is a machine translation of the Japanese version. Please refer to the Japanese version for the official content.

# Egaroucid Technical Explanation

<b>This page is only in Japanese. Translation by Google Translator is [here](https://www-egaroucid-nyanyan-dev.translate.goog/ja/technology/explanation/?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=ja&_x_tr_pto=wapp).</b>

**The content written here is not for beginners. Beginner-friendly articles on Othello AI are written [here](https://note.com/nyanyan_cubetech/m/m54104c8d2f12).**

This page contains various technical documents related to Othello AI. I will add to it slowly when I feel like it. The order of the chapters has no particular meaning. Please read only the parts you are interested in!

Last updated: After 2025/11/29, listed at the end of each section

INSERT_TABLE_OF_CONTENTS_HERE



## Board Implementation

Regardless of the method used to create an Othello AI, board implementation (Othello rule implementation) is necessary. Here, I will describe various tips, mainly focusing on the bitboard used in Egaroucid.

### Bitboard

The most recommended board implementation method for Othello AI is the bitboard. Egaroucid also uses a bitboard. If written well, a bitboard can execute Othello rules (legal move generation, stone flipping calculation) incredibly fast, and also reduce memory usage.

In a bitboard, to handle an Othello board of 64 squares (8x8), two 64-bit integers are used: one 64-bit integer to represent the presence or absence of your own stones, and another 64-bit integer to represent the presence or absence of opponent's stones. With this, the Othello board can be represented with just 128 bits (two 64-bit integers). This significantly reduces memory usage.

Low memory usage itself can lead to speed improvements by increasing cache hit rates, but bitboards offer the advantage of simply increasing calculation speed, even more than memory efficiency. Computers inherently store information as bit strings, so they are very good at bitwise operations, meaning they can perform bitwise operations at lightning speed. And by using bitboards, Othello rules can be described with bitwise operations. Furthermore, since a 64-bit integer can represent 64 squares, multiple squares can be affected by a single operation. This means that Othello rules can be implemented with fewer instructions. Since fast bitwise operations are used with a small number of instructions, it becomes incredibly fast.

### Speedup with SIMD

Bitboards are very fast on their own, but further speedup is possible with SIMD (Single Instruction, Multiple Data: performing one instruction on multiple data simultaneously). The appeal of bitboards is that they are very compatible with SIMD due to their heavy use of bitwise operations. Egaroucid has received technical contributions from various people (GitHub pull requests) and has achieved maximum speedup through SIMDization. However, since SIMD (especially the AVX2 instruction set) cannot be used on some CPUs (those before 2013), Egaroucid also provides a non-SIMD version as a Generic version.

SIMD processes bit strings of 128 bits, 256 bits, and even 512 bits at the CPU instruction level. If used effectively, for example, four 64-bit integer calculations (total of 4 operations) can be replaced by one 256-bit calculation. This allows for significant speedup. In Egaroucid, the SIMD version is about 1.5 times faster than the non-SIMD version.

For speedup with SIMD, [Mr. Okuhara's explanation of bitboard speedup using AVX in Edax](http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm) is very helpful.

I would like to explain SIMD on this site someday, but it has been put off due to my lack of technical skill and the excellent explanation by Mr. Okuhara.

### Utilizing Perft

Boards can be implemented in various ways other than bitboards, but regardless of the method, it is necessary to verify that there are no implementation errors. Also, while it is good for board processing to be fast, it is necessary to firmly confirm how much performance improvement has been achieved before and after changes.

For such debugging and verification purposes, a method called Perft is convenient. Perft is a method that expands the entire game tree to a predetermined depth and counts the number of leaf nodes. Since it counts the number of leaves, if there is a bug in the board implementation, it can be noticed immediately. Also, since game tree expansion involves a large number of legal move generations and stone flipping calculations, the speed of these board operations is conspicuously reflected in the execution time. Perft stands for "PERformance Test, move path enumeration" (according to Chess Programming Wiki). The implementation method is [described in detail on the Chess Programming Wiki](https://www.chessprogramming.org/Perft).

In Othello, the game ends after 9 moves. If the game ends, that node is a leaf of the game tree, so it is counted as one leaf even if it has not reached the specified depth.

The Perft values for Othello (number of leaf nodes in the game tree for a given depth) are as follows. Two types of values are listed: Mode 1 is general Perft, and Mode 2 is Egaroucid's unique Perft. Mode 1 counts passes as one move, while Mode 2 counts passes as zero moves. In Othello, one square is filled with one move, so in AI implementation, passes are often counted as zero moves. Therefore, Egaroucid has prepared this as Mode 2. The values below were measured with Egaroucid, but Mode 1 values are also described [here](https://aartbik.blogspot.com/2009/02/perft-for-reversi.html).

<div class="table_wrapper"><table>
    <tr><th>Depth</th><th>Number of leaves (Mode 1: Pass is 1 move)</th><th>Number of leaves (Mode 2: Pass is 0 moves)</th></tr>
    <tr><td>1</td><td>4</td><td>4</td></tr>
    <tr><td>2</td><td>12</td><td>12</td></tr>
    <tr><td>3</td><td>56</td><td>56</td></tr>
    <tr><td>4</td><td>244</td><td>244</td></tr>
    <tr><td>5</td><td>1396</td><td>1396</td></tr>
    <tr><td>6</td><td>8200</td><td>8200</td></tr>
    <tr><td>7</td><td>55092</td><td>55092</td></tr>
    <tr><td>8</td><td>390216</td><td>390216</td></tr>
    <tr><td>9</td><td>3005288</td><td>3005320</td></tr>
    <tr><td>10</td><td>24571284</td><td>24571420</td></tr>
    <tr><td>11</td><td>212258800</td><td>212260880</td></tr>
    <tr><td>12</td><td>1939886636</td><td>1939899208</td></tr>
    <tr><td>13</td><td>18429641748</td><td>18429791868</td></tr>
    <tr><td>14</td><td>184042084512</td><td>184043158384</td></tr>
    <tr><td>15</td><td>1891832540064</td><td>1891845643044</td></tr>
</table></div>

Egaroucid has been equipped with the perft function since console version 7.4.0. This is mainly for me, the developer, to easily debug and speed up. Other developers can also use this for debugging purposes or speed comparison.

Last updated: 2025/11/29



## Selection of Search Algorithm (Minimax or MCTS)

For creating game AI, the minimax method has long been famous. Deep Blue, which defeated a human in chess in 1997, and Logistello, which defeated a human in Othello in the same year, both used algorithms derived from the minimax method. However, in the 2000s, MCTS (Monte Carlo Tree Search) developed, and in the 2010s, PV-MCTS (Policy Value MCTS), which further updated MCTS, developed. AlphaGo, which defeated a human in Go in 2016, is PV-MCTS.

These two algorithms are fundamentally different. For example, in Shogi AI, both minimax and MCTS systems seem to have their merits, but which one is better to use in Othello? It is probably not easy to draw a simple conclusion on which is better for Othello, the minimax system or the MCTS system. However, Egaroucid uses the Negascout method, which is a minimax system.

Here, I will describe the discussion of whether to adopt the minimax system or the MCTS system in Othello, including the background of why Egaroucid adopted the minimax system.

### Accuracy of Evaluation Function

In minimax-based algorithms, if the game cannot be read to the end due to computational complexity, an evaluation function is used before the end, and it is used as if it were the confirmed value read to the end. Therefore, if the accuracy of the evaluation function is poor, the AI will inevitably be weak.

MCTS was originally developed for Go. In Go, minimax-based algorithms were difficult to use. The reason for this was the difficulty in creating an evaluation function for Go. In Shogi, the NNUE evaluation function seems to be strong, and the combination of NNUE + minimax system and PV-MCTS seems to be in a close contest.

In Othello, after the pattern-based evaluation function was proposed in papers on Logistello [(Buro 1997)](https://skatgame.net/mburo/ps/improve.pdf), [(Buro 1999)](https://skatgame.net/mburo/ps/pattern.pdf), it became possible to create highly accurate evaluation functions (relatively) easily. Therefore, the accuracy of the evaluation function is not much of a problem in Othello. In this respect, the minimax system is suitable for Othello.

References

- (Buro 1997): Michael Buro: [Experiments with Multi-ProbCut and a new high-quality eval-uation function for Othello](https://skatgame.net/mburo/ps/improve.pdf), NECI Tech. Rep. 96 (1997)
- (Buro 1999): Michael Buro: [From simple features to sophisticated evaluation functions](https://doi.org/10.1007/3-540-48957-6_8), in Proc. 1st Int. Comput. Games (Lecture Notes in Computer Science), 1999, pp. 126–145, [PDF](https://skatgame.net/mburo/ps/pattern.pdf)

Last updated: 2025/11/29

### Number of Legal Moves

MCTS has other advantages besides not requiring an evaluation function. It can achieve reasonable performance even in games with many legal moves. MCTS was devised for Go, and indeed, Go has a very large number of legal moves, which makes it unsuitable for minimax-based algorithms.

If the number of legal moves for each position is $b$, and the search depth is $d$, then the computational complexity of the minimax method is $b^d$, and the αβ method, which applies effective pruning to the minimax method, is $\sqrt{b^d}$. In games with many legal moves, the base of the exponential function becomes large, which tends to cause a computational explosion. This problem is very significant in games like Go, which is why minimax-based systems are not suitable. However, for Othello, the number of legal moves is relatively small, around 10 throughout the game, and in this respect, Othello is also suitable for minimax-based systems.

### Games that Maximize Score

Unlike Go and Shogi, Othello is a game where the task is to maximize the difference in the number of stones with the opponent at the end of the game (final stone difference) in order to determine victory or defeat. In fact, in human Othello tournaments, if there are multiple people with the same win/loss record, the person with the larger total stone difference (the person who won many games by a large margin) will generally be ranked higher.

Empirically, MCTS-based systems are said to be weak at games that maximize scores. Therefore, in Othello, the minimax system is considered better in this respect.

However, in 2024, a method for handling games that maximize scores with MCTS was proposed [(Mikoshima, Sakaji, Noda 2024)](http://id.nii.ac.jp/1001/00232803/), and I received the impression that this method is very well-designed. Perhaps the problem of maximizing scores can be resolved with MCTS-side ingenuity.

References

- (Mikoshima, Sakaji, Noda 2024): Kazuya Mikoshima, Taiki Sakaji, Itsuki Noda: [Proposal of a Score Distribution Prediction Model in Deep Reinforcement Learning Using Self-Play](http://id.nii.ac.jp/1001/00232803/), IEICE Technical Report, Vol. 2024-GI-51, No. 29, pp. 1-8

Last updated: 2025/11/29

### Implementing Perfect Reading

In Othello, the final 30 moves (which is more like the middle of the game than the end, as Othello always ends in 60 moves) can be perfectly read by a personal computer in a few seconds to about 10 seconds. Perfect reading, as the name suggests, strictly reads to the end of the game, so there is no room for error (unless there are bugs). Therefore, in Othello especially, how early perfect reading (or reading that excludes obviously bad moves) can be performed directly leads to strength.

Since MCTS-based algorithms are not suitable for strict searches like perfect reading, perfect reading is performed using minimax-based algorithms. Therefore, if an Othello AI is built entirely on minimax-based algorithms, including mid-game search, the implementation will likely be simpler.

### Results of Experiments with Both Algorithms

When participating in the [CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1) AI contest, I tried both minimax and MCTS, but my MCTS implementation didn't get very strong. This is limited to my implementation skills at the time, so I cannot discuss whether MCTS is truly unsuitable for Othello, but at that time, I concluded that I should use minimax for now.

However, there was an AI with a MCTS-like playing style among the top ranks of this contest, so it is more likely that the problem was with my implementation skills rather than MCTS actually being weak.

### Personal Enjoyment

This is simply my personal preference. I personally love code that humans have painstakingly tuned, and I want to carefully examine and tune code myself. Minimax-based algorithms, in that respect, were personally appealing to me because tiny speedups can have a huge impact on overall speed.

When I talk to Shogi AI developers, it seems that minor speedups are more effective for minimax-based Othello AI compared to MCTS-based Shogi AI. As someone who loves minor speedups, I found minimax-based systems more interesting. However, of course, this does not mean that Shogi AI does not need speed, and Shogi AI must also be working hard on minor speedups.



## Evaluation Function Model

The pattern evaluation widely used in existing strong Othello AIs was proposed in Logistello [(Buro 1997)](https://skatgame.net/mburo/ps/improve.pdf), [(Buro 1999)](https://skatgame.net/mburo/ps/pattern.pdf), and has been passed down to today's Edax with almost no change. Egaroucid is also based on this pattern evaluation, but with some added features.

Egaroucid added additional features to the existing method of evaluating stones as patterns, and the sum of all scores was used as the evaluation value. The evaluation function prepared a total of 60 phases, with each move as one phase, and a large amount of training data (1.8 billion positions in total for all phases) was prepared and optimized with Adam.

References

- (Buro 1997): Michael Buro: [Experiments with Multi-ProbCut and a new high-quality eval-uation function for Othello](https://skatgame.net/mburo/ps/improve.pdf), NECI Tech. Rep. 96 (1997)
- (Buro 1999): Michael Buro: [From simple features to sophisticated evaluation functions](https://doi.org/10.1007/3-540-48957-6_8), in Proc. 1st Int. Comput. Games (Lecture Notes in Computer Science), 1999, pp. 126–145, [PDF](https://skatgame.net/mburo/ps/pattern.pdf)

Last updated: 2025/11/29

### Stone Evaluation Patterns

The patterns used in Egaroucid 7.7.0 for the board are as follows.

<div class="centering_box">
    <img class="pic2" src="img/patterns.png">
</div>
In Othello, the number of legal moves can be effectively used to evaluate the board position, so it is good to cover all linear continuous patterns of 3 or more squares so that potential information about the number of legal moves can be obtained from the stone patterns [(Buro 1999)](https://skatgame.net/mburo/ps/pattern.pdf).

By rotating and flipping these patterns, 64 features shown below are generated, and the sum of the scores assigned to these features is used as the evaluation value.

<div class="centering_box">
    <img class="pic2" src="img/features.png">
</div>

The heatmap showing how many features each square belongs to is as follows.

<div class="centering_box">
    <img class="pic2" src="img/heatmap.png">
</div>
In Othello, it is generally said that the shape of the corners and edges strongly affects the board position. Therefore, corner and edge squares are included in many patterns.

References

- (Buro 1999): Michael Buro: [From simple features to sophisticated evaluation functions](https://doi.org/10.1007/3-540-48957-6_8), in Proc. 1st Int. Comput. Games (Lecture Notes in Computer Science), 1999, pp. 126–145, [PDF](https://skatgame.net/mburo/ps/pattern.pdf)

Last updated: 2025/11/29

### Other Features

Egaroucid also uses the number of stones of the current player as an additional feature. Since Egaroucid uses an evaluation function divided into 60 phases based on the total number of stones on the board (both players), using only the number of stones of the current player implicitly includes the number of opponent's stones as a feature.

### Trade-off between Accuracy and Speed

In general, an evaluation function becomes more accurate and stronger by using a complex model. However, complex models often take a long time to compute. In that case, the depth that can be looked ahead per unit time becomes shallower, and the strength per unit time may actually decrease. In particular, αβ search algorithms inherently execute the evaluation function many times, so the speed of the evaluation function strongly affects the overall search speed.

However, it can also be argued that if the evaluation function is accurate, it can perform accurate move ordering, which can reduce the number of visited nodes in αβ search, and consequently speed up the search. Since the game tree grows exponentially with depth, for deep searches, a slightly more complex evaluation function may reduce the number of visited nodes and potentially speed up the entire search.

These two perspectives are ultimately a matter of balance. Egaroucid uses a slightly more complex evaluation function than Edax, and therefore its evaluation function is slower than Edax's. In Egaroucid 6.X.X, in addition to the patterns and features mentioned above, patterns using legal move positions were also used. From Egaroucid 7.X.X, legal move position patterns were excluded from the evaluation function. This means that accuracy was slightly sacrificed to gain speed. However, in Egaroucid 7.X.X, the training data was re-examined, and the strength is comparable to that of Egaroucid 6.X.X's evaluation function.

To consider the "complex evaluation function makes it slow" issue more concretely, random memory access often becomes the bottleneck. To some extent, this is unavoidable. The speed of memory reference changes depending on the number and size of patterns used, so when considering maximizing speed, it is necessary to carefully examine the patterns chosen. That being said, at this point, after trying various things, it hasn't had much effect...

### NNUE Evaluation Function

There is an evaluation function called [NNUE](https://github.com/ynasu87/nnue) that was developed for Shogi AI and later imported into Chess AI. Simply put, NNUE uses a very small neural network as an evaluation function. I experimented a little to see if this evaluation function, which has been successful in Shogi and Chess, could be used in Othello. The conclusion, for now, is that "it's not at all impossible to completely rule it out, but creating a high-performance evaluation function seems quite challenging."

Neural networks tend to give the impression of being slow, and slow speeds might seem unsuitable for algorithms like αβ search. However, NNUE makes the neural network scale very small, and by calculating differences in the input layer for each move and accelerating the hidden layer with SIMD, it achieves sufficiently practical speeds on the CPU. In fact, in Shogi, it is said to be as fast as existing three-piece relationship evaluation functions.

The advantage of NNUE is that it is a nonlinear evaluation function. Pattern evaluation and Shogi's three-piece relationship are both linear evaluation functions, so their expressive power has inherent limitations. However, neural networks do not have this limitation.

I tried to see if a model of this scale could be sufficiently trained for Othello. As an experiment, I tentatively created a model (see figure below) with a total of 128 inputs (i.e., bitboard directly), 64 for the presence or absence of black stones and 64 for the presence or absence of white stones, with two fully connected layers of 32 nodes as hidden layers and ReLU as the activation function, and trained it with sufficient training data.

<div class="centering_box">
    <img class="pic3" src="img/nnue_model.png">
</div>

As a result, the evaluation performance did not reach that of existing pattern evaluations. Specifically, regarding the mean absolute error (MAE) of the evaluation function against the training data, pattern evaluation was about 3.5 stones, while this test model was about 5.4 stones. The advantage of NNUE was the improvement in evaluation performance due to its nonlinearity, so in Othello, I could not feel the superiority of NNUE in this experiment. If NNUE is to be used in Othello, I feel that devising the input layer or increasing the volume of the hidden layer is important for making NNUE practical. However, these solutions may impair inference speed, so it is necessary to consider this while balancing with speed.

By the way, a model of the size shown above seemed indeed to be able to infer quite fast on a CPU. Therefore, I feel that NNUE is worth trying a little harder.



## Training Data for Evaluation Function

Once the evaluation function model is decided, a large amount of data is prepared to optimize (train) the parameters included in that model. Here, we describe the process with an evaluation function that predicts the final stone difference for a given position in mind.

<h3>How to Create Training Data for Mid-game and Beyond</h3>

It is good to generate training data through self-play between Othello AIs. Egaroucid used self-play between past versions of Egaroucid. The AI's strength was set to Level 11 (mid-game 11-ply lookahead, end-game 25-ply perfect reading) considering the balance between calculation speed and accuracy.

There are various ways to create correct labels, but Egaroucid simply used the final stone difference of the game as is for all positions. To improve data accuracy, a method of performing lookahead with Othello AI for all positions included in the training data and using the results could be considered, but this method is very time-consuming. For Egaroucid's Level 11, it is strong enough, and stone losses during self-play are considered acceptable.

In self-play, if battles are only conducted from the first move, data diversity cannot be ensured. Therefore, random moves were made for a fixed number of opening moves, and then self-play was performed from there. To ensure the quality of the training data, positions from the random move part are excluded from the training data. At this time, it is better to prepare many different numbers of random moves, not just one, so that the quality of the training data does not become biased. Egaroucid performed random moves at various ply counts, such as 12, 18, 21, 24, 30, 31, 32, 35, 39, 40, ..., 58.

<h3>How to Create Training Data for the Opening</h3>

Training data generated by random moves + self-play is a method that can achieve quantity, diversity, and accuracy, but it has the weakness that opening training data cannot be created because random moves are made in the opening. Therefore, opening training data is created using a different method.

In the opening, the number of positions that appear is very small, so it is sufficient to enumerate all possible positions in advance, perform lookahead for a fixed number of moves with an existing Othello AI for all enumerated positions, and calculate the evaluation values to use them as correct labels. In Othello, only about 20 million ($2 \times 10^7$) positions appear up to the 11th move, so Egaroucid adopted this method for the first 11 moves. The results of this investigation are written in [Othello Opening Expansion Count](./#ゲームの特性_オセロの序盤の展開数). Please also refer to this.

When actually creating the data, all unique openings up to the 11th move were enumerated, and all of them were evaluated by Egaroucid 7.4.0 at Level 17, and evaluation values were assigned.

<h3>Publicly Available Game Record Data</h3>

Egaroucid does not have problems with training data because it can use the "self-play with the previous version to increase data" method. However, when creating an Othello AI for the first time, it can be difficult to prepare training data. Therefore, I will introduce publicly available game record data that can be used for training and how to convert the data. With these, you should be able to start training immediately without having to create data yourself.

<ul>
    <li>[Egaroucid Training Data](./../train-data/)</li>
    <li>[La base WTHOR (Game records of human matches distributed by the French Othello Federation)](https://www.ffothello.org/informatique/la-base-wthor/)</li>
    <li>[How to Read the Othello Game Record Database WTHOR](https://qiita.com/tanaka-a/items/e21d32d2931a24cfdc97)</li>
</ul>


<h2>Optimization of Evaluation Function</h2>

The optimization (training) of the evaluation function is performed automatically by a computer, not manually. If the number of parameters is sufficiently small, it may be acceptable to do it manually, but even for about 10 parameters, it would be difficult to adjust them manually.

### Gradient Descent

Gradient descent has long been used for optimizing evaluation functions, especially pattern evaluation. This was devised in Logistello [(Buro 1997)](https://skatgame.net/mburo/ps/improve.pdf), [(Buro 1999)](https://skatgame.net/mburo/ps/pattern.pdf). For optimization using gradient descent, the document by the author of Othello AI Thell [(Seal Software 2005)](https://sealsoft.jp/thell/learning.pdf) is detailed.

Egaroucid uses a self-implemented Adam, an improved algorithm of gradient descent, in CUDA, because simple gradient descent converges slowly. Also, since all training data can be loaded into GPU memory (24GB in my environment), batching is not performed.

References

- (Buro 1997): Michael Buro: [Experiments with Multi-ProbCut and a new high-quality eval-uation function for Othello](https://skatgame.net/mburo/ps/improve.pdf), NECI Tech. Rep. 96 (1997)
- (Buro 1999): Michael Buro: [From simple features to sophisticated evaluation functions](https://doi.org/10.1007/3-540-48957-6_8), in Proc. 1st Int. Comput. Games (Lecture Notes in Computer Science), 1999, pp. 126–145, [PDF](https://skatgame.net/mburo/ps/pattern.pdf)
- (Seal Software 2005): Seal Software: [Optimization of Reversi Evaluation Function](https://sealsoft.jp/thell/learning.pdf) (2005)

Last updated: 2025/11/29

<h4>Training Data and Validation Data</h4>

The training data and validation data are randomly separated before training, and their data quality is identical. Basically, they are game data from matches played at Egaroucid Level 11. At that time, some random moves were made in the opening, and then the AI played, creating data variability. The random move part is not included in the training/validation data.

For Phase 11 (up to the 11th move of the opening), it is realistic to enumerate all possible positions and assign correct labels to each, so training data was created by enumerating all opening developments and having Egaroucid calculate their evaluation values. Since these data cover all possible positions in the game of Othello, the same data was used for training and validation without separation.

<h4>Test Data</h4>

The test data is generated by making random moves for the first $N (0 \leq N \leq 12)$ moves, and then playing with Egaroucid Level 30. This means that positions with a series of bad moves after the opening are not included. This is where it differs from the training/validation data. Data before Phase 11 is the same as the training/validation data.

<h4>Egaroucid 7.5.0 Training Results</h4>

The following figures plot the loss of the evaluation function for the training data and test data. The horizontal axis represents the phase (synonymous with the number of moves), and the vertical axis represents the loss (MSE is the mean squared error, MAE is the mean absolute error). Since the teacher data is the final stone difference, the units are stone difference squared and stone difference, respectively.

<div class="centering_box">
    <img class="pic2" src="img/eval_loss_mse.png">
    <img class="pic2" src="img/eval_loss_mae.png">
</div>
The training/validation data varies in quality across phases due to the timing of random moves, so the graphs have some irregular shapes.

Overall, it can be seen that the model was sufficiently trained without overfitting. However, observing that `test_mse` increases after Phase 20 suggests that the evaluation function model may be too small for the complexity of the Othello game beyond Phase 20.

### Deep Learning

Optimization of the evaluation function can also be done using deep learning. This does not mean performing inference during search, but rather performing inference for all possible states when the Othello AI starts up and restoring the evaluation function table. Parameters generated by deep learning can often reduce the amount of data compared to the original evaluation function model, so they can be used for compression. Also, empirically, deep learning tends to create stronger evaluation functions with less data than gradient descent-based methods.

Originally, Egaroucid was developed for the purpose of participating in a programming contest called [CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1). Therefore, to create a strong evaluation function while adhering to the contest's unique restriction of "code must be within 100,000 characters," it was devised as a compression of the evaluation function [(Yamana, Hoshino 2025)](https://doi.org/10.1109/TG.2025.3624825), [(Yamana, Hoshino 2024)](http://doi.org/10.20729/00239899), [(Yamana 2022)](https://ipsj.ixsq.nii.ac.jp/records/218735).

For each pattern appearing in the evaluation function, a neural network model like the one below is constructed. A separate model is prepared for each pattern, and finally, the sum of the output results of all patterns is used as the evaluation value and compared with the correct label. At this time, although there are multiple symmetric features for a single pattern, these features share the same neural network model.

<div class="centering_box">
    <img class="pic3" src="img/nnue_model.png">
</div>

In the current Egaroucid, since we have succeeded in creating a very large amount of training data (over 3.2 billion positions!), we are using Adam instead of deep learning.

My deep learning training is also summarized in the following materials.

References

- (Yamana, Hoshino 2025): Takuto Yamana, and Junichi Hoshino: [Compressing the Evaluation Function With Small-Scale Deep Learning on Othello](https://doi.org/10.1109/TG.2025.3624825), IEEE Transactions on Games (2025)
- (Yamana, Hoshino 2024): Takuto Yamana, Junichi Hoshino: [Development and Evaluation of a Strong Othello AI Using Compression with Small-Scale Deep Learning](https://doi.org/10.20729/00239899), IPSJ Transactions on Computer Games, Vol.65, No.10, pp.1545-1553 (2024)
- (Yamana 2022): Takuto Yamana: [Creation of a Strong Othello AI Using Compression with Deep Learning](https://ipsj.ixsq.nii.ac.jp/records/218735), IEICE Technical Report, Vol. 2022-GI-48, No. 5, pp. 1-5 (2022)
- Takuto Yamana: [Othello AI Textbook 7 [Evaluation] Pattern evaluation, etc.](https://note.com/nyanyan_cubetech/n/nb6067ce73ccd)

Last updated: 2025/11/29



## Fast Execution of Evaluation Function

Once the evaluation function has been trained, it is necessary to create a part that executes the evaluation function during the search. Since the evaluation function is executed many times during the search, speeding up its execution leads to speeding up the entire search. We have already considered a fast evaluation function when designing the evaluation function model itself, but small implementation工夫 (clever tricks) can lead to further speedups, so I will list some tips here. These tips mainly concern pattern evaluation.

### Using Differential Calculation

In pattern evaluation, for all patterns (considering symmetrical shapes), all feature indices are calculated, and then optimized evaluation parameters are looked up using those indices. At this time, the most computationally intensive process is the calculation of feature indices. Therefore, it is good to make this part differential.

If all indices are calculated at the start of the search and only the changed parts are updated each time a move is made, feature calculation becomes fast. When a move is made, for each square where a stone is placed and each stone that is flipped, differential calculation can be implemented by adding or subtracting $3^x$ for the pattern to which that square belongs. At this time, the pattern to which the square belongs and $x$ are constants determined by the square and pattern, so they are pre-calculated.

Now, let's consider the computational cost (calculation amount + estimation of calculation frequency). Let $F$ be the number of patterns (which Egaroucid calls "features" in the code) that are considered identical due to symmetry, $G$ be the average number of features a single square belongs to, $S$ be the average number of squares included in one pattern, and $D$ be the number of stones flipped per move. Then, the cost of calculating all features is $O(FS)$, and the cost of differential calculation per move is $O(GS(D+1))$. In Egaroucid, $F=64$, $G=8.8125$, and $S=8.8125$. Also, in Othello, on average $D=2.24$ (see [Average Number of Flipped Stones](./#ゲームの特性_返る石数の平均値) for details), so the number of calculations when calculating all features is about 564, and the number of calculations for differential calculation is about 252.

In Othello AI, in addition to making moves, undoing moves is also frequently used. Undoing moves can also be implemented with differential calculation, but since Othello ends in at most 60 moves, there is no need to specifically use differential calculation. If 60 moves' worth of pattern evaluation features are prepared in advance and copied while performing differential calculation for each move, then for undoing moves, it is only necessary to refer to the previous calculation results.

### Speedup of Differential Calculation with SIMD

Differential calculation can also be sped up with SIMD. This uses an idea from [Mr. Okuhara, who wrote Edax-AVX](http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm), adapted for Egaroucid.

If differential calculation is done naively, it seems to involve looping through each feature that a changing square belongs to one by one. However, different features can be calculated independently, so this can be parallelized (in a SIMD-like manner).

The indices used for pattern evaluation can be limited to $3^{10} (<2^{16})$ types if the number of squares included in a pattern is limited to 10. Therefore, all indices can be represented with 16 bits. If there are 64 features, then 4 ($=F(=64)\times16/256$) 256-bit SIMD vectors would be sufficient. For these 4 256-bit vectors, information about "how to operate if this square is flipped" is pre-calculated, and differential calculation is performed 4 times (number of vectors) for each flipped stone square.

In this case, the computational complexity of differential calculation (to emphasize the importance of the constant factor) is $O(F\times16/256\times (D+1))$, and the number of calculations is approximately 13. This is a dramatic speedup. Of course, the speed differs depending on the instruction, so it is not possible to simply discuss speed in terms of the number of calculations, but SIMDization of differential calculation is very beneficial.

### Speedup of Table Lookup with SIMD

Now, that differential calculation has been SIMDized, table lookups can also be SIMDized. This is also based on [Mr. Okuhara's idea](http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm).

Here, we use gather instructions. Gather instructions read a SIMD vector in 32-bit chunks, interpret them as indices, and perform memory accesses. When referencing a single array multiple times, if you provide the starting address of the array and a vector containing the indices to be referenced, you can reference them all at once, which is convenient. Although memory access itself is said to be very slow compared to other instructions, in this case, memory access must be performed, so any speedup here is welcome.

Since the indices for differential calculation were implemented as four 256-bit vectors, it is very easy to implement this by directly passing them to the gather instruction. However, gather instructions only have one type that reads 32 bits at a time as an index (`_mm256_i32gather_epi32`). Since 16-bit indices were used in differential calculation, 16 bits $\times$ 16 elements of 256 bits must be temporarily converted into two 256-bit vectors of 32 bits $\times$ 8 elements and then passed to the gather instruction. Furthermore, gather instructions only come in two types: 32-bit memory access or 64-bit memory access. In Egaroucid, since the array to be referenced is represented by 16-bit integers, 32-bit memory access is performed unnecessarily, and the upper 16 bits are discarded (i.e., masked with `0x0000ffff`). This seems wasteful, but there is no other way.

When converting one vector of 16 bits $\times$ 16 elements into two vectors of 32 bits $\times$ 8 elements, it is good to use a union of a 256-bit vector and a 128-bit vector, read 128 bits, interpret them as 8 16-bit numbers, convert each number to 32 bits using the `_mm256_cvtepu16_epi32` instruction, and then create a 256-bit vector.

Regarding the process of performing 32-bit memory access with a gather instruction and then discarding the upper 16 bits, if the upper 16 bits were discarded every time gather is used, the number of instructions would increase. In Egaroucid, the upper limit of the evaluation function parameters (=4092) is set, and (upper limit) $\times$ (number of times gather instruction is used (8 times in Egaroucid)) = 32736 is ensured to fit within 16 bits. This way, the process of discarding the upper 16 bits only needs to be done once at the end.



## Move Ordering

In game tree search algorithms based on the αβ method, if child nodes are arranged in a promising order (order of good moves) at each node and searched in that order, pruning is more likely to occur. This reordering of moves is called Move Ordering.

Here, I will explain the Move Ordering used in Egaroucid. Various features are used, but Egaroucid does not always use all elements; instead, it changes the features used and the importance of each feature depending on the situation.

### Results of Shallow Search (Previous Search)

Based on the idea that moves judged good in a shallow search will also be good moves in a deep search, the results of shallow searches are incorporated into Move Ordering. Also, if the results obtained in previous searches are in the transposition table, they may simply be referenced.

By the way, iterative deepening may be performed in αβ search, and one reason for such an implementation is to keep shallow search results in a transposition table and refer to previous search results as appropriate during Move Ordering to perform efficient searches. Of course, there is also a demand to search as much as possible within the time limit.

### Value of Dedicated Evaluation Function

Egaroucid 7.0.0 and later uses a lightweight evaluation function dedicated to move ordering in the endgame search. The normally used evaluation function is too slow to be used for endgame move ordering, but we still wanted to use an evaluation function for endgame move ordering, so we created a small evaluation function using only some patterns. By reducing the number of patterns, the number of memory accesses was reduced, resulting in very high speed. Also, the accuracy is about 4.2 stones in absolute error for the training data, which is sufficient for move ordering.

The endgame move ordering dedicated evaluation function in Egaroucid 7.4.0 uses the following four patterns. These patterns are part of the patterns used in the regular evaluation function, so they require minimal computational cost. Once the endgame search begins, differential calculation of evaluation values is performed only for these patterns. This calculates the endgame move ordering dedicated evaluation value at about 1/4 the cost of the regular evaluation function.

<div class="centering_box">
    <img class="pic2" src="img/move_ordering_end_pattern.png">
</div>

### Number of Opponent's Legal Moves After Own Move

Prioritize moves that result in fewer legal moves for the opponent after your own move. This can be interpreted in two ways.

First, the game of Othello is a game where you reduce the number of moves your opponent can make (more precisely, the "number of hands" in human Othello, but it is approximated by simply the number of places to play). In other words, moves that reduce the places an opponent can play are often strong moves in Othello itself.

Also, if there are fewer legal moves, the number of child nodes to expand is simply smaller, so it can be expected that continuing such a greedy method will reduce the number of visited nodes required for search. Especially when aiming for fail high in Null Window Search (NWS), if fail high occurs, it is not always necessary to search for the best move, so this Move Ordering is often effective.

As an aside, I think it is a very beautiful story that this simple metric of "opponent's legal moves" is very rational from both the perspective of the essence of the game of Othello and the convenience of game tree search.

This method of searching in ascending order of the number of legal moves is sometimes called "speed-first search."

### Number of Empty Squares Adjacent to Own Stones After a Move (Potential Legal Moves)

After making a move, prioritize moves that result in fewer empty squares adjacent to one's own stones in any of the 8 surrounding directions. This is largely the same meaning as reducing the places the opponent can play.

In Othello, legal moves are always only to empty squares, and the rule is to flip opponent's stones by sandwiching them. Therefore, where the opponent moves is always an empty square, and there is at least one of one's own stones adjacent to that empty square.

Empty squares adjacent to one's own stones may not become legal moves for the opponent immediately, but they may become legal moves for the opponent in the future. Therefore, it is judged better to have fewer such empty squares. This idea is sometimes called "potential legal moves."

In human Othello, the concept of "openness" is (essentially) similar to this potential number of legal moves.

### Parity of Empty Squares When the Board is Divided into 4 (Quasi-Parity Theory)

In Othello, due to the rule of flipping opponent's stones by sandwiching them, stones placed and flipped by the last move of the game are never flipped back. Furthermore, in a similar vein, if the last square is played "locally," the stones placed and flipped by that move are less likely to be flipped back again. Therefore, it is desirable to play first in locally odd-numbered empty squares, and if there are three empty squares, it is desirable to finish that local area with one's own move, such as me -> opponent -> me. In human Othello, this idea is called the parity theory. Due to its characteristics, the parity theory is a very useful tactic in the endgame of Othello.

Quasi-parity theory (which I named arbitrarily) is an approximate implementation of parity theory in Othello. The difficulty in implementing parity theory lies in the concept of "local." When observing game records of strong human Othello players, it is often found that squares near the corners remain empty in the endgame. Based on this fact, quasi-parity theory fixes this "local" area as the board divided into four 4x4 sub-boards. The number of empty squares belonging to each sub-board is counted, and legal moves belonging to odd-numbered empty sub-boards are prioritized for search.

This explanation is my interpretation of "Parity Ordering" originally used in Edax.

### Position of the Square to Play

Just as corners are said to be strong in human Othello, the importance of squares in Othello varies depending on their position. Utilizing this property, for example, efforts are made to prioritize playing in a corner slightly. However, this metric is static and not highly accurate, so personally, I use it with the feeling of "if the scores are tied with other metrics, it's better than choosing randomly."



## Transposition Table

Although the game tree is called a "tree," merges often occur. In other words, it often happens that the same position is reached through different sequences of moves. For example, in Othello, the tiger opening `f5d6c3d3c4` can also be reached from the cat system as `f5d6c4d3c3`. In this case, since the position reached is the same regardless of the move sequence, the result of searching one can be directly reused as the result of searching the other. To perform such processing, past search results are memoized in a table called a transposition table. The transposition table is also useful for move ordering when iterative deepening search is performed (in Othello, I feel this effect is even greater than when a merge occurs).

Transposition tables are implemented as hash tables. However, what is special about game tree search is that it is not always necessary to retain past data. If data is not in the transposition table, it is simply necessary to search again, so implementations that prioritize speeding up access to the transposition table, such as overwriting past data when hash collisions occur, are suitable.

In practice, efforts are made to preferentially keep deep searches because they are difficult to re-search. Also, if a hash collision occurs and an adjacent area in memory (corresponding to a different hash value) is empty, data is stored there, which is a light hash collision countermeasure.

Regarding the problem of which to keep, existing data or new data, when a hash collision occurs, Egaroucid makes a judgment based on the following metrics. Higher items are prioritized.

<ul>
    <li>Number of moves read from the position registered there (deeper reads are harder to search, so they are worth keeping)</li>
    <li>Probability of Multi ProbCut when searching that position (higher MPC probability means harder search and more accurate value, so it's worth keeping)</li>
    <li>Newness of search (results of new searches are preferentially kept)</li>
</ul>
The importance is calculated from these perspectives, and when a hash collision occurs, the one with higher importance is kept. Furthermore, transposition tables are basically not reset and are reused for all searches. This eliminates the process of resetting a huge transposition table. However, if a transposition table is reused for a certain period, it will be filled with old and costly search data. Egaroucid implements a process to set the importance of all data to the minimum value when this happens.

### Multithreading of Transposition Table

Othello AI often parallelizes searches to speed them up. In this case, handling the transposition table becomes more complex than in single-threaded operation. There are two ways to handle the transposition table in multithreaded operation:

<ul>
    <li>Each thread has its own transposition table</li>
    <li>All threads share one transposition table</li>
</ul>

These two methods each have their pros and cons.

In the former, where each thread has its own transposition table, there is no need to temporarily lock the transposition table for data rewriting, which speeds up this process. However, with this method, even if redundant searches can be omitted within the same thread, redundancy tends to occur because the results searched by other threads cannot be referenced. In Egaroucid 7.3.0, each thread has its own transposition table only near the leaf nodes of the tree during endgame search. This is because, near the leaf nodes of the tree, the cost of redundant searches is smaller than the cost of locking. This idea was provided by someone else via a GitHub pull request.

In the latter, where all threads share the transposition table, it is necessary to lock the transposition table each time it is rewritten, but this significantly reduces redundant searches. Especially when parallelizing with many threads, performing efficient searches even if it incurs the cost of locking tends to speed up the search overall. However, near the leaf nodes of the tree, the cost of locking tends to be relatively large, so not using a transposition table near the leaf nodes is faster. When locking the transposition table, if only one element is locked instead of the entire transposition table, other threads trying to access the transposition table (different elements) at the same time will not be blocked, which speeds it up.



## Backward Pruning

Egaroucid prunes searches in various ways. Here, I will introduce backward pruning (those that do not change the search results). Particularly complex pruning methods are based on those implemented in Edax.

### Negascout Method

Egaroucid performs minimax-based search, so it increases pruning using the Negascout method. In addition to αβ search, the Negascout method searches for the best (or seemingly best) move (Principal Variation = PV) with a normal window [α, β], and then performs Null Window Search (NWS) for the remaining moves with [α, α+1]. If this fails high, it searches again with a normal window to update the PV, and if it fails low, it can be left as is.

I wrote in detail about Negascout on [note](https://note.com/nyanyan_cubetech/n/nf810b043fb78).

### Transposition Cutoff

The game tree can mostly be considered a tree structure, but there are sometimes merging lines of play. In such cases, it is wasteful to search the same position again. Therefore, search results are memoized in a transposition table (a special hash table). At each node, before searching, the transposition table is referenced, and if a value is registered, it can be returned immediately. Also, registering lower and upper bounds of evaluation values can be expected to have the effect of narrowing the search window.

Even if a value is registered in the transposition table, if it is the result of a shallower search than the current search, it should not be adopted. Egaroucid records the search depth, Multi ProbCut probability, and search newness in addition to the lower and upper bounds of the evaluation value to decide whether to adopt the transposition table value as the result of the current search.

Note that accessing the transposition table has overhead, so access to the transposition table is performed only at nodes close enough to the root that the overhead is negligible. By doing this only at nodes close to the root, the data to be registered in the transposition table can also be reduced to the bare minimum necessary.

Furthermore, by also recording the best move from past searches in the transposition table, it can also be used for move ordering in the current search.

### Enhanced Transposition Cutoff

When searching a node, even if that node is not registered in the transposition table, expanding its child nodes and referencing the transposition table may narrow the search window or determine the evaluation value. This is an idea implemented in Edax as Enhanced Transposition Cutoff (ETC).

Suppose a node is searched with a search window of [α, β]. Also, child nodes are expanded, and for each child node, the minimum value L and maximum value H (from the perspective of the child node's turn) recorded in the transposition table are examined. Then, if β < max(-{L}), β can be updated to max(-{L}), and similarly, if α > max(-{H}), α can be updated to max(-{H}).

This repeatedly references the transposition table after expanding child nodes, so the overhead is quite large. This is performed only at nodes close to the root.

### Stability Cutoff

For a given board position, if there are B black stable stones and W white stable stones, the final position will be somewhere between B vs 64-B and 64-W vs W. For example, the evaluation value (final stone difference) from black's perspective will be between 2B-64 and 64-2W. This can sometimes be used to narrow the search window. This is an idea implemented in Edax as Stability Cutoff.

This method can only be used when the node being searched is sufficiently close to the end of the game, i.e., when there is a high possibility of stable stones existing. Also, calculating stable stones has overhead, so if it is too close to the end of the game, simply searching may often be faster.

Note that if the number of stable stones is small, the range of the search window that can be narrowed will be close to -64 or +64. Therefore, if the number of stable stones is expected to be small before calculating stable stones, this is done only when the target search window for narrowing includes areas close to -64 or 64. If the number of stable stones is expected to be large, pruning is attempted even when the target search window for narrowing is close to 0. The expected value of the number of stable stones tends to increase as the game progresses, so this expectation uses a correspondence between depth and the upper/lower bounds of the search window.

The calculation of stable stones itself involves:

<ul>
    <li>Best move from past searches (value registered in the transposition table)</li>
    <li>Stable stones on the edges (pre-calculated and determined by referencing the shape of the edges)</li>
    <li>Stones for which all 8 directions (vertical, horizontal, diagonal) are filled with other stones are considered stable stones.</li>
    <li>Stones surrounded by 8 adjacent stable stones are considered stable stones (processed in a loop).</li>
</ul>

Of course, this alone cannot perfectly find all stable stones. However, in the context of Othello AI pruning, this is sufficient, and being too accurate and slow is meaningless.



## Forward Pruning

In contrast to backward pruning, which guarantees not to change search results, forward pruning uses methods such as "this move seems obviously bad, so there's no need to look ahead." Forward pruning may change search results, but it can generate many more prunings than backward pruning and contributes to search speed. Also, the benefit of increased search depth due to improved search speed often outweighs the possibility of changing search results.

### Multi-ProbCut

This is an old technique devised in Logistello [(Buro 1997)](https://skatgame.net/mburo/ps/improve.pdf) and has been adopted by various Othello AIs since Logistello. I plan to add more details about this in the future (2024/06/10).

References

- (Buro 1997): Michael Buro: [Experiments with Multi-ProbCut and a new high-quality eval-uation function for Othello](https://skatgame.net/mburo/ps/improve.pdf), NECI Tech. Rep. 96 (1997)

Last updated: 2025/11/29

## Speedup with Implementation Tricks

The basic flow for searching a certain position is "transposition table pruning -> legal move generation -> enhanced transposition table pruning -> Multi-ProbCut -> move ordering -> search next positions in sorted order," but various omissions are possible through implementation tricks. This idea comes from Mr. eukaryo's article [About Pruning Methods](https://eukaryote.hateblo.jp/entry/2020/04/27/110543), and I will explain the implementation in Egaroucid with some adaptations. In Egaroucid, the process from legal move generation to searching the next position is executed in the following order. With clever implementation ordering, a speedup of several percent can be expected.

<ol>
    <li>Transposition Table Pruning
        <ul>
            <li>If α-cut or β-cut can be performed on the spot, return that value and end the search.</li>
            <li>The transposition table contains the best move from previous searches, so refer to it.</li>
        </ul>
    </li>
    <li>Legal Move Generation
        <ul>
            <li>Enumerate all legal moves.</li>
            <li>Calculate the stones flipped by that move.</li>
            <li>Verify if the opponent can be wiped out (if wiped out, immediately set score to +64 and end search).</li>
        </ul>
    </li>
    <li>Enhanced Transposition Table Pruning
        <ul>
            <li>Expand one move and refer to the transposition table to check for pruning.</li>
            <li>Even if the current search depth is not met, nodes visited in previous searches are often good moves, so memoize them for preferential search.</li>
        </ul>
    </li>
    <li>Multi-ProbCut</li>
    <li>Search Previous Best Move
        <ul>
            <li>The previously best move is likely the best move in the current search as well, so search it before move ordering.</li>
            <li>If a β-cut occurs here, it's okay to end the search immediately. This saves time spent on move ordering, etc.</li>
        </ul>
    </li>
    <li>Move Ordering</li>
    <li>Search moves in sorted order</li>
</ol>


<h2>Parallelization</h2>

Minimax-based algorithms can perform significant pruning using the αβ method (αβ pruning), but this pruning assumes sequential search, so care must be taken to prevent pruning efficiency from decreasing when parallelizing the search.

Egaroucid performs parallelization in a shared memory environment, so it uses a combination of two algorithms: YBWC and Lazy SMP. I will introduce each of them.

### YBWC

YBWC (Young Brothers Wait Concept) is one of the parallelization algorithms for the αβ method. In the αβ method, move ordering searches from the leftmost move (the best move) sequentially, efficiently narrowing the search window. YBWC focuses on the fact that moves other than the leftmost move (if move ordering is perfect) do not contribute to narrowing the search window. Therefore, only the leftmost move is searched sequentially, and other moves ("Young Brothers") are searched in parallel after waiting for the leftmost move's search to complete ("Wait").

In Edax and Egaroucid, as a further implementation trick, efforts are made to not parallelize the rightmost two or three moves so that threads do not remain idle.

YBWC is a very convenient parallelization algorithm, but it is known for having poor parallelization efficiency.

### Lazy SMP

Lazy SMP is considered an effective alternative to YBWC for Chess AI and Shogi AI. It cleverly utilizes the shared transposition table between threads to run searches of various depths simultaneously across threads. It is very simple to implement but is considered quite useful.

There are no comprehensive websites summarizing the detailed implementation of Lazy SMP, but I personally referred to a doctoral thesis on Chess AI [(Østensen, 2016)](http://urn.nb.no/URN:NBN:no-56882).

Egaroucid has been using Lazy SMP for mid-game search since version 7.0.0. However, Lazy SMP alone can sometimes make the calculation time too slow (although the strength per unit time probably improves, searching a fixed depth can be slower), so it is used in conjunction with YBWC. Also, after endgame perfect reading, the benefits of Lazy SMP cannot be fully utilized, so only YBWC is used.

References

- (Østensen, 2016): Østensen Emil Fredrik: [A complete chess engine parallelized using lazy smp](http://urn.nb.no/URN:NBN:no-56882), MS thesis (2016) (As of November 2025, it seems to be no longer accessible)

Last updated: 2025/11/29

### Parallelization Algorithms for Distributed Memory Environments

Although not incorporated in Egaroucid, there are also parallelization algorithms for the αβ method for distributed memory environments. I will only introduce their names.

<ul>
    <li>[APHID](https://www.chessprogramming.org/APHID)</li>
    <li>[ADABA](https://doi.org/10.1007/s10462-022-10269-3)</li>
</ul>



## Final N-Move Optimization

A characteristic feature of Othello is the perfect reading in the endgame. While mid-game search may lead to mistakes depending on the accuracy of the evaluation function, once perfect reading is performed, there is absolutely no room for error (unless there are bugs).

Therefore, the speed of perfect reading greatly affects the strength of an Othello AI. If perfect reading can be performed early, the Othello AI becomes stronger. Egaroucid changes the timing of perfect reading depending on the level, but for example, at the default Level 21, perfect reading is performed when there are 24 empty squares or fewer. Also, at Level 21, reading (exploring to the end of the game but skipping some moves deemed bad by the evaluation function) is performed from 30 empty squares.

For example, perfect reading with 24 empty squares visits a considerable number of nodes, ranging from 1e7 to 1e8. Also, since the game tree is, as the name suggests, a tree structure, there are many nodes close to the leaves. This means that if only the last few moves of the endgame are optimized using dedicated functions, there is a prospect of speeding up the entire perfect reading process.

Therefore, as "final N-move optimization," I will explain the optimization methods for the last 1 to 4 moves used in Egaroucid.

In reality, it is difficult to determine whether it is close to the endgame, so the transition to a dedicated function is made by looking at how many empty squares are on the board.

### Optimization for 1 Empty Square

Normally, stone counting is done by counting the stones on the board after placing a stone, but for 1 empty square, the process of placing a stone on the board is no longer necessary. What is needed are only two numbers:

<ul>
    <li>The current stone difference on the board.</li>
    <li>The number of stones flipped when playing on the empty square.</li>
</ul>

The current stone difference on the board can be calculated by simply counting the number of stones of one player, utilizing the characteristic that the board always has 1 empty square.

The number of stones flipped when playing on an empty square is found by creating a dedicated function. The number of flipped stones is calculated for each of the four directions (vertical, horizontal, diagonal) from the empty square and then summed up. At this time, the number of flipped stones for each line is pre-calculated, and in the actual search, it is implemented by extracting lines and performing table lookups.

Also, if it is clearly possible to prune compared to the search window at the point when the current stone difference on the board is determined, the calculation of the number of flipped stones can be omitted.

Furthermore, if the number of flipped stones is 0, it is necessary to handle passes, and if neither player can make a move, the game ends as is. Here, if the game ends with 1 empty square (odd number of empty squares), there are no draws, which can slightly speed up the game ending process.

### Optimization for 2 Empty Squares

For 2 empty squares, legal move generation was omitted, and if there were opponent's stones around the empty squares, they were simply considered legal moves, the flipped stones were calculated, and a move was made if stones were flipped.

Also, for 2 empty squares, quasi-parity theory (see Move Ordering section) has no meaning, so no move ordering is performed. However, if there is a move sequence that guarantees two consecutive moves, that should be prioritized (i.e., prioritize searching squares that the opponent can also play).

### Optimization for 3 Empty Squares

For 3 empty squares, move ordering can be performed using quasi-parity theory. Looking at each quadrant of the board, if one quadrant has 1 empty square and another quadrant has 2 empty squares, moves in the quadrant with 1 empty square are prioritized. To eliminate conditional branching in this ordering, the move ordering is achieved by using the position of the square (represented by 0 to 63) and a bit that corresponds to the quadrant, along with SIMD's shuffle function and table lookups. This idea is based on [Mr. Okuhara's explanation](http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm) of optimizing Edax for AVX.

For 3 empty squares, similar to 2 empty squares, legal move generation is omitted, and legal moves are simply determined, and moves are attempted in order. If the game ends with 3 empty squares, there are no draws, so the stone difference is calculated with a function that omits draw handling.

### Optimization for 4 Empty Squares

For 4 empty squares, move ordering is performed using quasi-parity theory. If there are 1 empty square, 1 empty square, 2 empty squares, and 0 empty squares per quadrant, the two quadrants with 1 empty square are prioritized for search. This also uses shuffle-based optimization to eliminate conditional branching, similar to 3 empty squares.

For 4 empty squares, legal move generation is omitted, and legal moves are simply determined, and moves are attempted in order.



## Countermeasures against Special Endings

Egaroucid performs mid-game search in the opening. This is a search that uses evaluation values from an evaluation function as search results without reading to the end of the game.

However, evaluation functions are naturally not perfect. For example, if a large number of stones are taken, even if the position is overwhelmingly advantageous, there may be special endings that cannot be found by the evaluation function alone, such as mistakenly falling into a wipeout line.

For example, in the following position, it is White's turn, and White is in an overwhelmingly advantageous situation. However, if White mistakenly plays e6, Black plays e3, and White is wiped out and loses.

<div class="centering_box">
    <img class="pic2" src="img/clog.png">
</div>


To avoid such special endings (e.g., advantageous but with one very bad move), Egaroucid quickly enumerates all possible moves up to a certain depth, confirms that such special endings are not found, and then performs a mid-game search. Since only bitboard processing is performed, it can be implemented very quickly.



## Search Using GPU

As of version 7.7.0, Egaroucid uses only the CPU for searching. However, I have experimented with whether searching could be sped up further using a GPU. I will briefly explain this, including previous efforts. Overall, unfortunately, the results have not been sufficient to replace CPU-based searching.

Last updated: 2025/11/29

<h4>My Research in 2024</h4>

I devised a method for searching endgame positions using a GPU and compared execution times with the CPU [(Yamana 2024)](https://ipsj.ixsq.nii.ac.jp/records/232914).

This research uses a partial parallel αβ search with two types of load balancing algorithms and a simple move ordering algorithm. Load balancing was performed based on problem units and problem division.

<div class="centering_box">
    <img class="pic2" src="img/gpu_search_load_balancing1.png">
    <img class="pic2" src="img/gpu_search_load_balancing2.png">
</div>

I used an RTX3090 as the GPU and a Core-i9 13900K as the CPU to investigate how much speedup was achieved with the GPU compared to the CPU. The results are shown in the graph below. If the vertical axis is greater than 1, it means the GPU is faster than the CPU. In this graph, the number of problems was fixed at $2^{24}=16777216$, and the search time was measured by varying the number of empty squares (almost equivalent to search depth). As a result, the GPU was faster than the CPU up to 13 empty squares (approximately 13-ply lookahead), but for more empty squares, the CPU was faster.

<div class="centering_box">
    <img class="pic2" src="img/gpu_search_graph1.png">
</div>

Furthermore, I plotted the ratio of visited nodes and NPS to the CPU when the number of empty squares was varied while keeping the number of problems fixed. It can be seen that as the number of empty squares increases, the number of visited nodes increases dramatically in the GPU, and NPS gradually decreases. The increase in the number of visited nodes is likely due to the simpler GPU move ordering compared to the CPU version. The decrease in NPS is likely due to the limitations of the load balancing algorithm.

<div class="centering_box">
    <img class="pic" src="img/gpu_search_graph3.png">
</div>

In the example above, the number of problems was fixed, but I also conducted an experiment where the number of problems was varied while keeping the number of empty squares fixed. Although this is an experiment with slightly older code, I will post it. It can be seen that when the number of problems generally exceeds $10^5$, the GPU achieves top speed. Since the key to speeding up a GPU is how to operate a large number of threads in parallel, with my algorithm, many problems were required to bring out the GPU's performance. I believe that for any algorithm other than mine, if searching with a GPU, top speed will not be achieved unless there is a certain number of problems.

<div class="centering_box">
    <img class="pic2" src="img/gpu_search_graph2.png">
</div>

References

- (Yamana 2024): Takuto Yamana: [Fast Othello Search Method Using GPGPU and αβ Algorithm](https://ipsj.ixsq.nii.ac.jp/records/232914), IEICE Technical Report, Vol. 2024-GI-51, No. 26, pp. 1-9 (2024)

Last updated: 2025/11/29

<h4>Mr. Sōsu-Poyo's Efforts Around 2016</h4>

This is an effort to perform a perfect read of 1,091,780 positions with 10 empty squares using a GPU [(Sōsu-Poyo 2016-2018)](https://primenumber.hatenadiary.jp/entry/2016/12/20/003746). As a result of various efforts, it took less than 9 seconds with a GPU (GTX1080) and 1 minute and 13 seconds with a CPU (Core i7-6700K). As a result of various efforts, it took less than 9 seconds with a GPU (GTX1080) and 1 minute and 13 seconds with a CPU (Core i7-6700K).

Normally, search is written recursively, but it seems to have been sped up by unfolding it into a loop and storing the stack contents in Shared Memory. Furthermore, it also sped up by using 4 threads in a SIMD-like manner, reading one position with 4 threads.

Related code is publicly available on GitHub [(Sōsu-Poyo 2024)](https://github.com/primenumber/GPUOthello2).

References

- (Sōsu-Poyo 2016-2018): Sōsu-Poyo: [Solving Othello Super Fast with GPGPU](https://primenumber.hatenadiary.jp/entry/2016/12/20/003746) (2016-2018)
- (Sōsu-Poyo 2024): Sōsu-Poyo, primenumber/GPUOthello2: [GPUOthello2](https://github.com/primenumber/GPUOthello2)

Last updated: 2025/11/29

<h4>Mr. Okuhara's Efforts</h4>

There is an effort to perform a perfect read of 5 empty squares using a GPU [(Okuhara 20XX)](http://www.amy.hi-ho.ne.jp/okuhara/flipcuda.htm). This is not a regular αβ method, but a brute-force search that tries playing on all empty squares (including illegal moves). Therefore, for 5 empty squares, $5!=120$ searches are performed.

With this approach, the result was ultimately an order of magnitude slower than the CPU's αβ method. The code is also publicly available on the cited page.

References

- (Okuhara 20XX): Toshihiko Okuhara: [Experiments on bitboard implementation using GPU computing](http://www.amy.hi-ho.ne.jp/okuhara/flipcuda.htm)

Last updated: 2025/11/29



## Game Characteristics

Here, I will summarize my knowledge about the game of Othello itself, obtained as a byproduct of Othello AI development.

### Number of Othello Opening Expansions

In Othello, there is one position after one move, considering symmetrical shapes as identical (because playing in any of the four initial squares is essentially the same). After two moves, there are three positions (vertical capture, diagonal capture, parallel capture). By thinking this way, the total number of Othello opening expansions can be understood. I calculated this with a computer. "Considering symmetrical shapes as identical" means that positions that match by rotation, reflection, or point symmetry are all considered identical. Also, if the game ends midway, that position is not counted after the end of the game (Othello ends in a minimum of 9 moves, but for example, a position that ends in 9 moves is not counted after the 10th move).

<div class="table_wrapper"><table>
<tr>
<th>Moves Played</th>
<th>Number of Positions</th>
</tr>
<tr>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>3</td>
<td>14</td>
</tr>
<tr>
<td>4</td>
<td>60</td>
</tr>
<tr>
<td>5</td>
<td>322</td>
</tr>
<tr>
<td>6</td>
<td>1773</td>
</tr>
<tr>
<td>7</td>
<td>10649</td>
</tr>
<tr>
<td>8</td>
<td>67245</td>
</tr>
<tr>
<td>9</td>
<td>434029</td>
</tr>
<tr>
<td>10</td>
<td>2958586</td>
</tr>
<tr>
<td>11</td>
<td>19786627</td>
</tr>
<tr>
<td>12</td>
<td>137642461</td>
</tr>
</table></div>

Actually, this information was investigated when I thought "let's use all expansions for the first N moves as training data" when generating Egaroucid's evaluation function, and then determined the appropriate N.

By the way, previous researchers have done similar calculations, such as [counting the number of continuations after fixing only the first move](https://hasera.net/othello/mame009.html) and [not identifying any symmetrical positions (Perft for Reversi)](https://www.aartbik.com/strategy.php).

### Estimation of State Space Size

Let's consider the "complexity" of the game of Othello. When considering the complexity of a game, it is often common to use the number of reachable positions (legal positions) or the number of possible game records (game tree size) in that game. These two values are very confusing, so first, I will explain the difference between these numbers, and then I will introduce the latest research (to my knowledge) regarding each number.

The number of legal positions is well-known as $10^{28}$ in [English Wikipedia](https://en.wikipedia.org/wiki/Game_complexity), etc., but as of 2025, there is recent research estimating it to be $10^{26}$. Also, for the game tree size, the value of $10^{58\sim60}$ is well-known as a common belief, but there is an anonymous discussion estimating it to be $10^{54}$.

Last updated: 2025/11/30

<h4>Difference between Legal Positions and Game Tree Size</h4>

There are two types of numbers that indicate the complexity of a game, and they are very confusing, so I will briefly explain them. Sometimes you see descriptions like "the number of Othello states is $10^{58\sim60}$," but this is a different number from the number of legal positions (State-space complexity), and it is a number that considers the size of the game tree (Game tree size).

The number of legal positions counts the number of positions that can be reached from the initial board. In other words, it represents the number of existing positions in that game itself. On the other hand, the size of the game tree is the number of leaf nodes in the game tree, which is equivalent to the number of possible game records in that game.

In general, the game tree size tends to be larger than the number of legal positions. For example, in Othello, `f5d6c3d3c4` and `f5d6c4d3c3` are different game records (whether to go from the tiger system or the cat system to the tiger opening), but they result in exactly the same state. When counting legal positions, these positions are considered identical, but when considering the size of the game tree, they are considered different. This situation, where different game records lead to the same state, occurs very frequently in Othello. Therefore, the game tree size is larger than the number of legal positions (although these two numbers count completely different things).

<div class="centering_box">
    <img class="pic2" src="img/confluence_example.png">
</div>
The estimate of $10^{60}$ for Othello's game tree size is likely based on the fact that Othello generally has about 10 legal moves per position, and a game typically ends in about 60 moves. The $10^{58}$ estimate likely came from reducing one move for the symmetry of the first move and one move for the last move, as there is no choice for the last move. However, after 9 empty squares, the number of legal moves obviously falls below 10, and the number of legal moves in the opening is often less than 10. On the other hand, the number of legal moves in the mid-game often exceeds 10. Therefore, this estimate of the game tree size is not one that can be expected to be highly accurate.

By the way, a trivial upper bound for the number of legal positions in Othello is $2^4 \times 3^{60} = 6.783 \times 10^{29}$, considering the arrangement of the four central stones (black/white) and the arrangement of the other 60 squares (black/white/empty). Even seeing that this number is extremely small compared to the commonly cited $10^{58}$ for game tree size, it is clear that $10^{58}$ is an unsuitable value to represent the number of legal positions.

Last updated: 2025/11/30

<h4>Number of Legal Positions - Research by Ishii and Tanaka in 2025</h4>

[(Ishii, Tanaka 2025)](https://ipsj.ixsq.nii.ac.jp/records/2005522) and [(Ishii, Tanaka 2025 Presentation Slides)](https://github.com/u-tokyo-gps-tanaka-lab/othello_complexity_rs/blob/master/conference-slide-ja.pdf) estimate the number of legal positions in Othello. As a result, it was found that in Othello, there are between $1.675 \times 10^{26}$ and $3.765 \times 10^{26}$ legal positions with a significance level of 99.5%.

This research estimates the number of legal positions in Othello by randomly generating one million positions with stones randomly placed on the board and counting how many of them were legal positions. The method for checking the legality of random positions is very well thought out and interesting (which is why I haven't fully grasped it myself), so please read the paper and code. The legality determination is explained not only in the paper but also in detail in the publicly available code [(Ishii, Tanaka 2025 GitHub)](https://github.com/u-tokyo-gps-tanaka-lab/othello_complexity_rs), and I am currently studying it. Also, this research includes positions whose legality is unknown, but it cleverly takes them into statistical consideration, which is very interesting.

In this research, symmetrical (line symmetry, rotational symmetry) board positions are considered identical. Also, positions where the same position occurs with a different turn are also considered identical (although such positions would be rare in Othello).

References:

- (Ishii, Tanaka 2025): Sotaro Ishii, Tetsuro Tanaka: [Estimation of the Number of Reachable Positions in Othello](https://ipsj.ixsq.nii.ac.jp/records/2005522), Game Programming Workshop 2025 Proceedings, Vol. 2025, pp.171-178 (2025)
- (Ishii, Tanaka 2025 Presentation Slides): Sotaro Ishii, Tetsuro Tanaka. u-tokyo-gps-tanaka-lab/othello_complexity_rs: [Estimation of the Number of Reachable Positions in Othello](https://github.com/u-tokyo-gps-tanaka-lab/othello_complexity_rs/blob/master/conference-slide-ja.pdf)
- (Ishii, Tanaka 2025 GitHub): Sotaro Ishii, Tetsuro Tanaka. [u-tokyo-gps-tanaka-lab/othello_complexity_rs](https://github.com/u-tokyo-gps-tanaka-lab/othello_complexity_rs)

Last updated: 2025/11/29

<h4>Game Tree Size - Result of Anonymous Discussion</h4>

Between 2005 and 2006, a discussion was held to estimate the game tree size of Othello [(@WIKI 2005-2006)](https://w.atwiki.jp/othello/pages/35.html). As a result, the game tree size was finally estimated to be $6.449\times10^{54}$. The basic approach was to obtain data on the number of legal moves at each turn through random games and then estimate based on that. At this time, the ingenuity seems to be to take the (arithmetic) mean of "the product of the number of legal moves at each position up to n moves" instead of taking the average of the number of legal moves for m positions at the nth move.

Note that this number does not seem to consider the symmetry of playable moves. For example, there are four possible moves for the first move (d3, c4, f5, e6), but they are all essentially the same move. However, here these moves are all treated as distinct.

[(@WIKI's Memo 2006)](https://w.atwiki.jp/othello/pages/32.html) describes the estimation of game tree size at each depth, and it can be seen that the values up to the 21st move are quite close to the strict values shown in [(Brobecker et al. 2006-2021)](https://oeis.org/A124004).

References:

- (@WIKI 2005-2006): [How many Othello game results are there? @Wiki](https://w.atwiki.jp/othello/pages/35.html) (2005-2006)
- (@WIKI's Memo 2006): [Memo 05 Average of Product of Moves](https://w.atwiki.jp/othello/pages/32.html), How many Othello game results are there? @Wiki (2006)
- (Brobecker et al. 2006-2021): Alain Brobecker, Paul Byrne, Charles R Greathouse IV, and Dominic Hofer: [Number of possible Reversi games at the end of the n-th ply.](https://oeis.org/A124004) (2006-2021)

Last updated: 2025/11/30

<h4>Number of Legal Positions - My Rough Estimate</h4>

For now, let's try a rough estimate without considering whether it can be reached from the first move. Othello is played on a 64-square board, and each square can be black, white, or empty, so $3^{64} = 3.4 \times 10^{30}$ is an upper bound on the size of the state space. Considering that the initial 4 squares are filled (only two types, black or white), it is $2^4 \times 3^{60} = 6.8 \times 10^{29}$.

Also, let's consider how many positions there are for each move. The number of legal positions $s(p)$ possible for the $p$-th move can be roughly estimated as:

$s(p) = 2^4 \times {}_{60}\mathrm{C}_p \times 2^p$

The first term is the combination of the initially filled 4 squares, the second term is the number of ways to fill $p$ squares with stones out of the remaining 60 squares, and the third term is the combination of colors of the stones filled up to the $p$-th move. Calculating this for $0 \leq p \leq 60$ gives the graph below. The number of positions increases from the beginning to around the 40th move, and then decreases slightly as it approaches the end of the game. The blue line is the upper bound for reference. Also, for $p \leq 12$, the exact value is known because all opening expansions were calculated above. I also plotted that.

<div class="centering_box">
    <img class="pic2" src="img/state_space_complexity_0.png">
</div>
This upper bound is known to be not very accurate, but I will accept it for now.

Another method for estimating the size of the state space that might be useful is to investigate the relationship between the size of the evaluation function and generalization performance. For example, [Egaroucid's evaluation function training results](#評価関数の最適化_最急降下法) showed that `test_mse` was maximized around the 45th move. Therefore, it might be possible to imagine that the peak in the graph above, which is around the 40th move, is actually around the 45th move.



### Shortest Draw

In Othello, games sometimes end with many empty squares, but here we consider a draw with a sparse board.

The shortest draw was 20 moves, with 185 game records (standardized to start with f5). The following are examples of game records.

`f5f4c3f6g7f7f3h7f8b2h6e7h8e3d7g3f2f1a1c5`
`f5f4c3f6g7f7f3h7f8b2h6e7h8e3d7g2f2f1a1c5`
`f5f4c3f6g7f7f3h7f8b2h6e3a1e7f2f1h8e2d7c5`
`f5f6e6f4g7c6f3g2f2f7b7a8h2f1f8h3e2h1c4d2`
`f5d6c6b6b7f6b5a6e6a8b8f3f7g6g2c8h6h1d3b4`
`f5d6c6b6b7f6b5a6f7f3g2b4e6g6b3h1h6a8d3b2`
`f5f6e6f4g5c6f3g2f2f7b7a8h2f1f8h3e2h1c4d2`
`f5f4g3e6c4b3b4g4f7g8b2a2h4b5b6a4d7b7d6b1`
`f5f4g3e6c4b3b4g4b2a4f7c2h4g8d2d1d7e2d6a2`
`f5f4g3e6c4b3b4g4f7g8b2a2d6b5b6c2h4a4d7b7`


There were 10 types of game endings, considering line symmetry, point symmetry, and rotational symmetry as identical.

<div class="centering_box">
    <img class="pic2" src="img/shortest_draw.png">
</div>

I wrote a [detailed explanation article](https://qiita.com/Nyanyan_Cube/items/ccab30af5c6a2b9d1e06) on the shortest draw on Qiita. Also, the [code used to find this result](https://github.com/Nyanyan/Shortest_Draw_Othello) is publicly available on GitHub.

One of the shortest draws was found manually before my efforts. A [manual search explanation article](https://note.com/berlin9/n/nc0e02c83b636) is available on note.



### Shortest Stoner

In Othello, there is a tactic called Stoner. Stoner activates in a minimum of 13 moves from the start of the game. An example of a 13-move Stoner is as follows:

`e6d6c6d7c8b6c7f7f6e8f8g8b7`

<div class="centering_box">
    <img class="pic2" src="img/shortest_stoner.png">
</div>

I wrote a [detailed explanation article](https://qiita.com/Nyanyan_Cube/items/bd0f808dbece005256e6) on finding the shortest Stoner on Qiita. Also, the [code used to find this result](https://github.com/Nyanyan/Shortest_Stoner) is publicly available on GitHub.



### Average Number of Flipped Stones

In Othello, what is the average number of flipped stones? I tried to estimate this. As a result, it was found that the average number of stones flipped per legal move, from the 1st move to the 60th move, is approximately 2.244. The code used for this experiment is publicly available [here](https://github.com/Nyanyan/Count_Flipped_Othello).

Plotting this result for each move is as follows. The blue line shows the average, and the yellow shaded area represents the range of average ± standard deviation. This means that if a random board is set up and a random move is made, the number of flipped stones (assuming the distribution of flipped stones is a binomial distribution) will fall within the yellow area with approximately 68% probability. It does not mean that the average value itself will fluctuate within the yellow range.

Note that the results up to the 14th move (green vertical line) are exact, calculated by enumerating all possible states. The results from the 15th move onwards are estimates based on $10^7$ games (by random moves).

<div class="centering_box">
    <img class="pic2" src="img/n_flipped_graph.png">
</div>



## Reference Materials

Here are some materials (and more) that I have referenced from being an Othello AI beginner to where I am now.

I will introduce Othello AIs that publish code, Othello AIs that provide unique algorithm explanations, and recommended books for Othello AI development. The order is arbitrary.

### Documents and Articles

<h4>Othello AI Textbook</h4>

Something I wrote before. Unlike this site, it covers the basics carefully with sample code, making it easy to learn. However, some of the content is a bit old... I will update it someday.

<ul>
    <li>[Othello AI Textbook](https://note.com/nyanyan_cubetech/m/m54104c8d2f12)</li>
</ul>
<h4>Practical Introduction to Search Algorithms Learned from Games</h4>

This is a book by thunder, who is active in game AI contests, and the articles that were its source. Due to the publication date, I couldn't reference it in real-time, but it's a good book for learning about search algorithms broadly, not just Othello AI. From an Othello AI perspective, the explanation of αβ pruning is very easy to understand and recommended.

Actually, about a month before I started developing Othello AI, I had the opportunity to be taught about the minimax method by thunder by chance.

<ul>
    <li>[Practical Introduction to Search Algorithms Learned from Games ~ Tree Search and Metaheuristics (Gihyo)](https://gihyo.jp/book/2023/978-4-297-13360-3)</li>
    <li>[Introduction to Game Tree Search Taught from Scratch by a Four-Time World Champion AI Engineer](https://qiita.com/thun-c/items/058743a25c37c87b8aa4)</li>
</ul>
<h4>How to Create a Reversi Program</h4>

This is a collection of articles that carefully explain Othello AI creation from the basics. I personally found the explanation of MPC (Multi-ProbCut) particularly helpful. It also comes with sample programs. Although it seems to be a somewhat old article, it still contains a lot of useful information.

<ul>
    <li>[How to Create a Reversi Program](http://www.es-cube.net/es-cube/reversi/sample/index.html)</li>
</ul>
<h4>Chess Programming Wiki</h4>

Considering Chess AI, it contains various information related to game AI. There are many items with little description, but it can also be used to trace cited literature, which is useful.

<ul>
    <li>[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)</li>
</ul>

### Othello AI

<h4>Edax</h4>

Edax is a widely used strong Othello AI. It is very well-balanced in all aspects, and reading its code is very educational. All its code, in addition to binaries, is publicly available on GitHub, and I print it all out and read it occasionally.

<ul>
    <li>[edax-reversi (GitHub)](https://github.com/abulmo/edax-reversi)</li>
</ul>

<h4>Edax-AVX</h4>

This is a SIMDized and sped-up version of Edax. Edax does not use CPU-level instructions, but this version speeds up Edax by a constant factor through various SIMD optimizations. It is the fastest Othello AI I know of. By the way, the [paper that weakly solved Othello](https://doi.org/10.48550/arXiv.2310.19387) states that this Othello AI was used for searching. Also, Mr. Okuhara, who performed the SIMDization, provides very helpful (Japanese!) explanations of various SIMDization techniques for Othello AI. The code for Edax-AVX is also publicly available, and I also print it out and read it occasionally.

<ul>
    <li>[edax-reversi-AVX (GitHub)](https://github.com/okuhara/edax-reversi-AVX)</li>
    <li>[Reversi Bitboard Techniques](http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm)</li>
    <li>[Edax AVX - Optimizations other than bitboard](http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm)</li>
</ul>

<h4>Thell</h4>

An Othello AI developed until around 2005. It uniquely publishes technical information (in Japanese!). Especially, the materials on evaluation functions are very useful. Note that Thell's board implementation method is unique, and I tried it myself, but I eventually concluded that bitboards are faster than Thell's method.

<ul>
    <li>[Thell](https://sealsoft.jp/thell/index.html)</li>
    <li>[Thell Algorithm Explanation](https://sealsoft.jp/thell/algorithm.html)</li>
    <li>[Optimization of Reversi Evaluation Function](https://sealsoft.jp/thell/learning.pdf)</li>
</ul>

<h4>Zebra</h4>

Although a somewhat old Othello AI, before Edax, it was used by Othello players as the "only choice." It uniquely publishes technical explanations (in English) and also freely distributes a huge Book. However, looking at this Book with modern Othello AIs, its accuracy is not very high. Nevertheless, I thought it could be very effectively used as a seed for Book generation, so Egaroucid obtained permission and created its own Book based on Zebra's Book (though almost none of the original remains now...).
Also, since someone has translated the technical explanation into Japanese, I will also provide that link.

<ul>
    <li>[Zebra](http://radagast.se/Othello/)</li>
    <li>[Writing an Othello Program](http://radagast.se/Othello/howto.html)</li>
    <li>[Internal workings of a strong Othello program](http://www.amy.hi-ho.ne.jp/okuhara/howtoj.htm)</li>
</ul>

<h4>Logistello</h4>

An Othello AI that fought and defeated the human world champion (Takeshi Murakami, then 7-dan) in 1997. Although a very old Othello AI, the author's papers are very useful for Othello AI development and understanding the history of Othello AI.

<ul>
    <li>[LOGISTELLO](https://skatgame.net/mburo/log.html)</li>
    <li>[Experiments with Multi-ProbCut and a New High-Quality Evaluation Function for Othello](https://skatgame.net/mburo/ps/improve.pdf)</li>
    <li>[From Simple Features to Sophisticated Evaluation Functions](https://skatgame.net/mburo/ps/pattern.pdf)</li>
    <li>[Takeshi Murakami vs. Logistello](https://skatgame.net/mburo/ps/match-report.pdf)</li>
</ul>


<h4>FOREST</h4>

FOREST is an Othello AI with a slightly different flavor from the Othello AIs mentioned so far. It incorporates deep learning into the αβ method, and its characteristic is that it performs inference during search. It seems to be designed to prioritize the accuracy of the evaluation function. Also, FOREST has been continuously developed from 1994 to the present (2023), so it has a history. There are English technical documents and Japanese translations of them.

<ul>
    <li>[FOREST, my Othello™ AI program](https://lapagedolivier.fr/forest.htm)</li>
    <li>[Developing an Artificial Intelligence for Othello/Reversi](https://lapagedolivier.fr/neurone.htm)</li>
    <li>[[Translation] Developing an Artificial Intelligence for Othello/Reversi](https://qiita.com/sensuikan1973/items/2fda85acc0411698ee8c)</li>
</ul>
