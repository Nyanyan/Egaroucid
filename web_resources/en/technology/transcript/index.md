# Egaroucid Self-Play Transcripts



## Download

Please download [Egaroucid_Transcript.zip](https://github.com/Nyanyan/Egaroucid/releases/download/transcript/Egaroucid_Transcript.zip) and unzip it.



## Abstract

This is a set of transcripts played by Egaroucid, Othello AI.

There are many data (2 million games), and you can use it in creating your Othello AI.

Website: [https://www.egaroucid.nyanyan.dev/ja/](https://www.egaroucid.nyanyan.dev/ja/)

GitHub Repository: [https://github.com/Nyanyan/Egaroucid](https://github.com/Nyanyan/Egaroucid)

Author: Takuto Yamana ( [https://nyanyan.dev/ja/](https://nyanyan.dev/ja/) )



## Terms of Service

<ul>
    <li>You can use this data freely in your activities, such as creating an evaluation function of Othello.
        <ul>
            <li>If you used this data and you thought very useful, I, Takuto Yamana, would be very pleased if you told me that, or you wrote something like "I used Egaroucid's self-play data for training my Othello AI".</li>
        </ul>
    </li>
    <li>I am not responsible for any damage caused by using this data. Please use at your own risk.</li>
    <li>Redistribution of this data is prohibited.
        <ul>
            <li>Please advertise Egaroucid's website or GitHub if you would like to promote.</li>
        </ul>
    </li>
</ul>



## Details

You can see text files formatted like ```XXXXXXX.txt``` in each directory,  and inside it, there are 10 thousand ```f5d6``` formatted transcripts.

Since this data was generated for training Egaroucid's evaluation function, the first $N$ moves are played randomly. This number $N$ is determined by the following method.

1. Set constants $N_{min},N_{max}$
2. In each game, determine $N$ randomly. $N$ satisfies  $N_{min}\leq N \leq N_{max}$
3. Play first $N$ moves randomly, then start self-play

The details of the data for each directory are summarized below.

<div class="table_wrapper"><table>
<tr>
	<th>Directory</th>
	<th>0000_egaroucid_6_3_0_lv11</th>
</tr>
<tr>
	<td>AI Name</td>
	<td>Egaroucid for Console 6.3.0</td>
</tr>
<tr>
	<td>Level</td>
	<td>11</td>
</tr>
<tr>
	<td>Number of Games</td>
	<td>2,000,000</td>
</tr>
<tr>
	<td> $N_{min}$ </td>
	<td>10</td>
</tr>
<tr>
	<td> $N_{max}$ </td>
	<td>19</td>
</tr>
    </table></div>



## History

<div class="table_wrapper"><table>
<tr>
	<th>Date</th>
	<th>Done</th>
</tr>
<tr>
	<td>2023/07/17</td>
	<td>First Release</td>
</tr>
    </table></div>
