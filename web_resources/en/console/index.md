# Egaroucid for Console

Operations differ depending on your OS. This software is for Windows, Linux and MacOS.



## Windows

Please download the zip file and unzip it, then you can use it. For the best performance, you can also build it on your own. Please see documents for Linux to build it.

### Download

Please download a zip file that is suitable to your environment, and unzip wherever you want. Then execute <code>Egaroucid_for_console.exe</code> to run.



Egaroucid is optimized to SIMD version, which requires AVX2  instructions, but old CPUs (created in 2013 or older) might not be able  to run it. If so, please install Generic version.



<table>
    <tr>
        <th>OS</th>
        <th>CPU</th>
        <th>Requirements</th>
        <th>Date</th>
        <th>Download</th>
    </tr>
    <tr>
        <td>Windows</td>
        <td>x64 (Standard)</td>
        <td>AVX2 (Standard)</td>
        <td>2023/09/01</td>
        <td>[Egaroucid for Console 6.4.0 Windows x64 SIMD](https://github.com/Nyanyan/Egaroucid/releases/download/console_v6.4.0/Egaroucid_for_Console_6_4_0_Windows_x64_SIMD.zip)</td>
    </tr>
    <tr>
        <td>Windows</td>
        <td>x64 (Standard)</td>
        <td>-</td>
        <td>2023/09/01</td>
        <td>[Egaroucid for Console 6.4.0 Windows x64 Generic](https://github.com/Nyanyan/Egaroucid/releases/download/console_v6.4.0/Egaroucid_for_Console_6_4_0_Windows_x64_Generic.zip)</td>
    </tr>
    <tr>
        <td>Windows</td>
        <td>x86</td>
        <td>-</td>
        <td>2023/09/01</td>
        <td>[Egaroucid for Console 6.4.0 Windows x86 Generic](https://github.com/Nyanyan/Egaroucid/releases/download/console_v6.4.0/Egaroucid_for_Console_6_4_0_Windows_x86_Generic.zip)</td>
    </tr>
</table>






Please visit [GitHub Releases](https://github.com/Nyanyan/Egaroucid/releases) to see older versions and release notes.



## Linux / MacOS

Please build on your own. You can use cmake or g++.

### Build with cmake

Please download source codes ([console_v6.3.0.zip](https://github.com/Nyanyan/Egaroucid/archive/refs/tags/console_v6.3.0.zip)) from GitHub Release, then unzip it.



Change directory.



<code>$ cd Egaroucid</code>



Then use <code>cmake</code> command to build.



<code>$ cmake -S . -B build [options]</code>



You can add additional options in <code>[options]</code>. Available options are:



<table>
    <tr>
        <th>You want to</th>
        <th>Add this option</th>
    </tr>
    <tr>
        <td>Build without AVX2</td>
        <td>-DHAS_NO_AVX2=ON</td>
    </tr>
    <tr>
        <td>Use ARM processors</td>
        <td>-DHAS_ARM_PROCESSOR=ON</td>
    </tr>
    <tr>
        <td>Use 32-bit environment</td>
        <td>-DHAS_32_BIT_OS=ON</td>
    </tr>
</table>




Then,



<code>$ cmake --build build</code>



That's all. You can see <code>Egaroucid_for_Console.out</code> in <code>Egaroucid/bin</code> directory. You can run with commands below.



<code>$ ./bin/Egaroucid_for_Console.out</code>



### Build with g++

Requirements are:

<ul>
    <li><code>g++</code> command
        <ul>
            <li>I tested with version 12.2.0 on Windows</li>
            <li>I tested with version 11.3.0 on Ubuntu</li>
        </ul>
    </li>
    <li>C++17</li>
</ul>

Please download source codes ([console_v6.3.0.zip](https://github.com/Nyanyan/Egaroucid/archive/refs/tags/console_v6.3.0.zip)) from GitHub Release, then unzip it.



Change directory.



<code>$ cd Egaroucid/src</code>



Then compile it with <code>g++</code> command. You can change the output name.



<code>$ g++ -O2 ./src/Egaroucid_console.cpp -o ./bin/Egaroucid_for_Console.out -mtune=native -march=native -mfpmath=both -pthread -std=c++17 -Wall -Wextra [options]</code>



You can add additional options in <code>[options]</code>. Available options are:

<table>
    <tr>
        <th>You want to</th>
        <th>Add this option</th>
    </tr>
    <tr>
        <td>Build without AVX2</td>
        <td>-DHAS_NO_AVX2</td>
    </tr>
    <tr>
        <td>Use ARM processors</td>
        <td>-DHAS_ARM_PROCESSOR</td>
    </tr>
    <tr>
        <td>Use 32-bit environment</td>
        <td>-DHAS_32_BIT_OS</td>
    </tr>
</table>




Then execute the output file.



<code>$ ./bin/Egaroucid_for_console.out</code>





## Usage

<code>$ Egaroucid_for_Console.exe -help</code>



or



<code>$ ./Egaroucid_for_Console.out -help</code>



to see how to use.



## Directory Structure

Egaroucid for Console uses some external files. If you've got a trouble, please check it.

<ul>
    <li>Egaroucid_for_Console.exe</li>
    <li>resources
        <ul>
            <li>hash (Files for hash)
                <ul>
                    <li>hash23.eghs</li>
                    <li>hash24.eghs</li>
                    <li>hash25.eghs</li>
                    <li>hash26.eghs</li>
                    <li>hash27.eghs</li>
                </ul>
            </li>
            <li>book.egbk (book file)</li>
            <li>eval.egev (evaluation file)</li>
        </ul>
    </li>
</ul>


## About Changing Book Format

Book format is changed in Egaroucid for Console 6.3.0. The new book's filename extension is ```.egbk2``` and it uses ```resources/book.egbk2``` as default book, but you can also use old book, which extension is ```.egbk```. If you want to use another book or old book, you can execute like this: ```Egaroucid_for_Console.exe -book [Book File]```.




## Documents for Go Text Protocol (GTP) users

GTP is a communication protocol made for game of Go, but you can play Othello with GTP on some applications. Some GTP commands are available on Egaroucid for Console, so these applications can communicate with Egaroucid for Console.



If you want to use GTP commands, please type this.



<code>$ Egaroucid_for_Console.exe -gtp</code>



I tested it works with GoGui on Windows and Quarry on Ubuntu.



### GoGui

GoGui with Egaroucid is something like this.

<div class="centering_box">
    <img class="pic2" src="img/gogui_with_egaroucid.png">
</div>
First, you have to register Egaroucid. Please add <code>-gtp</code> to the command, and set working directory <code>Egaroucid/src</code>.

<div class="centering_box">
    <img class="pic2" src="img/gogui_new_program.png">
    <img class="pic2" src="img/gogui_new_program2.png">
</div>
Then you can execute Egaroucid.

<div class="centering_box">
    <img class="pic2" src="img/gogui_launch.png">
</div>
On GoGui, board orientation is horizontally flipped, so you can see the ordinal board with flip the board horizontally again.

<div class="centering_box">
    <img class="pic2" src="img/gogui_orientation.png">
</div>



### Quarry

Egaroucid on Quarry is something like this.

<div class="centering_box">
    <img class="pic2" src="img/quarry_with_egaroucid.png">
</div>
First, you have to add Egaroucid. Open <code>Manage Engine List</code> via <code>New Game</code> or <code>Preferences</code>. Please add <code>-gtp</code> to the command.



Then start game to run Egaroucid.

<div class="centering_box">
    <img class="pic2" src="img/quarry_setting1.png">
    <img class="pic2" src="img/quarry_setting2.png">
</div>


