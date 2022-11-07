#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "util/board.hpp"

using namespace std;

#define N_DIRS 1

#define MIN_EMPTIES 1
#define MAX_EMPTIES 50

string idx_to_coord(int idx){
    int y = HW_M1 - idx / HW;
    int x = HW_M1 - idx % HW;
    const string x_coord = "abcdefgh";
    return x_coord[x] + to_string(y + 1);
}

string board_to_string(Board board){
    string res = "................................................................";
    uint_fast8_t cell;
    for (cell = first_bit(&(board.player)); board.player; cell = next_bit(&(board.player)))
        res[cell] = 'p';
    for (cell = first_bit(&(board.opponent)); board.opponent; cell = next_bit(&(board.opponent)))
        res[cell] = 'o';
    return res;
}

string score_to_string(int score){
    int s = (score + HW2) / 2; // 0 to 64
    return (string){(char)('!' + s)};
}

string coord_to_string(int coord){
    return (string){(char)('!' + coord)}; // coord: 0 to 63
}

void add_data(string transcript, ofstream *ofs){
    Board board;
    Flip flip;
    board.player = 0x0000000810000000ULL;
    board.opponent = 0x0000001008000000ULL;
    int ply, y, x, coord;
    vector<pair<string, int>> data;
    int player = 0;
    int n_discs = 4, n_empties;
    for (ply = 0; ply < (int)transcript.size(); ply += 2){
        if (board.get_legal() == 0ULL){
            board.pass();
            player ^= 1;
        }
        x = transcript[ply] - 'a';
        y = transcript[ply + 1] - '1';
        coord = HW2_M1 - (y * HW + x);
        n_empties = HW2 - n_discs;
        //if (MIN_EMPTIES <= n_empties && n_empties <= MAX_EMPTIES)
        data.emplace_back(make_pair(board_to_string(board) + coord_to_string(coord), player));
        calc_flip(&flip, &board, (uint_fast8_t)coord);
        board.move(&flip);
        player ^= 1;
        ++n_discs;
    }
    int score = board.score_player();
    int proc_score;
    for (pair<string, int> datum: data){
        if (datum.second == player)
            proc_score = score;
        else
            proc_score = -score;
        *ofs << datum.first << score_to_string(proc_score) << endl;
    }
}

int main(){
    bit_init();
    board_init();
    flip_init();
    
    string data_dir = "./../evaluation/third_party/";
    string output_dir = "data/";
    string sub_dirs[N_DIRS] = {
        "records16/"
    };
    int n_data[N_DIRS] = {
        1
    };
    for (int dir_idx = 0; dir_idx < N_DIRS; ++dir_idx){
        for (int file_idx = 0; file_idx < n_data[dir_idx]; ++file_idx){
            ostringstream sout;
            sout << setfill('0') << setw(7) << file_idx;
            string file_name = sout.str();
            cerr << data_dir + sub_dirs[dir_idx] + file_name + ".txt" << endl;
            ifstream ifs(data_dir + sub_dirs[dir_idx] + file_name + ".txt");
            string output_file = output_dir + sub_dirs[dir_idx] + file_name + ".txt";
            ofstream ofs;
            ofs.open(output_file, ios::out);
            string transcript;
            while (getline(ifs, transcript))
                add_data(transcript, &ofs);
            ofs.close();
        }
    }
    cerr << "done" << endl;
    return 0;
}