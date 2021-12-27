#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "board.hpp"

using namespace std;

class joseki{
    private:
        vector<pair<board, string>> arr;
    public:
        inline void init(){
            ifstream ifs("resources/joseki.txt");
            if (ifs.fail()) {
                cerr << "joseki file not found" << endl;
                exit(0);
            }
            string line, name;
            int board_arr[hw2];
            int i;
            board b;
            while (getline(ifs, line)) {
                for (i = 0; i < hw2; ++i){
                    if (line[i] == '0')
                        board_arr[i] = black;
                    else if (line[i] == '1')
                        board_arr[i] = white;
                    else
                        board_arr[i] = vacant;
                }
                b.translate_from_arr(board_arr, black);
                name.clear();
                for (i = hw2 + 1; i < (int)line.size(); ++i)
                    name += line[i];
                arr.push_back(make_pair(b, name));
            }
        }

        inline string get(){

        }

};

joseki joseki;

inline void joseki_init(){
    joseki.init();
}