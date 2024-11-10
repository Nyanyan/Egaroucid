#include "./../../engine/board.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

void pck_init(){
    bit_init();
    mobility_init();
    flip_init();
}

std::string board_to_str(Board board, int8_t p){
    uint_fast8_t arr[HW2];
    board.translate_to_arr(arr, p);
    std::string res;
    for (int i = 0; i < HW2; ++i){
        if (arr[i] == BLACK)
            res += "X";
        else if (arr[i] == WHITE)
            res += "O";
        else
            res += "-";
    }
    res += " ";
    if (p == BLACK)
        res += "X";
    else
        res += "O";
    return res;
}

int main(int argc, char* argv[]){
    if (argc < 4){
        std::cerr << "input [in_file] [n_discs_start] [n_discs_end] [n_problems]" << std::endl;
        return 1;
    }
    pck_init();
    std::string in_file = std::string(argv[1]);
    int n_discs_start = atoi(argv[2]);
    int n_discs_end = atoi(argv[3]);
    int n_problems = atoi(argv[4]);
    std::cerr << in_file << " " << n_discs_start << " " << n_discs_end << " " << n_problems << std::endl;
    FILE* fp;
    if (fopen_s(&fp, in_file.c_str(), "rb") != 0) {
        std::cerr << "can't open " << in_file << std::endl;
        return 1;
    }
    int n = 0;
    Board board;
    int8_t player, score, policy;
    while (n < n_problems){
        if (fread(&(board.player), 8, 1, fp) < 1)
            break;
        fread(&(board.opponent), 8, 1, fp);
        fread(&player, 1, 1, fp);
        fread(&policy, 1, 1, fp);
        fread(&score, 1, 1, fp);
        int n_discs = board.n_discs();
        if (n_discs >= n_discs_start && n_discs <= n_discs_end && myrandom() < 0.01){
            std::cout << board_to_str(board, player) << std::endl;
            ++n;
        }
    }
    std::cerr << n << " results printed" << std::endl;
    return 0;
}