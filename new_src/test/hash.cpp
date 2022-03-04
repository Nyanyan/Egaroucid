#include <iostream>
#include <fstream>
#include "./../board.hpp"

using namespace std;

#define N_TESTCASES (999 * 430)

#define HASH_SIZE (1U << 28)
#define HASH_MASK ((1U << 28) - 1)

#define OFFSET 0

uint32_t result[HASH_SIZE];

void convert_board(string str, Board *board){
    board->player = 0;
    board->opponent = 0;
    for (int i = 0; i < HW2; ++i){
        if (str[i] == '0')
            board->player |= 1ULL << (HW2_M1 - i);
        else if (str[i] == '1')
            board->opponent |= 1ULL << (HW2_M1 - i);
    }
    board->p = str[65] - '0';
    if (board->p == 1)
        swap(board->player, board->opponent);
    if (board->player == 0 && board->opponent == 0)
        cerr << str << endl;
}

int main(){
    bit_init();
    flip_init();
    board_init();
    Board board;
    string str;
    uint32_t n_testcases = 0;
    for (uint32_t i = 0; i < HASH_SIZE; ++i)
        result[i] = 0;
    for (uint32_t file = OFFSET; file < 10 + N_TESTCASES / 999; ++file){
        cerr << "=";
        if ((file - OFFSET) % 25 == 24)
            cerr << endl;
        ifstream ifs("C:/github/egaroucid/Egaroucid5/evaluation/data/records4/00000" + to_string(file) + ".txt");
        for (uint32_t i = 0; i < 999; ++i){
            if (!getline(ifs, str))
                break;
            convert_board(str, &board);
            //if ((board.hash() & HASH_MASK) == 495270){
            //    //board.print();
            //    cerr << str << endl;
            //}
            //cerr << (board.hash() & HASH_MASK) << endl;
            ++result[board.hash() & HASH_MASK];
            ++n_testcases;
        }
        ifs.close();
    }
    uint32_t hash_conf = 0;
    for (uint32_t i = 0; i < HASH_SIZE; ++i){
        if (result[i] > 1){
            hash_conf += result[i] - 1;
            //cerr << i << " " << result[i] << endl;
        }
    }
    cerr << n_testcases << " " << HASH_SIZE << " " << ((double)HASH_SIZE / n_testcases) << endl;
    cerr << hash_conf << " " << (double)hash_conf / n_testcases << endl;

    return 0;
}