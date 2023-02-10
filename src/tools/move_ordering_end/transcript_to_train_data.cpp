#include "./../../engine/board.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "move_evaluation_definition_20230210.hpp"

void trs_init(){
    bit_init();
    mobility_init();
    flip_init();
}

std::string trs_fill0(int n, int d){
    std::ostringstream ss;
	ss << std::setw(d) << std::setfill('0') << n;
	return ss.str();
}

struct Trs_Convert_transcript_info{
    Board board;
    int8_t player;
    int8_t policy;
};

void trs_convert_transcript(std::string transcript, std::ofstream *fout){
    int8_t y, x;
    std::vector<Trs_Convert_transcript_info> boards;
    Trs_Convert_transcript_info board_info;
    Flip flip;
    board_info.board.reset();
    board_info.player = BLACK;
    for (int i = 0; i < (int)transcript.size(); i += 2){
        if (board_info.board.get_legal() == 0){
            board_info.board.pass();
            board_info.player ^= 1;
        }
        x = (int)(transcript[i] - 'a');
        if (x < 0 || x >= HW)
            x = (int)(transcript[i] - 'A');
        y = (int)(transcript[i + 1] - '1');
        board_info.policy = HW2_M1 - (y * HW + x);
        if (47 <= i && i < 53)
            boards.emplace_back(board_info);
        calc_flip(&flip, &board_info.board, board_info.policy);
        if (flip.flip == 0ULL){
            std::cerr << "illegal move found in move " << i / 2 << " in " << transcript << std::endl;
            return;
        }
        board_info.board.move_board(&flip);
        board_info.player ^= 1;
    }
    if (board_info.board.get_legal()){
        return;
    }
    board_info.board.pass();
    board_info.player ^= 1;
    if (board_info.board.get_legal()){
        return;
    }
    uint64_t legal;
    uint16_t eval_idxes[ADJ_N_FEATURES];
    uint16_t score;
    for (Trs_Convert_transcript_info &datum: boards){
        legal = datum.board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            adj_calc_features(&datum.board, cell, eval_idxes);
            for (int i = 0; i < ADJ_N_FEATURES; ++i)
                fout->write((char*)&(eval_idxes[i]), 2);
            score = (cell == datum.policy);
            fout->write((char*)&score, 2);
        }
    }
}

int main(int argc, char* argv[]){
    if (argc < 5){
        std::cerr << "input [in_dir] [start_file_num] [end_file_num] [out_file]" << std::endl;
        return 1;
    }
    trs_init();
    std::string in_dir = std::string(argv[1]);
    int strt_file_num = atoi(argv[2]);
    int end_file_num = atoi(argv[3]);
    std::string out_file = std::string(argv[4]);
    std::ofstream fout;
    fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }
    int t = 0;
    for (int file_num = strt_file_num; file_num < end_file_num; ++file_num){
        std::cerr << "=";
        std::string file = in_dir + "/" + trs_fill0(file_num, 7) + ".txt";
        std::ifstream ifs(file);
        if (!ifs) {
            std::cerr << "can't open " << file << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(ifs, line)){
            trs_convert_transcript(line, &fout);
            ++t;
        }
    }
    std::cerr << std::endl;
    std::cout << t << " games found" << std::endl;
    return 0;
}