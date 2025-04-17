#include "./../../engine/board.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

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

#define MODE_OUTPUT_ERROR 0
#define MODE_OUTPUT_OK 1

void trs_convert_transcript(std::string transcript, int expected_score_black, int mode){
    int8_t y, x;
    std::vector<Trs_Convert_transcript_info> boards;
    Trs_Convert_transcript_info board;
    Flip flip;
    board.board.reset();
    board.player = BLACK;
    for (int i = 0; i < (int)transcript.size(); i += 2){
        if (board.board.get_legal() == 0){
            board.board.pass();
            board.player ^= 1;
        }
        x = (int)(transcript[i] - 'a');
        if (x < 0 || x >= HW)
            x = (int)(transcript[i] - 'A');
        y = (int)(transcript[i + 1] - '1');
        board.policy = HW2_M1 - (y * HW + x);
        boards.emplace_back(board);
        calc_flip(&flip, &board.board, board.policy);
        if (flip.flip == 0ULL){
            std::cerr << "illegal move found in move " << i / 2 << " in " << transcript << std::endl;
            return;
        }
        board.board.move_board(&flip);
        board.player ^= 1;
    }
    if (board.board.get_legal()){
        std::cerr << "transcript not completed " << transcript << std::endl;
        return;
    }
    board.board.pass();
    board.player ^= 1;
    if (board.board.get_legal()){
        std::cerr << "transcript not completed " << transcript << std::endl;
        return;
    }
    int8_t score_black = board.board.score_player();
    if (board.player != BLACK){
        board.player ^= 1;
        score_black = -score_black;
    }
    if (mode == MODE_OUTPUT_OK && score_black == expected_score_black){
        std::cout << transcript << std::endl;
    } else if (mode == MODE_OUTPUT_ERROR && score_black != expected_score_black){
        std::cout << transcript << " " << (int)score_black << std::endl;
    }
}

int main(int argc, char* argv[]){
    if (argc < 6) {
        std::cerr << "input [in_dir] [start_file_num] [end_file_num] [expected_score_black] [0: output error 1: output OK]" << std::endl;
        return 1;
    }
    trs_init();
    std::string in_dir = std::string(argv[1]);
    int strt_file_num = atoi(argv[2]);
    int end_file_num = atoi(argv[3]);
    int expected_score_black = atoi(argv[4]);
    int mode = atoi(argv[5]);
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
            trs_convert_transcript(line, expected_score_black, mode);
            ++t;
        }
    }
    std::cerr << t << " transcript processed" << std::endl;
    return 0;
}