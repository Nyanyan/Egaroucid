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

void trs_convert_board(std::string line, std::ofstream *fout){
    Trs_Convert_transcript_info board;
    board.board.player = 0ULL;
    board.board.opponent = 0ULL;
    for (int i = 0; i < HW2; ++i){
        if (line[i] == 'o'){
            board.board.player |= 1ULL << (HW2_M1 - i);
        } else if (line[i] == 'x'){
            board.board.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    board.player = line[65] == '0' ? BLACK : WHITE;
    if (board.player == WHITE){
        board.board.pass();
    }
    board.policy = HW2;
    int8_t score = stoi(line.substr(67));
    fout->write((char*)&(board.board.player), 8);
    fout->write((char*)&(board.board.opponent), 8);
    fout->write((char*)&(board.player), 1);
    fout->write((char*)&(board.policy), 1);
    fout->write((char*)&score, 1);
}

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "input [out_file] [files]" << std::endl;
        return 1;
    }
    trs_init();
    std::string out_file = std::string(argv[1]);
    int n_files = argc - 2;
    std::ofstream fout;
    fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }
    int t = 0;
    for (int file_num = 0; file_num < n_files; ++file_num){
        std::cerr << "=";
        std::ifstream ifs(argv[2 + file_num]);
        if (!ifs) {
            std::cerr << "can't open " << argv[2 + file_num] << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(ifs, line)){
            trs_convert_board(line, &fout);
            ++t;
        }
    }
    std::cerr << std::endl;
    std::cout << t << " boards found" << std::endl;
    return 0;
}