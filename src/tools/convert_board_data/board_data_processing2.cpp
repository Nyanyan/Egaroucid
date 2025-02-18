// for Egaroucid_Train_Data.zip

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
        if (line[i] == 'X'){
            board.board.player |= 1ULL << (HW2_M1 - i);
        } else if (line[i] == 'O'){
            board.board.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    board.policy = HW2;
    int8_t score = stoi(line.substr(65));
    fout->write((char*)&(board.board.player), 8);
    fout->write((char*)&(board.board.opponent), 8);
    fout->write((char*)&(board.player), 1);
    fout->write((char*)&(board.policy), 1);
    fout->write((char*)&score, 1);
}

int main(int argc, char* argv[]){
    if (argc < 5){
        std::cerr << "input [in_dir] [start_file_num] [end_file_num] [out_file]" << std::endl;
        return 1;
    }
    trs_init();
    std::string in_dir = std::string(argv[1]);
    int strt = atoi(argv[2]);
    int end = atoi(argv[3]);
    std::string out_file = std::string(argv[4]);
    std::ofstream fout;
    fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open" << std::endl;
        return 1;
    }
    int t = 0;
    for (int file_num = strt; file_num < end; ++file_num){
        std::cerr << "=";
        std::string file = in_dir + trs_fill0(file_num, 7) + ".txt";
        std::ifstream ifs(file);
        if (!ifs) {
            std::cerr << "can't open " << file << std::endl;
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