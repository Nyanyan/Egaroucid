#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdint>

inline bool file_open(FILE **fp, const char *file, const char *mode){
    #ifdef _WIN64
        return fopen_s(fp, file, mode) == 0;
    #elif _WIN32
        return fopen_s(fp, file, mode) == 0;
    #else
        *fp = fopen(file, mode);
        return *fp != NULL;
    #endif
}

bool translate_data(std::string book_file, std::string out_file){
    FILE* fp;
    if (!file_open(&fp, book_file.c_str(), "rb")){
        std::cerr << "[ERROR] can't open Egaroucid book " << book_file << std::endl;
        return false;
    }
    std::ofstream fout;
    fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
    int n_boards;
    char value, level, leaf_value, leaf_move, leaf_level;
    uint32_t n_lines;
    uint64_t p, o;
    char egaroucid_str[10];
    char egaroucid_str_ans[] = "DICUORAGE";
    char elem_char;
    char book_version;
    // Header
    if (fread(egaroucid_str, 1, 9, fp) < 9) {
        std::cerr << "[ERROR] file broken" << std::endl;
        fclose(fp);
        return false;
    }
    for (int i = 0; i < 9; ++i){
        if (egaroucid_str[i] != egaroucid_str_ans[i]){
            std::cerr << "[ERROR] This is not Egarocuid book, found " << egaroucid_str[i] << ", " << (int)egaroucid_str[i] << " at char " << i << ", expected " << egaroucid_str_ans[i] << " , " << (int)egaroucid_str_ans[i] << std::endl;
            fclose(fp);
            return false;
        }
    }
    if (fread(&book_version, 1, 1, fp) < 1) {
        std::cerr << "[ERROR] file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (book_version != 3){
        std::cerr << "[ERROR] This is not Egarocuid book version 3, found version " << (int)book_version << std::endl;
        fclose(fp);
        return false;
    }
    // Book Information
    if (fread(&n_boards, 4, 1, fp) < 1){
        std::cerr << "[ERROR] book broken at n_book data" << std::endl;
        fclose(fp);
        return false;
    }
    std::cerr << n_boards << " boards to read" << std::endl;
    // for each board
    int percent = -1;
    char player_fixed = 0; // BLACK
    char policy_fixed = 99;
    for (int i = 0; i < n_boards; ++i) {
        int n_percent = (double)i / n_boards * 100;
        // if (n_percent > percent){
        //     percent = n_percent;
        //     std::cerr << "loading book " << percent << "%" << std::endl;
        // }
        // read board, player
        if (fread(&p, 8, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read board, opponent
        if (fread(&o, 8, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // board error check
        if (p & o){
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read value
        if (fread(&value, 1, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read level
        if (fread(&level, 1, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read n_lines
        if (fread(&n_lines, 4, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read leaf value
        if (fread(&leaf_value, 1, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read leaf move
        if (fread(&leaf_move, 1, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        // read leaf level
        if (fread(&leaf_level, 1, 1, fp) < 1) {
            std::cerr << "ERR" << std::endl;
            fclose(fp);
            return false;
        }
        if (std::popcount(p | o) == 8){
            std::cerr << (int)value << std::endl;
            for (int j = 0; j < 64; ++j){
                if (1 & (p >> (63 - j))){
                    std::cerr << "X ";
                } else if (1 & (o >> (63 - j))){
                    std::cerr << "O ";
                } else{
                    std::cerr << ". ";
                }
                if (j % 8 == 7){
                    std::cerr << std::endl;
                }
            }
        }
        // push elem
        fout.write((char*)&p, 8);
        fout.write((char*)&o, 8);
        fout.write((char*)&player_fixed, 1);
        fout.write((char*)&policy_fixed, 1);
        fout.write((char*)&value, 1);
    }
    return true;
}

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "input [book_file] [out_file]" << std::endl;
        return 1;
    }
    std::string book_file = std::string(argv[1]);
    std::string out_file = std::string(argv[2]);
    if (translate_data(book_file, out_file)){
        std::cerr << "completed!" << std::endl;
    }
    return 0;
}