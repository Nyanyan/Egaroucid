#include "./../../engine/board.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "move_evaluation_definition_20230210_2.hpp"

struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
    double weight;
};

double adj_eval_arr[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<Adj_Data> adj_test_data;

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

void adj_import_eval(std::string file) {
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "evaluation file " << file << " not exist" << std::endl;
        return;
    }
    std::cerr << "importing eval params " << file << std::endl;
    std::string line;
    for (int pattern_idx = 0; pattern_idx < ADJ_N_EVAL; ++pattern_idx) {
        for (int pattern_elem = 0; pattern_elem < adj_eval_sizes[pattern_idx]; ++pattern_elem) {
            if (!getline(ifs, line)) {
                std::cerr << "ERROR evaluation file broken" << std::endl;
                return;
            }
            adj_eval_arr[pattern_idx][pattern_elem] = stof(line);
        }
    }
}

inline double adj_predict(uint16_t features[]) {
    double res = 0.0;
    for (int i = 0; i < ADJ_N_FEATURES - 16; ++i) {
        res += adj_eval_arr[adj_feature_to_eval_idx[i]][features[i]];
    }
    res /= ADJ_MO_SCORE_MAX;
    return res;
}

/*
// old version
inline double adj_predict(uint16_t features[]) {
    constexpr int cell_weight[10] = {18, 4, 16, 12, 2, 6, 8, 14, 10, 0};
    double res = 0.0;
    res += cell_weight[features[0]]; // cell weight
    res += 8 * features[1]; // parity
    res += -16 * features[3]; // n_mobility
    return res;
}
*/

void trs_convert_transcript(std::string transcript, int res[], int *n){
    int8_t y, x;
    std::vector<Trs_Convert_transcript_info> boards;
    Trs_Convert_transcript_info board_info;
    Flip flip;
    board_info.board.reset();
    board_info.player = BLACK;
    int n_discs = 4;
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
        if (64 - 13 <= n_discs && n_discs < 60 - 7)
            boards.emplace_back(board_info);
        calc_flip(&flip, &board_info.board, board_info.policy);
        if (flip.flip == 0ULL){
            std::cerr << "illegal move found in move " << i / 2 << " in " << transcript << std::endl;
            return;
        }
        board_info.board.move_board(&flip);
        board_info.player ^= 1;
        ++n_discs;
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
    double score;
    int rank;
    for (Trs_Convert_transcript_info &datum: boards){
        legal = datum.board.get_legal();
        std::vector<std::pair<double, uint_fast8_t>> scores;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            adj_calc_features(&datum.board, cell, eval_idxes);
            score = adj_predict(eval_idxes);
            scores.emplace_back(std::make_pair(score, cell));
        }
        std::sort(scores.begin(), scores.end());
        rank = 15;
        for (int i = (int)scores.size() - 1; i >= 0; --i){
            if (scores[i].second == datum.policy){
                rank = (int)scores.size() - 1 - i;
                break;
            }
        }
        ++(*n);
        ++res[rank];
    }
}

int main(int argc, char* argv[]){
    if (argc < 5){
        std::cerr << "input [in_dir] [start_file_num] [end_file_num] [eval_file]" << std::endl;
        return 1;
    }
    trs_init();
    std::string in_dir = std::string(argv[1]);
    int strt_file_num = atoi(argv[2]);
    int end_file_num = atoi(argv[3]);
    std::string eval_file = std::string(argv[4]);
    adj_import_eval(eval_file);
    int t = 0;
    int benchmark[16];
    for (int i = 0; i < 16; ++i)
        benchmark[i] = 0;
    int n = 0;
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
            trs_convert_transcript(line, benchmark, &n);
            ++t;
        }
    }
    std::cerr << std::endl;
    std::cout << t << " games found" << std::endl;
    std::cout << "n " << n << std::endl;
    for (int i = 0; i < 16; ++i){
        double rate = (double)benchmark[i] / n;
        std::cout << i << " " << benchmark[i] << " " << rate << std::endl;
    }
    return 0;
}