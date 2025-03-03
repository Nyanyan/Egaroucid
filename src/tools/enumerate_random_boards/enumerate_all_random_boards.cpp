#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <ios>
#include <iomanip>
#include "./../../engine/engine_all.hpp"
//#include "./../../engine/util.hpp"

size_t hash_rand_player_enumerate[4][65536];
size_t hash_rand_opponent_enumerate[4][65536];

void enumerate_hash_init_rand(){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player_enumerate[i][j] = 0;
            while (pop_count_uint(hash_rand_player_enumerate[i][j]) < 9)
                hash_rand_player_enumerate[i][j] = myrand_ull();
            hash_rand_opponent_enumerate[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent_enumerate[i][j]) < 9)
                hash_rand_opponent_enumerate[i][j] = myrand_ull();
        }
    }
}

/*
    @brief Hash function

    @param board                board
    @return hash code
*/
struct Enumerate_hash {
    size_t operator()(Board board) const{
        const uint16_t *p = (uint16_t*)&board.player;
        const uint16_t *o = (uint16_t*)&board.opponent;
        return 
            hash_rand_player_enumerate[0][p[0]] ^ 
            hash_rand_player_enumerate[1][p[1]] ^ 
            hash_rand_player_enumerate[2][p[2]] ^ 
            hash_rand_player_enumerate[3][p[3]] ^ 
            hash_rand_opponent_enumerate[0][o[0]] ^ 
            hash_rand_opponent_enumerate[1][o[1]] ^ 
            hash_rand_opponent_enumerate[2][o[2]] ^ 
            hash_rand_opponent_enumerate[3][o[3]];
    }
};

std::unordered_set<uint64_t> all_silhouettes;
std::unordered_set<Board, Enumerate_hash> all_boards;

inline void first_update_representative_board(Board *res, Board *sym){
    uint64_t vp = vertical_mirror(sym->player);
    uint64_t vo = vertical_mirror(sym->opponent);
    if (res->player > vp || (res->player == vp && res->opponent > vo)){
        res->player = vp;
        res->opponent = vo;
    }
}

inline void update_representative_board(Board *res, Board *sym){
    if (res->player > sym->player || (res->player == sym->player && res->opponent > sym->opponent))
        sym->copy(res);
    uint64_t vp = vertical_mirror(sym->player);
    uint64_t vo = vertical_mirror(sym->opponent);
    if (res->player > vp || (res->player == vp && res->opponent > vo)){
        res->player = vp;
        res->opponent = vo;
    }
}

inline uint64_t get_representative_silhouette(uint64_t silhouette){
    uint64_t res = silhouette;
    uint64_t mirrored = vertical_mirror(silhouette);
    res = std::min(res, mirrored);
    //res = std::min(res, vertical_mirror(silhouette));
    
    silhouette = black_line_mirror(silhouette);
    res = std::min(res, silhouette);
    mirrored = vertical_mirror(silhouette);
    res = std::min(res, mirrored);
    //res = std::min(res, vertical_mirror(silhouette));

    silhouette = horizontal_mirror(silhouette);
    res = std::min(res, silhouette);
    mirrored = vertical_mirror(silhouette);
    res = std::min(res, mirrored);
    //res = std::min(res, vertical_mirror(silhouette));

    silhouette = white_line_mirror(silhouette);
    res = std::min(res, silhouette);
    mirrored = vertical_mirror(silhouette);
    res = std::min(res, mirrored);
    //res = std::min(res, vertical_mirror(silhouette));

    return res;

}

inline Board get_representative_board(Board b){
    Board res = b;
    first_update_representative_board(&res, &b);
    b.board_black_line_mirror();
    update_representative_board(&res, &b);
    b.board_horizontal_mirror();
    update_representative_board(&res, &b);
    b.board_white_line_mirror();
    update_representative_board(&res, &b);
    return res;
}

std::string fill0(int n, int d){
    std::stringstream ss;
    ss << std::setfill('0') << std::right << std::setw(d) << n;
    return ss.str();
}

void fill_silhouette(uint64_t silhouette, const std::vector<int> &cell_list, int cell_list_strt_idx, int n_filled) {
    if (n_filled == 0) { // silhouette filled
        //all_silhouettes.emplace(silhouette);
        all_silhouettes.emplace(get_representative_silhouette(silhouette));
    }
    for (int i = cell_list_strt_idx; i < (int)cell_list.size(); ++i) {
        silhouette ^= 1ULL << cell_list[i];
            fill_silhouette(silhouette, cell_list, i + 1, n_filled - 1);
        silhouette ^= 1ULL << cell_list[i];
    }
}

void fill_discs_p(const uint64_t silhouette, uint64_t silhouette_white, int n_white_discs_remaining, uint64_t white) {
    //std::cerr << n_white_discs_remaining << " " << pop_count_ull(silhouette_white) << std::endl;
    if (n_white_discs_remaining == 0) {
        Board board;
        board.player = silhouette ^ white;
        board.opponent = white;
        Board rboard = get_representative_board(board);
        all_boards.emplace(rboard);
        return;
    }
    if (silhouette_white) {
        uint64_t next_white_disc = 1ULL << ctz(silhouette_white);
        silhouette_white ^= next_white_disc;
        fill_discs_p(silhouette, silhouette_white, n_white_discs_remaining, white); // next_white_disc = 0
        fill_discs_p(silhouette, silhouette_white, n_white_discs_remaining - 1, white ^ next_white_disc); // next_white_disc = 1
    }
}

void fill_discs(uint64_t silhouette, int n_discs, int n_white_min_discs, int n_white_max_discs) {
    for (int n_white = n_white_min_discs; n_white <= n_white_max_discs; ++n_white) {
        fill_discs_p(silhouette, silhouette, n_white, 0);
    }
}

std::string get_str(Board board, int turn_color) {
    std::string res;
    for (int cell = HW2 - 1; cell >= 0; --cell) {
        uint64_t cell_bit = 1ULL << cell;
        if (turn_color == BLACK) {
            if (board.player & cell_bit) {
                res += "X";
            } else if (board.opponent & cell_bit) {
                res += "O";
            } else {
                res += "-";
            }
        } else {
            if (board.player & cell_bit) {
                res += "O";
            } else if (board.opponent & cell_bit) {
                res += "X";
            } else {
                res += "-";
            }
        }
    }
    res += " ";
    if (turn_color == BLACK) {
        res += "X";
    } else {
        res += "O";
    }
    return res;
}

#define N_BOARDS_PER_FILE 10000

int main(int argc, char *argv[]){
    if (argc < 2){
        std::cerr << "input [n_discs]" << std::endl;
        return 1;
    }

    mobility_init();
    flip_init();
    enumerate_hash_init_rand();

    int n_discs = atoi(argv[1]);

    if (n_discs < 4) {
        n_discs = 4;
    }

    uint64_t avoid_cells = 0xC3C300000000C3C3ULL;

    std::vector<std::vector<int>> cell_distance_list;
    for (int i = 0; i < HW; ++i) {
        cell_distance_list.emplace_back(std::vector<int>());
    }
    for (int cell = 0; cell < HW2; ++cell) {
        if ((avoid_cells & (1ULL << cell)) == 0) {
            int x = cell % HW;
            int y = cell / HW;
            int d = (std::max(std::abs(2 * x - (HW - 1)), std::abs(2 * y - (HW - 1))) - 1) / 2;
            cell_distance_list[d].emplace_back(cell);
        }
    }
    std::vector<int> always_filled_cells;
    std::vector<int> randomly_filled_cells;
    int n_randomly_filled = 0;
    int n_filled = 0;
    for (int d = 0; d < HW; ++d) {
        if (n_filled + cell_distance_list[d].size() <= n_discs) {
            for (int &cell: cell_distance_list[d]) {
                always_filled_cells.emplace_back(cell);
            }
            n_filled += cell_distance_list[d].size();
        } else if (n_filled < n_discs) {
            n_randomly_filled = n_discs - n_filled;
            for (int &cell: cell_distance_list[d]) {
                randomly_filled_cells.emplace_back(cell);
            }
            n_filled = n_discs;
        }
    }
    std::cerr << "always filled " << always_filled_cells.size() << " : ";
    for (const int &cell: always_filled_cells) {
        std::cerr << idx_to_coord(cell) << " ";
    }
    std::cerr << std::endl;
    std::cerr << "randomly filled " << n_randomly_filled << " / " << randomly_filled_cells.size() <<  " : ";
    for (const int &cell: randomly_filled_cells) {
        std::cerr << idx_to_coord(cell) << " ";
    }
    std::cerr << std::endl;

    uint64_t silhouette = 0ULL;
    for (const int &cell: always_filled_cells) {
        silhouette ^= 1ULL << cell;
    }
    bit_print_board(silhouette);

    fill_silhouette(silhouette, randomly_filled_cells, 0, n_randomly_filled);
    std::cerr << "n_silhouettes " << all_silhouettes.size() << std::endl;

    int d2_n_discs = n_discs / 2;
    int imb = d2_n_discs / 3;
    int n_white_min_discs = d2_n_discs - imb;
    int n_white_max_discs = d2_n_discs + imb;
    if (n_discs % 2 == 1) {
        ++d2_n_discs;
        imb = d2_n_discs / 3;
        n_white_min_discs = std::min(n_white_min_discs, d2_n_discs - imb);
        n_white_max_discs = std::min(n_white_max_discs, d2_n_discs + imb);
    }
    std::cerr << "white min_discs " << n_white_min_discs << " max_discs " << n_white_max_discs << std::endl;
    for (uint64_t sil: all_silhouettes) {
        size_t pre = all_boards.size();
        fill_discs(sil, n_discs, n_white_min_discs, n_white_max_discs);
        std::cerr << all_boards.size() - pre << std::endl;
        bit_print_board(sil);
    }
    std::cerr << "all boards " << all_boards.size() << std::endl;

    int turn_color = (n_discs % 2) ? WHITE : BLACK;
    std::cerr << "turn_color " << turn_color << std::endl;
    

    std::vector<Board> all_boards_vector;
    for (const Board &board: all_boards) {
        all_boards_vector.emplace_back(board);
    }
    if (turn_color == WHITE) {
        for (int i = 0; i < (int)all_boards_vector.size(); ++i) {
            all_boards_vector[i].pass();
        }
    }
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::shuffle(all_boards_vector.begin(), all_boards_vector.end(), engine);

    int n_files = (all_boards.size() + N_BOARDS_PER_FILE - 1) / N_BOARDS_PER_FILE;
    std::cerr << "n_files " << n_files << std::endl;
    int global_idx = 0;
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::string file = "output/" + std::to_string(n_discs) + "/" + fill0(file_idx, 7) + ".txt";
        std::cerr << "output file " << file << std::endl;
        std::ofstream ofs(file);
        for (int i = 0; i < N_BOARDS_PER_FILE && global_idx < (int)all_boards_vector.size(); ++i) {
            std::string board_str = get_str(all_boards_vector[global_idx], turn_color);
            ofs << board_str << std::endl;
            ++global_idx;
        }
    }

    /*
    if (turn_color == WHITE) {
        std::swap(board.player, board.opponent);
    }
    */

    return 0;
}