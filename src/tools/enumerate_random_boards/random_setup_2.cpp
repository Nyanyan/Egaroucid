#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <ios>
#include <iomanip>
#include <filesystem>
#include "./../../engine/board.hpp"

size_t hash_rand_player_enumerate[4][65536];
size_t hash_rand_opponent_enumerate[4][65536];

constexpr int RANDOM_SETUP_2_START_DISCS = 5;
constexpr int GGS_RANDOM_SETUP_2_MIN_DISCS = 10;
constexpr int GGS_RANDOM_SETUP_MAX_DISCS = 48;
constexpr uint64_t AVOID_CELLS = 0xC3C300000000C3C3ULL;

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

std::unordered_set<Board, Enumerate_hash> initial_boards;
std::unordered_set<Board, Enumerate_hash> all_boards;
std::unordered_set<Board, Enumerate_hash> searched_boards[HW2 + 1];
uint64_t n_searched_nodes[HW2 + 1];
uint64_t n_dead_ends = 0;
uint64_t n_rejected_by_disc_count = 0;
int target_n_discs = 0;

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

std::vector<int> get_possible_n_white_discs(int n_discs) {
    bool seen[HW2 + 1] = {};
    std::vector<int> bases;
    bases.emplace_back(n_discs / 2);
    if (n_discs & 1) {
        bases.emplace_back(n_discs / 2 + 1);
    }
    for (int base: bases) {
        int imb = base / 3;
        for (int delta = -imb; delta <= imb; ++delta) {
            int n_white = base + delta;
            if (0 <= n_white && n_white <= n_discs) {
                seen[n_white] = true;
            }
        }
    }
    std::vector<int> res;
    for (int n_white = 0; n_white <= n_discs; ++n_white) {
        if (seen[n_white]) {
            res.emplace_back(n_white);
        }
    }
    return res;
}

void register_initial_board(const uint64_t silhouette, const uint64_t white) {
    Board board;
    board.player = silhouette ^ white; // black
    board.opponent = white;
    board.pass(); // random_setup(5) leaves white to move
    initial_boards.emplace(board);
}

void fill_initial_discs_p(const uint64_t silhouette, uint64_t silhouette_white, int n_white_discs_remaining, uint64_t white) {
    if (n_white_discs_remaining == 0) {
        register_initial_board(silhouette, white);
        return;
    }
    if (silhouette_white) {
        uint64_t next_white_disc = 1ULL << ctz(silhouette_white);
        silhouette_white ^= next_white_disc;
        fill_initial_discs_p(silhouette, silhouette_white, n_white_discs_remaining, white);
        fill_initial_discs_p(silhouette, silhouette_white, n_white_discs_remaining - 1, white ^ next_white_disc);
    }
}

void fill_initial_discs(uint64_t silhouette) {
    for (int n_white: get_possible_n_white_discs(RANDOM_SETUP_2_START_DISCS)) {
        fill_initial_discs_p(silhouette, silhouette, n_white, 0ULL);
    }
}

void fill_initial_silhouette(uint64_t silhouette, const std::vector<int> &cell_list, int cell_list_strt_idx, int n_filled) {
    if (n_filled == 0) {
        fill_initial_discs(silhouette);
        return;
    }
    for (int i = cell_list_strt_idx; i < (int)cell_list.size(); ++i) {
        silhouette ^= 1ULL << cell_list[i];
        fill_initial_silhouette(silhouette, cell_list, i + 1, n_filled - 1);
        silhouette ^= 1ULL << cell_list[i];
    }
}

void enumerate_initial_boards() {
    std::vector<std::vector<int>> cell_distance_list;
    for (int i = 0; i < HW; ++i) {
        cell_distance_list.emplace_back(std::vector<int>());
    }
    for (int cell = 0; cell < HW2; ++cell) {
        if ((AVOID_CELLS & (1ULL << cell)) == 0) {
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
        if (n_filled + cell_distance_list[d].size() <= RANDOM_SETUP_2_START_DISCS) {
            for (int &cell: cell_distance_list[d]) {
                always_filled_cells.emplace_back(cell);
            }
            n_filled += cell_distance_list[d].size();
        } else if (n_filled < RANDOM_SETUP_2_START_DISCS) {
            n_randomly_filled = RANDOM_SETUP_2_START_DISCS - n_filled;
            for (int &cell: cell_distance_list[d]) {
                randomly_filled_cells.emplace_back(cell);
            }
            n_filled = RANDOM_SETUP_2_START_DISCS;
        }
    }

    uint64_t silhouette = 0ULL;
    for (const int &cell: always_filled_cells) {
        silhouette ^= 1ULL << cell;
    }
    fill_initial_silhouette(silhouette, randomly_filled_cells, 0, n_randomly_filled);
}

inline int get_turn_color(int n_discs) {
    return (n_discs & 1) ? WHITE : BLACK;
}

void register_final_board(Board board) {
    const int turn_color = get_turn_color(target_n_discs);
    Board black_to_move_board = board;
    if (turn_color == WHITE) {
        black_to_move_board.pass();
    }

    const int n_black = black_to_move_board.count_player();
    const int n_white = black_to_move_board.count_opponent();
    const int min_count = std::max(target_n_discs / 4, 3);
    if (n_black <= min_count || n_white <= min_count) {
        ++n_rejected_by_disc_count;
        return;
    }

    all_boards.emplace(get_representative_board(black_to_move_board));
}

void enumerate_random_setup_2(Board board) {
    const int n_discs = board.n_discs();
    Board representative_board = get_representative_board(board);
    if (searched_boards[n_discs].find(representative_board) != searched_boards[n_discs].end()) {
        return;
    }
    searched_boards[n_discs].emplace(representative_board);
    ++n_searched_nodes[n_discs];

    if (n_discs == target_n_discs) {
        register_final_board(board);
        return;
    }

    uint64_t legal = board.get_legal() & ~AVOID_CELLS;
    if (legal == 0ULL) {
        ++n_dead_ends;
        return;
    }

    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            enumerate_random_setup_2(board);
        board.undo_board(&flip);
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

    target_n_discs = atoi(argv[1]);
    if (target_n_discs < GGS_RANDOM_SETUP_2_MIN_DISCS || target_n_discs > GGS_RANDOM_SETUP_MAX_DISCS) {
        std::cerr << "random_setup_2 is used by GGS for n_discs from "
                  << GGS_RANDOM_SETUP_2_MIN_DISCS << " to " << GGS_RANDOM_SETUP_MAX_DISCS << std::endl;
        return 1;
    }

    mobility_init();
    flip_init();
    enumerate_hash_init_rand();

    uint64_t strt = tim();
    enumerate_initial_boards();
    std::cerr << "initial boards " << initial_boards.size() << std::endl;

    for (const Board &board: initial_boards) {
        enumerate_random_setup_2(board);
    }

    for (int n_discs = RANDOM_SETUP_2_START_DISCS; n_discs <= target_n_discs; ++n_discs) {
        std::cerr << "n_discs " << n_discs << " searched " << n_searched_nodes[n_discs] << std::endl;
    }
    std::cerr << "dead ends " << n_dead_ends << std::endl;
    std::cerr << "rejected by disc count " << n_rejected_by_disc_count << std::endl;
    std::cerr << "all boards " << all_boards.size() << " in " << tim() - strt << " ms" << std::endl;

    int turn_color = get_turn_color(target_n_discs);
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

    const std::string output_dir = "output/" + std::to_string(target_n_discs) + "_autoplay";
    std::filesystem::create_directories(output_dir);

    int n_files = (all_boards.size() + N_BOARDS_PER_FILE - 1) / N_BOARDS_PER_FILE;
    std::cerr << "n_files " << n_files << std::endl;
    int global_idx = 0;
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::string file = output_dir + "/" + fill0(file_idx, 7) + ".txt";
        std::cerr << "output file " << file << std::endl;
        std::ofstream ofs(file);
        for (int i = 0; i < N_BOARDS_PER_FILE && global_idx < (int)all_boards_vector.size(); ++i) {
            std::string board_str = get_str(all_boards_vector[global_idx], turn_color);
            ofs << board_str << std::endl;
            ++global_idx;
        }
    }

    std::cerr << "all done in " << tim() - strt << " ms" << std::endl;

    return 0;
}
