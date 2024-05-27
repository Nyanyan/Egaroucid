#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <ios>
#include <iomanip>
#include "./../../engine/board.hpp"
#include "./../../engine/util.hpp"

#define MAX_ENUMERATE_LINE 12


/*
    @brief array for calculating hash code for book
*/
size_t hash_rand_player_book[4][65536];
size_t hash_rand_opponent_book[4][65536];

/*
    @brief initialize hash array for book randomly
*/
void book_hash_init_rand(){
    int i, j;
    for (i = 0; i < 4; ++i){
        for (j = 0; j < 65536; ++j){
            hash_rand_player_book[i][j] = 0;
            while (pop_count_uint(hash_rand_player_book[i][j]) < 9)
                hash_rand_player_book[i][j] = myrand_ull();
            hash_rand_opponent_book[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent_book[i][j]) < 9)
                hash_rand_opponent_book[i][j] = myrand_ull();
        }
    }
}

/*
    @brief Hash function for book

    @param board                board
    @return hash code
*/
struct Book_hash {
    size_t operator()(Board board) const{
        const uint16_t *p = (uint16_t*)&board.player;
        const uint16_t *o = (uint16_t*)&board.opponent;
        return 
            hash_rand_player_book[0][p[0]] ^ 
            hash_rand_player_book[1][p[1]] ^ 
            hash_rand_player_book[2][p[2]] ^ 
            hash_rand_player_book[3][p[3]] ^ 
            hash_rand_opponent_book[0][o[0]] ^ 
            hash_rand_opponent_book[1][o[1]] ^ 
            hash_rand_opponent_book[2][o[2]] ^ 
            hash_rand_opponent_book[3][o[3]];
    }
};

std::unordered_set<Board, Book_hash> all_boards;
std::unordered_set<uint64_t> silhouette;
std::unordered_set<uint64_t> connected_seen;
int max_n_legal[4 + MAX_ENUMERATE_LINE + 1];
Board max_n_legal_board[4 + MAX_ENUMERATE_LINE + 1];

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

inline uint64_t get_representative_discs(uint64_t discs){
    discs = std::min(discs, black_line_mirror(discs));
    discs = std::min(discs, white_line_mirror(discs));
    discs = std::min(discs, horizontal_mirror(discs));
    discs = std::min(discs, vertical_mirror(discs));
    discs = std::min(discs, rotate_90(discs));
    discs = std::min(discs, rotate_180(discs));
    discs = std::min(discs, rotate_270(discs));
    return discs;
}

void enumerate(Board board, const int n_moves){
    int n_discs = board.n_discs();
    if (n_discs == n_moves + 4){
        uint64_t rdiscs = get_representative_discs(board.player | board.opponent);
        silhouette.emplace(rdiscs);
        uint64_t legal = board.get_legal();
        int n_legal = pop_count_ull(legal);
        if (max_n_legal[n_discs] < n_legal){
            max_n_legal[n_discs] = n_legal;
            max_n_legal_board[n_discs] = board;
        }
        return;
    }
    Board rboard = get_representative_board(board);
    if (all_boards.find(rboard) != all_boards.end()){
        return;
    }
    all_boards.emplace(rboard);
    if (n_discs < n_moves + 4){
        uint64_t legal = board.get_legal();
        if (legal == 0){
            board.pass();
            legal = board.get_legal();
            if (legal == 0){
                return;
            }
        }
        int n_legal = pop_count_ull(legal);
        if (max_n_legal[n_discs] < n_legal){
            max_n_legal[n_discs] = n_legal;
            max_n_legal_board[n_discs] = board;
        }
        Flip flip;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
                enumerate(board, n_moves);
            board.undo_board(&flip);
        }
    }
}

void get_silhouette(int n_moves){
    Board board;
    board.reset();
    uint64_t strt = tim();
    //std::cerr << "start!" << std::endl;
    enumerate(board, n_moves);
    //std::cerr << "finish in " << tim() - strt << " ms" << std::endl;
    //std::cerr << silhouette.size() << " boards found at depth " << n_moves << std::endl;
}








#define HW2 64

int max_n_connected = 0;
uint64_t max_connected_bits;

inline uint64_t calc_connected_cells(uint64_t discs){
    uint64_t hmask = discs & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = discs & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = discs & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << 8) | (vmask >> 8) | 
        (hvmask << 7) | (hvmask >> 7) | 
        (hvmask << 9) | (hvmask >> 9);
    return (~discs) & res;
}

inline uint64_t calc_puttable_cells(uint64_t discs){
    uint64_t hmask = discs & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = discs & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = discs & 0x007E7E7E7E7E7E00ULL;
    uint64_t hmask2 = discs & 0x3C3C3C3C3C3C3C3CULL;
    uint64_t vmask2 = discs & 0x0000FFFFFFFF0000ULL;
    uint64_t hvmask2 = discs & 0x00003C3C3C3C0000ULL;
    uint64_t res = 
        ((hmask << 1) & (hmask2 << 2)) | ((hmask >> 1) & (hmask2 >> 2)) | 
        ((vmask << 8) & (vmask2 << 16)) | ((vmask >> 8) & (vmask2 >> 16)) | 
        ((hvmask << 7) & (hvmask2 << 14)) | ((hvmask >> 7) & (hvmask2 >> 14)) | 
        ((hvmask << 9) & (hvmask2 << 18)) | ((hvmask >> 9) & (hvmask2 >> 18));
    return (~discs) & res;
}

void find_max_connected(uint64_t discs, int depth){
    if (depth == 0){
        int n_connected = pop_count_ull(calc_puttable_cells(discs));
        if (max_n_connected < n_connected){
            max_n_connected = n_connected;
            max_connected_bits = discs;
        }
        return;
    }
    uint64_t rdiscs = get_representative_discs(discs);
    if (connected_seen.find(rdiscs) != connected_seen.end()){
        return;
    }
    connected_seen.emplace(rdiscs);
    //int n_connected = pop_count_ull(calc_connected_cells(discs));
    //if (n_connected + 4 * depth <= max_n_connected){ // lazy cutoff
    //    return;
    //}
    uint64_t puttable = calc_puttable_cells(discs);
    //std::cerr << std::hex << discs << " " << puttable << std::endl;
    for (int cell = first_bit(&puttable); puttable; cell = next_bit(&puttable)){
        discs ^= 1ULL << cell;
            find_max_connected(discs, depth - 1);
        discs ^= 1ULL << cell;
    }
}

int main(){
    mobility_init();
    flip_init();
    book_hash_init_rand();
    for (int i = 4; i <= 4 + MAX_ENUMERATE_LINE; ++i){
        max_n_legal[i] = 0;
    }
    uint64_t strt_silhouette = tim();
    get_silhouette(MAX_ENUMERATE_LINE);
    std::cout << "silhouette done in " << tim() - strt_silhouette << " ms" << std::endl;
    for (int n_discs = 4; n_discs <= 4 + MAX_ENUMERATE_LINE; ++n_discs){
        std::cout << "n_discs " << std::dec << n_discs << " max_n_legal " << max_n_legal[n_discs] << " board " << std::hex << max_n_legal_board[n_discs].player << " " << max_n_legal_board[n_discs].opponent << std::endl;
    }
    for (int n_discs = 4 + MAX_ENUMERATE_LINE + 1; n_discs < HW2; ++n_discs){
        uint64_t strt = tim();
        max_n_connected = 0;
        connected_seen.clear();
        for (uint64_t discs: silhouette){
            find_max_connected(discs, n_discs - pop_count_ull(discs));
        }
        std::cout << "n_discs " << std::dec << n_discs << " max_n_connected " << max_n_connected << " bits " << std::hex << max_connected_bits << std::dec << " time " << tim() - strt << " ms" << std::endl;
    }
    std::cout << "finished" << std::endl;
}
