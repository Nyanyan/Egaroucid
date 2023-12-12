/*
    Egaroucid Project

    @file book.hpp
        Book class
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "search.hpp"
#include "ai.hpp"

inline Search_result tree_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, bool use_multi_thread);
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);
Search_result ai_specified_moves(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal);

#define BOOK_N_ACCEPT_LEVEL 11
#define BOOK_ACCURACY_LEVEL_INF 10
#define BOOK_LEAF_LEVEL 5

#define LEVEL_UNDEFINED -1
#define LEVEL_HUMAN 70
#define BOOK_LOSS_IGNORE_THRESHOLD 8
#define LEAF_CALCULATE_LEVEL 5

#define FORCE_BOOK_LEVEL false
#define FORCE_BOOK_DEPTH false

#define BOOK_EXTENSION ".egbk3"

#define ADD_LEAF_SPECIAL_LEVEL -1

#define MAX_N_LINES 4000000000

/*
    @brief book result structure

    @param policy               selected best move
    @param value                registered score
*/
struct Book_value{
    int policy;
    int value;

    Search_result to_search_result(){
        Search_result res;
        res.policy = policy;
        res.value = value;
        res.depth = SEARCH_BOOK;
        res.time = 0;
        res.nodes = 0;
        res.clog_time = 0;
        res.clog_nodes = 0;
        res.nps = 0;
        res.is_end_search - false;
        res.probability = -1;
        return res;
    }
};

struct Leaf{
    int8_t value;
    int8_t move;

    Leaf(){
        value = SCORE_UNDEFINED;
        move = -1;
    }
};

/*
    @brief book element structure

    @param value                registered score
    @param level                AI level
    @param moves                each moves and values
*/
struct Book_elem{
    int8_t value;
    Leaf leaf;
    uint32_t n_lines;
    bool seen; // used in various situation

    Book_elem(){
        value = SCORE_UNDEFINED;
        n_lines = 0;
        seen = false;
    }
};

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
    @brief initialize hash array for book
*/
void book_hash_init(bool show_log){
    FILE* fp;
    if (!file_open(&fp, "resources/hash_book.eghs", "rb")){
        std::cerr << "[ERROR] can't open hash_book.eghs" << std::endl;
        book_hash_init_rand();
        return;
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_player_book[i], 8, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash_book.eghs broken" << std::endl;
            book_hash_init_rand();
            return;
        }
    }
    for (int i = 0; i < 4; ++i){
        if (fread(hash_rand_opponent_book[i], 8, 65536, fp) < 65536){
            std::cerr << "[ERROR] hash_book.eghs broken" << std::endl;
            book_hash_init_rand();
            return;
        }
    }
    if (show_log)
        std::cerr << "hash for book initialized" << std::endl;
    return;
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







/*
    @brief book data

    @param book                 book data
*/
class Book{
    private:
        std::mutex mtx;
        std::unordered_map<Board, Book_elem, Book_hash> book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading){
            delete_all();
            if (!import_file_egbk3(file, show_log, stop_loading)){ // try egbk3 format
                std::cerr << "failed egbk3 formatted book. trying egbk2 format." << std::endl;
                if (!import_file_egbk2(file, show_log, stop_loading)){ // try egbk2 format
                std::cerr << "failed egbk2 formatted book. trying egbk format." << std::endl;
                    return import_file_egbk(file, show_log, stop_loading); // try egbk format
                }
            }
            return true;
        }

        inline bool import_book_extension_determination(std::string file, bool *stop){
            bool result = false;
            std::vector<std::string> lst;
            auto offset = std::string::size_type(0);
            while (1) {
                auto pos = file.find(".", offset);
                if (pos == std::string::npos) {
                    lst.push_back(file.substr(offset));
                    break;
                }
                lst.push_back(file.substr(offset, pos - offset));
                offset = pos + 1;
            }
            if (lst[lst.size() - 1] == "egbk3") {
                std::cerr << "importing Egaroucid book (.egbk3)" << std::endl;
                result = import_file_egbk3(file, true, stop);
            }
            if (lst[lst.size() - 1] == "egbk2") {
                std::cerr << "importing Egaroucid legacy book (.egbk2)" << std::endl;
                result = import_file_egbk2(file, true, stop);
            }
            else if (lst[lst.size() - 1] == "egbk") {
                std::cerr << "importing Egaroucid legacy book (.egbk)" << std::endl;
                result = import_file_egbk(file, true, stop);
            }
            else if (lst[lst.size() - 1] == "dat") {
                std::cerr << "importing Edax book" << std::endl;
                result = import_file_edax(file, true, stop);
            }
            else {
                std::cerr << "this is not a book" << std::endl;
            }
            return result;
        }

        inline bool import_book_extension_determination(std::string file){
            bool stop = false;
            return import_book_extension_determination(file, &stop);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk3 file)
            @return book completely imported?
        */
        inline bool import_file_egbk3(std::string file, bool show_log, bool *stop_loading){
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value, leaf_value, leaf_move;
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
            if (show_log)
                std::cerr << n_boards << " boards to read" << std::endl;
            // for each board
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log){
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board, player
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 0 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board, opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 1 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // board error check
                if (p & o){
                    std::cerr << "[ERROR] book NOT FULLY imported 2 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 3 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                //if (value < -HW2 || HW2 < value) {
                //    std::cerr << "[ERROR] book NOT FULLY imported 4 got value " << (int)value << " " << book.size() << " boards" << std::endl;
                //    fclose(fp);
                //    return false;
                //    //std::cerr << "[WARNING] value error found " << (int)value << " " << book.size() << " boards" << std::endl;
                //    //value = SCORE_UNDEFINED;
                //}
                // read n_lines
                if (fread(&n_lines, 4, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 5 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf value
                if (fread(&leaf_value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 6 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf move
                if (fread(&leaf_move, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 7 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // push elem
                if (-HW2 <= value && value <= HW2) {
                    board.player = p;
                    board.opponent = o;
                    #if FORCE_BOOK_DEPTH
                        if (board.n_discs() <= 4 + 30){
                    #endif
                            book_elem.value = value;
                            book_elem.leaf.value = leaf_value;
                            book_elem.leaf.move = leaf_move;
                            book_elem.n_lines = n_lines;
                            merge(board, book_elem);
                    #if FORCE_BOOK_DEPTH
                        }
                    #endif
                }
            }
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk3(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_egbk3(file, show_log, &stop_loading);
        }


        void add_leaf(Board *board, int8_t value, int8_t policy){
            int rotate_idx;
            Board representive_board = get_representative_board(board, &rotate_idx);
            int8_t rotated_policy = policy;
            if (is_valid_policy(policy))
                rotated_policy = convert_coord_from_representative_board((int)policy, rotate_idx);
            Leaf leaf;
            leaf.value = value;
            leaf.move = rotated_policy;
            book[representive_board].leaf = leaf;
        }

        void search_leaf(Board board, int level){
            Book_elem book_elem = book[board];
            int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            uint64_t legal = board.get_legal();
            for (Book_value &link: links)
                legal ^= 1ULL << link.policy;
            if (legal){
                Search_result ai_result = ai_specified_moves(board, level, false, 0, true, false, legal);
                if (ai_result.value != SCORE_UNDEFINED){
                    new_leaf_value = ai_result.value;
                    if (level == ADD_LEAF_SPECIAL_LEVEL)
                        new_leaf_value = book_elem.value;
                    new_leaf_move = ai_result.policy;
                }
            } else
                new_leaf_move = MOVE_NOMOVE;
            add_leaf(&board, new_leaf_value, new_leaf_move);
        }

        void check_add_leaf_all_undefined(){
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                boards.emplace_back(itr->first);
            Flip flip;
            for (Board &board: boards){
                int leaf_move = book[board].leaf.move;
                bool need_to_rewrite_leaf = leaf_move < 0 || MOVE_UNDEFINED <= leaf_move;
                if (!need_to_rewrite_leaf){
                    calc_flip(&flip, &board, leaf_move);
                    board.move_board(&flip);
                        need_to_rewrite_leaf = contain(&board);
                    board.undo_board(&flip);
                }
                if (need_to_rewrite_leaf){
                    int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
                    add_leaf(&board, new_leaf_value, new_leaf_move);
                }
            }
        }

        void check_add_leaf_all_search(int level, bool *stop){
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                boards.emplace_back(itr->first);
            Flip flip;
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1, t = 0, n_boards = (int)boards.size();
            Book_elem book_elem;
            for (Board &board: boards){
                if (!global_searching || *stop)
                    break;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent){
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << std::endl;
                }
                ++t;
                book_elem = book[board];
                int leaf_move = book_elem.leaf.move;
                bool need_to_rewrite_leaf = leaf_move < 0 || MOVE_UNDEFINED <= leaf_move || (board.get_legal() & (1ULL << leaf_move)) == 0;
                if (!need_to_rewrite_leaf){
                    calc_flip(&flip, &board, leaf_move);
                    board.move_board(&flip);
                        if (board.get_legal() == 0){
                            board.pass();
                                need_to_rewrite_leaf = contain(&board);
                            board.pass();
                        } else
                            need_to_rewrite_leaf = contain(&board);
                    board.undo_board(&flip);
                }
                if (need_to_rewrite_leaf){
                    int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
                    std::vector<Book_value> links = get_all_moves_with_value(&board);
                    uint64_t legal = board.get_legal();
                    for (Book_value &link: links)
                        legal ^= 1ULL << link.policy;
                    if (legal){
                        int use_level = level;
                        if (level == ADD_LEAF_SPECIAL_LEVEL)
                            use_level = 1;
                        Search_result ai_result = ai_specified_moves(board, use_level, false, 0, true, false, legal);
                        if (ai_result.value != SCORE_UNDEFINED){
                            new_leaf_value = ai_result.value;
                            if (level == ADD_LEAF_SPECIAL_LEVEL)
                                new_leaf_value = book_elem.value;
                            new_leaf_move = ai_result.policy;
                            //std::cerr << "recalc leaf " << (int)new_leaf_value << " " << (int)new_leaf_move << " " << idx_to_coord(new_leaf_move) << std::endl;
                            //for (Book_value &link: links)
                            //    std::cerr << "link " << idx_to_coord(link.policy) << std::endl;
                            //board.print();
                        }
                    } else
                        new_leaf_move = MOVE_NOMOVE;
                    add_leaf(&board, new_leaf_value, new_leaf_move);
                }
            }
        }

        void recalculate_leaf_all(int level, bool *stop){
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                boards.emplace_back(itr->first);
            Flip flip;
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1, t = 0, n_boards = (int)boards.size();
            for (Board &board: boards){
                if (!global_searching || *stop)
                    break;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent){
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << std::endl;
                }
                ++t;
                search_leaf(board, level);
            }
        }

        /*
            @brief import Egaroucid-formatted book (old)

            @param file                 book file (.egbk2 file)
            @return book completely imported?
        */
        inline bool import_file_egbk2(std::string file, bool show_log, bool *stop_loading){
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value;
            uint64_t p, o;
            char level, n_moves, val, mov;
            char egaroucid_str[10];
            char egaroucid_str_ans[] = "DICUORAGE";
            char book_version;
            // Header
            if (fread(egaroucid_str, 1, 9, fp) < 9) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            for (int i = 0; i < 9; ++i){
                if (egaroucid_str[i] != egaroucid_str_ans[i]){
                    std::cerr << "[ERROR] This is not Egarocuid book version 2, found " << egaroucid_str[i] << ", " << (int)egaroucid_str[i] << " at char " << i << ", expected " << egaroucid_str_ans[i] << " , " << (int)egaroucid_str_ans[i] << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&book_version, 1, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            if (book_version != 2){
                std::cerr << "[ERROR] This is not Egarocuid book version 2, found version " << (int)book_version << std::endl;
                fclose(fp);
                return false;
            }
            // Book Information
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << n_boards << " boards to read" << std::endl;
            // for each board
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log){
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read level (ignore this in egbk3)
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read n_links
                if (fread(&n_moves, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // for each link
                for (uint8_t i = 0; i < n_moves; ++i){
                    // read value
                    if (fread(&val, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    // read move
                    if (fread(&mov, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    // ignore links...
                }
                board.player = p;
                board.opponent = o;
                #if FORCE_BOOK_DEPTH
                    if (b.n_discs() <= 4 + 30){
                #endif
                        book_elem.value = value;
                        book_elem.leaf.value = SCORE_UNDEFINED;
                        book_elem.leaf.move = MOVE_UNDEFINED;
                        merge(board, book_elem);
                #if FORCE_BOOK_DEPTH
                    }
                #endif
            }
            //check_add_leaf_all_undefined();
            bool stop = false;
            check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk2(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_egbk2(file, show_log, &stop_loading);
        }



        /*
            @brief import Egaroucid-formatted book (old)

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_egbk(std::string file, bool show_log, bool *stop_loading){
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value;
            uint64_t p, o;
            uint8_t value_raw;
            // Book Information
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << n_boards << " boards to read" << std::endl;
            // for each board
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log){
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value_raw, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                value = -((int8_t)value_raw - HW2);
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                board.player = p;
                board.opponent = o;
                #if FORCE_BOOK_DEPTH
                    if (b.n_discs() <= 4 + 30){
                #endif
                        book_elem.value = value;
                        book_elem.leaf.value = SCORE_UNDEFINED;
                        book_elem.leaf.move = MOVE_UNDEFINED;
                        merge(board, book_elem);
                #if FORCE_BOOK_DEPTH
                    }
                #endif
            }
            //check_add_leaf_all_undefined();
            bool stop = false;
            check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_egbk(file, show_log, &stop_loading);
        }

        /*
            @brief import Edax-formatted book

            @param file                 book file (.dat file)
            @return book completely imported?
        */
        inline bool import_file_edax(std::string file, bool show_log, bool *stop) {
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Edax book " << file << std::endl;
                return false;
            }
            char elem_char;
            int elem_int;
            int16_t elem_short;
            // Header
            for (int i = 0; i < 38; ++i){
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            // Book Information
            if (fread(&elem_int, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            int n_boards = elem_int;
            std::cerr << n_boards << " boards found" << std::endl;
            uint64_t player, opponent;
            int16_t value;
            uint32_t n_lines;
            char link = 0, link_value, link_move, level, leaf_value, leaf_move;
            Board board;
            Flip flip;
            Book_elem book_elem;
            int percent = -1;
            for (int i = 0; i < n_boards; ++i){
                if (*stop)
                    return false;
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log){
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&player, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&opponent, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read additional data (w/d/l)
                for (int j = 0; j < 3; ++j) {
                    if (fread(&elem_int, 4, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                // read n_lines
                if (fread(&n_lines, 4, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 2, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (value < -HW2 || HW2 < value) {
                    //std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    //fclose(fp);
                    //return false;
                    //std::cerr << "[WARNING] value error found " << (int)value << " " << book.size() << " boards" << std::endl;
                    value = SCORE_UNDEFINED;
                }
                // read additional data
                for (int j = 0; j < 2; ++j) {
                    if (fread(&elem_short, 2, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                // read link data
                if (fread(&link, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read level (ignore)
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // for each link (ignore)
                for (int j = 0; j < (int)link; ++j) {
                    if (fread(&link_value, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    if (fread(&link_move, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                // read leaf value
                if (fread(&leaf_value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf move
                if (fread(&leaf_move, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (leaf_value < -HW2 || HW2 < leaf_value || leaf_move < 0 || HW2 <= leaf_move){
                    leaf_value = SCORE_UNDEFINED;
                    leaf_move = MOVE_UNDEFINED;
                }
                if (value != SCORE_UNDEFINED && (player & opponent) == 0ULL && calc_legal(player, opponent)){
                    board.player = player;
                    board.opponent = opponent;
                    book_elem.value = value;
                    book_elem.leaf.value = leaf_value;
                    book_elem.leaf.move = leaf_move;
                    book_elem.n_lines = n_lines;
                    merge(board, book_elem);
                }
            }
            if (show_log)
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            return true;
        }

        /*
            @brief save as Egaroucid-formatted book (.egbk3)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_egbk3(std::string file, std::string bak_file){
            if (remove(bak_file.c_str()) == -1)
                std::cerr << "cannot delete backup. you can ignore this error." << std::endl;
            rename(file.c_str(), bak_file.c_str());
            std::ofstream fout;
            fout.open(file.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
            if (!fout){
                std::cerr << "can't open " << file << std::endl;
                return;
            }
            char elem;
            std::cerr << "saving book..." << std::endl;
            char egaroucid_str[] = "DICUORAGE";
            fout.write((char*)&egaroucid_str, 9);
            char book_version = 3;
            fout.write((char*)&book_version, 1);
            int n_book = (int)book.size();
            fout.write((char*)&n_book, 4);
            int t = 0, percent = -1, n_boards = (int)book.size();
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                ++t;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent){
                    percent = n_percent;
                    std::cerr << "saving book " << percent << "%" << std::endl;
                }
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                elem = (char)itr->second.value;
                fout.write((char*)&elem, 1);
                fout.write((char*)&itr->second.n_lines, 4);
                elem = (char)itr->second.leaf.value;
                fout.write((char*)&elem, 1);
                elem = (char)itr->second.leaf.move;
                fout.write((char*)&elem, 1);
            }
            fout.close();
            int book_size = (int)book.size();
            std::cerr << "saved " << t << " boards , book_size " << book_size << std::endl;
        }

        void get_pass_boards(Board board, std::unordered_set<Board, Book_hash> &pass_boards){
            board = get_representative_board(board);
            if (contain_representative(board)){
                if (book[board].seen)
                    return;
                book[board].seen = true;
            }
            uint64_t legal = board.get_legal();
            if (legal == 0ULL){
                if (pass_boards.find(board) != pass_boards.end())
                    return;
                Board passed_board = board.copy();
                passed_board.pass();
                if (!contain_representative(board) && contain(passed_board)){
                    pass_boards.emplace(board);
                    get_pass_boards(passed_board, pass_boards);
                }
            } else{
                std::vector<Book_value> links = get_all_moves_with_value(&board);
                Flip flip;
                for (Book_value &link: links){
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        get_pass_boards(board, pass_boards);
                    board.undo_board(&flip);
                }
            }
        }

        /*
            @brief save as Edax-formatted book (.dat)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin_edax(std::string file, int level){
            bool stop = false;
            check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            std::unordered_set<Board, Book_hash> pass_boards;
            Board root_board;
            root_board.reset();
            std::cerr << "pass board calculating..." << std::endl;
            reset_seen();
            get_pass_boards(root_board, pass_boards);
            reset_seen();
            std::cerr << "pass board calculated " << pass_boards.size() << std::endl;
            std::ofstream fout;
            fout.open(file.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
            if (!fout){
                std::cerr << "can't open " << file << std::endl;
                return;
            }
            std::cerr << "saving book..." << std::endl;
            char header[] = "XADEKOOB";
            for (int i = 0; i < 8; ++i)
                fout.write((char*)&header[i], 1);
            char ver = 4;
            fout.write((char*)&ver, 1);
            char rel = 4;
            fout.write((char*)&rel, 1);
            int year, month, day, hour, minute, second;
            calc_date(&year, &month, &day, &hour, &minute, &second);
            fout.write((char*)&year, 2);
            fout.write((char*)&month, 1);
            fout.write((char*)&day, 1);
            fout.write((char*)&hour, 1);
            fout.write((char*)&minute, 1);
            fout.write((char*)&second, 1);
            char dummy = 0;
            fout.write((char*)&dummy, 1);
            fout.write((char*)&level, 4);
            int n_empties = HW2;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                n_empties = std::min(n_empties, HW2 + 1 - itr->first.n_discs());
            fout.write((char*)&n_empties, 4);
            int err_mid = 0;
            fout.write((char*)&err_mid, 4);
            int err_end = 0;
            fout.write((char*)&err_end, 4);
            int verb = 0;
            fout.write((char*)&verb, 4);
            int n_position = book.size() + pass_boards.size();
            fout.write((char*)&n_position, 4);
            int n_win = 0, n_draw = 0, n_lose = 0;
            uint32_t n_lines;
            short short_val, short_val_min = -HW2, short_val_max = HW2;
            char char_level = (char)level;
            Book_elem book_elem;
            char link_value, link_move;
            int max_link_value, min_link_value;
            char leaf_val, leaf_move;
            char n_link;
            Flip flip;
            Board b;
            bool searching = true;
            int percent = -1;
            int n_boards = (int)book.size();
            int t = 0;
            for (Board pass_board: pass_boards){
                Board passed_board = pass_board.copy();
                passed_board.pass();
                Book_elem passed_elem = get(passed_board);
                n_lines = passed_elem.n_lines;
                short_val = (short)passed_elem.value;
                n_link = 1;
                link_value = (char)passed_elem.value;
                link_move = MOVE_PASS;
                leaf_val = SCORE_UNDEFINED;
                leaf_move = MOVE_NOMOVE;
                fout.write((char*)&pass_board.player, 8);
                fout.write((char*)&pass_board.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_lines, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val_min, 2);
                fout.write((char*)&short_val_max, 2);
                fout.write((char*)&n_link, 1);
                fout.write((char*)&char_level, 1);
                fout.write((char*)&link_value, 1);
                fout.write((char*)&link_move, 1);
                fout.write((char*)&leaf_val, 1);
                fout.write((char*)&leaf_move, 1);
            }
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                book_elem = itr->second;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent){
                    percent = n_percent;
                    std::cerr << "converting book " << percent << "%" << std::endl;
                }
                ++t;
                short_val = book_elem.value;
                //short_val_min = book_elem.value;
                //short_val_max = book_elem.value;
                b = itr->first;
                std::vector<Book_value> links = get_all_moves_with_value(&b);
                n_link = (char)links.size();
                leaf_val = itr->second.leaf.value;
                leaf_move = itr->second.leaf.move;
                if (leaf_val < -HW2 || HW2 < leaf_val || leaf_move < 0 || HW2 <= leaf_move){
                    leaf_val = SCORE_UNDEFINED;
                    leaf_move = MOVE_NOMOVE;
                }
                n_lines = itr->second.n_lines;
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_lines, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val_min, 2);
                fout.write((char*)&short_val_max, 2);
                fout.write((char*)&n_link, 1);
                fout.write((char*)&char_level, 1);
                for (Book_value &book_value: links){
                    link_value = (char)book_value.value;
                    link_move = (char)book_value.policy;
                    fout.write((char*)&link_value, 1);
                    fout.write((char*)&link_move, 1);
                }
                fout.write((char*)&leaf_val, 1);
                fout.write((char*)&leaf_move, 1);
            }
            fout.close();
            std::cerr << "saved " << t << " boards as a edax-formatted book " << n_position << " " << book.size() << std::endl;
        }

        /*
            @brief register a board to book

            @param b                    a board to register
            @param elem                 book element
        */
        inline void reg(Board b, Book_elem elem){
            register_symmetric_book(b, elem);
        }

        /*
            @brief register a board to book

            @param b                    a board pointer to register
            @param elem                 book element
        */
        inline void reg(Board *b, Book_elem elem){
            register_symmetric_book(b->copy(), elem);
        }

        /*
            @brief check if book has a board

            @param b                    a board to find
            @return if contains, true, else false
        */
        inline bool contain_representative(Board b){
            return book.find(b) != book.end();
        }

        /*
            @brief check if book has a board

            @param b                    a board pointer to find
            @return if contains, true, else false
        */
        inline bool contain_representative(Board *b){
            return contain_representative(b->copy());
        }

        inline bool contain(Board b){
            return contain_representative(get_representative_board(b));
        }

        inline bool contain(Board *b){
            return contain_representative(get_representative_board(b));
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get_representative(Board b, int idx){
            Book_elem res;
            if (!contain_representative(b))
                return res;
            res = book[b];
            if (is_valid_policy(res.leaf.move))
                res.leaf.move = convert_coord_from_representative_board(res.leaf.move, idx);
            return res;
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board pointer to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board *b){
            int rotate_idx;
            Board representive_board = get_representative_board(b, &rotate_idx);
            return get_representative(representive_board, rotate_idx);
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board b){
            int rotate_idx;
            Board representive_board = get_representative_board(b, &rotate_idx);
            return get_representative(representive_board, rotate_idx);
        }

        /*
            @brief get all best moves

            @param b                    a board pointer to find
            @return vector of best moves
        */
        inline std::vector<int> get_all_best_moves(Board *b){
            std::vector<int> policies;
            uint64_t legal = b->get_legal();
            int max_value = -INF;
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    int sgn = -1;
                    if (b->get_legal() == 0ULL){
                        sgn = 1;
                        b->pass();
                    }
                    if (contain(b)){
                        Book_elem elem = get(b);
                        if (sgn * elem.value > max_value){
                            max_value = sgn * elem.value;
                            policies.clear();
                        }
                        if (sgn * elem.value == max_value)
                            policies.emplace_back(cell);
                    }
                    if (sgn == 1)
                        b->pass();
                b->undo_board(&flip);
            }
            return policies;
        }

        /*
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Book_value> get_all_moves_with_value(Board *b){
            std::vector<Book_value> policies;
            uint64_t legal = b->get_legal();
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    int sgn = -1;
                    if (b->get_legal() == 0ULL){
                        sgn = 1;
                        b->pass();
                    }
                    if (contain(b)){
                        Book_value book_value;
                        book_value.policy = cell;
                        book_value.value = sgn * get(b).value;
                        policies.emplace_back(book_value);
                    }
                    if (sgn == 1)
                        b->pass();
                b->undo_board(&flip);
            }
            return policies;
        }

        inline Book_value get_random_specified_moves(Board *b, int acc_level, uint64_t use_legal){
            std::vector<std::pair<double, int>> value_policies;
            std::vector<std::pair<int, int>> value_policies_memo;
            uint64_t legal = b->get_legal();
            double best_score = -INF;
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                if ((1ULL << cell) & use_legal){
                    calc_flip(&flip, b, cell);
                    b->move_board(&flip);
                        int sgn = -1;
                        if (b->get_legal() == 0ULL){
                            sgn = 1;
                            b->pass();
                        }
                        if (contain(b)){
                            Book_value book_value;
                            book_value.policy = cell;
                            book_value.value = sgn * get(b).value;
                            if (book_value.value > best_score)
                                best_score = (double)book_value.value;
                            value_policies.emplace_back(std::make_pair((double)book_value.value, cell));
                            value_policies_memo.emplace_back(std::make_pair(book_value.value, cell));
                        }
                        if (sgn == 1)
                            b->pass();
                    b->undo_board(&flip);
                }
            }
            Book_elem board_elem = get(b);
            Book_value res;
            if (value_policies.size() == 0 || best_score < board_elem.value - BOOK_LOSS_IGNORE_THRESHOLD){
                res.policy = -1;
                res.value = -INF;
                return res;
            }
            double acceptable_min_value = best_score - 2.0 * acc_level - 0.5;
            double sum_exp_values = 0.0;
            for (std::pair<double, int> &elem: value_policies){
                if (elem.first < acceptable_min_value)
                    elem.first = 0.0;
                else{
                    double exp_val = (exp(elem.first - best_score) + 1.5) / 3.0;
                    elem.first = pow(exp_val, BOOK_ACCURACY_LEVEL_INF - acc_level);
                }
                sum_exp_values += elem.first;
            }
            for (std::pair<double, int> &elem: value_policies)
                elem.first /= sum_exp_values;
            double rnd = myrandom();
            double s = 0.0;
            bool res_got = false;
            for (std::pair<double, int> &elem: value_policies){
                s += elem.first;
                if (s >= rnd){
                    res.policy = elem.second;
                    for (std::pair<int, int> elem: value_policies_memo){
                        if (elem.second == res.policy)
                            res.value = elem.first;
                    }
                    res_got = true;
                    break;
                }
            }
            if (!res_got){
                res.policy = value_policies.back().second;
                for (std::pair<int, int> elem: value_policies_memo){
                    if (elem.second == res.policy)
                        res.value = elem.first;
                }
            }
            return res;
        }

        /*
            @brief get a best move

            @param b                    a board pointer to find
            @param acc_level            accuracy level, 0 is very good, 10 is very bad
            @return best move and value as Book_value structure
        */
        inline Book_value get_random(Board *b, int acc_level){
            return get_random_specified_moves(b, acc_level, b->get_legal());
        }

        /*
            @brief get how many boards registered in this book

            @return number of registered boards
        */
        inline int get_n_book(){
            return (int)book.size();
        }

        /*
            @brief change or register a board

            @param b                    a board to change or register
            @param value                a value to change or register
        */
        inline void change(Board b, int value){
            if (-HW2 <= value && value <= HW2){
                if (contain(b)){
                    Board bb = get_representative_board(b);
                    book[bb].value = value;
                } else{
                    Book_elem elem;
                    elem.value = value;
                    elem.leaf.move = MOVE_UNDEFINED;
                    elem.leaf.value = SCORE_UNDEFINED;
                    register_symmetric_book(b, elem);
                }
            }
        }

        /*
            @brief change or register a board

            @param b                    a board pointer to change or register
            @param value                a value to change or register
        */
        inline void change(Board *b, int value){
            Board nb = b->copy();
            change(nb, value);
        }

        /*
            @brief delete a board

            @param b                    a board to delete
        */
        inline void delete_elem(Board b){
            delete_symmetric_book(b);
            //if (delete_symmetric_book(b)){
            //    std::cerr << "deleted book elem " << book.size() << std::endl;
            //} else
            //    std::cerr << "book elem NOT deleted " << book.size() << std::endl;
        }

        /*
            @brief delete all board in this book
        */
        inline void delete_all(){
            book.clear();
            reg_first_board();
        }

        /*
            @brief fix book
        */
        inline void fix(bool *stop){
            negamax_book(stop);
            check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, stop);
        }

        /*
            @brief fix book
        */
        inline void fix(){
            bool stop = false;
            fix(&stop);
        }

        Book_elem negamax_book_p(Board board, int64_t *n_seen, int64_t *n_fix, int *percent, bool *stop){
            if (*stop){
                Book_elem stop_res;
                stop_res.value = SCORE_UNDEFINED;
                stop_res.n_lines = 0;
                return stop_res;
            }
            board = get_representative_board(&board);
            Book_elem res = book[board];
            if (res.seen)
                return res;
            res.seen = true;
            book[board].seen = true;
            if (res.value < -HW2 || HW2 < res.value){
                //std::cerr << "value error found " << (int)res.value << std::endl;
                //board.print();
                Book_elem stop_res;
                stop_res.value = SCORE_UNDEFINED;
                stop_res.n_lines = 0;
                return stop_res;
            }
            ++(*n_seen);
            int n_percent = (double)(*n_seen) / book.size() * 100;
            if (n_percent > (*percent)){
                *percent = n_percent;
                std::cerr << "negamaxing book... " << (*percent) << "%" << " fixed " << (*n_fix) << std::endl;
            }
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            uint32_t n_lines = 1;
            Flip flip;
            int v = -INF;
            Book_elem child_res;
            for (Book_value &link: links){
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    child_res = negamax_book_p(board, n_seen, n_fix, percent, stop);
                    if (child_res.value != SCORE_UNDEFINED){
                        v = std::max(v, -child_res.value);
                        if (n_lines + child_res.n_lines < n_lines)
                            n_lines = MAX_N_LINES;
                        else
                            n_lines += child_res.n_lines;
                    }
                board.undo_board(&flip);
            }
            if (v != -INF && v != res.value){
                res.value = v;
                ++(*n_fix);
            }
            res.n_lines = n_lines;
            book[board] = res;
            return res;
        }

        void negamax_book(bool *stop){
            Board root_board;
            root_board.reset();
            int64_t n_seen = 0, n_fix = 0;
            int percent = -1;
            reset_seen();
            negamax_book_p(root_board, &n_seen, &n_fix, &percent, stop);
            reset_seen();
            std::cerr << "negamaxed book fixed " << n_fix << " boards seen " << n_seen << " boards size " << book.size() << std::endl;
        }

        void reduce_book_flag_moves(Board board, int max_depth, int max_error_per_move, int lower, int upper, uint64_t *n_flags, bool *doing){
            if (!*(doing))
                return;
            if (board.n_discs() > 4 + max_depth)
                return;
            if (board.get_legal() == 0){
                board.pass();
                int tmp = lower;
                lower = -upper;
                upper = -tmp;
                if (board.get_legal() == 0){
                    flag_book_elem(board);
                    ++(*n_flags);
                    return;
                }
            }
            Book_elem book_elem = get(board);
            if (book_elem.value < lower || upper < book_elem.value)
                return;
            if (book_elem.seen)
                return;
            flag_book_elem(board);
            ++(*n_flags);
            if ((*n_flags) % 100 == 0)
                std::cerr << "keep " << (*n_flags) << " boards" << std::endl;
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            for (Book_value &link: links){
                if (link.value < book_elem.value - max_error_per_move)
                    continue;
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    reduce_book_flag_moves(board, max_depth, max_error_per_move, -upper, -lower, n_flags, doing);
                board.undo_board(&flip);
            }
        }

        void delete_unflagged_moves(Board board, uint64_t *n_delete, bool *doing){
            if (!(*doing))
                return;
            if (!contain(board))
                return;
            if (board.get_legal() == 0){
                board.pass();
                if (board.get_legal() == 0)
                    return;
            }
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            if (!get(board).seen){
                ++(*n_delete);
                delete_elem(board);
                if ((*n_delete) % 100 == 0)
                    std::cerr << "deleting " << (*n_delete) << " boards" << std::endl;
            }
            Flip flip;
            for (Book_value &link: links){
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    delete_unflagged_moves(board, n_delete, doing);
                board.undo_board(&flip);
            }
        }

        void reduce_book(Board root_board, int max_depth, int max_error_per_move, int max_error_sum, bool *doing){
            reset_seen();
            Book_elem book_elem = get(root_board);
            if (book_elem.value == SCORE_UNDEFINED)
                return;
            int lower = book_elem.value - max_error_sum;
            int upper = book_elem.value + max_error_sum;
            if (lower < -SCORE_MAX)
                lower = -SCORE_MAX;
            if (upper > SCORE_MAX)
                upper = SCORE_MAX;
            uint64_t n_flags = 0, n_delete = 0;
            uint64_t book_size = book.size();
            reduce_book_flag_moves(root_board, max_depth, max_error_per_move, lower, upper, &n_flags, doing);
            delete_unflagged_moves(root_board, &n_delete, doing);
            reset_seen();
            std::cerr << "book reduced size before " << book_size << " after " << book.size();
            *doing = false;
        }

        uint64_t size(){
            return book.size();
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

        inline Board get_representative_board(Board *b, int *idx){
            return get_representative_board(b->copy(), idx);
        }

    private:
        void reg_first_board(){
            Board board;
            board.reset();
            Book_elem elem;
            elem.value = 0;
            elem.leaf.value = 0;
            elem.leaf.move = 19;
            book[board] = elem;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_representative(Board b, Book_elem elem){
            /*
            if (elem.value < -HW2 && HW2 < elem.value)
                return false;
            std::vector<Book_value> moves;
            for (Book_value &move: elem.moves){
                if (-HW2 <= move.value && move.value <= HW2)
                    moves.emplace_back(move);
            }
            elem.moves = moves;
            */
            int f_size = book.size();
            book[b] = elem;
            return book.size() - f_size > 0;
        }

        /*
            @brief delete a board

            @param b                    a board to delete
            @return board deleted?
        */
        inline bool delete_representative_board(Board b){
            if (book.find(b) != book.end()){
                book.erase(b);
                return true;
            }
            return false;
        }

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

        inline void first_update_representative_board(Board *res, Board *sym, int *idx, int *cnt){
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            ++(*cnt);
            if (res->player > vp || (res->player == vp && res->opponent > vo)){
                res->player = vp;
                res->opponent = vo;
                *idx = *cnt;
            }
        }

        inline void update_representative_board(Board *res, Board *sym, int *idx, int *cnt){
            ++(*cnt);
            if (res->player > sym->player || (res->player == sym->player && res->opponent > sym->opponent)){
                sym->copy(res);
                *idx = *cnt;
            }
            uint64_t vp = vertical_mirror(sym->player);
            uint64_t vo = vertical_mirror(sym->opponent);
            ++(*cnt);
            if (res->player > vp || (res->player == vp && res->opponent > vo)){
                res->player = vp;
                res->opponent = vo;
                *idx = *cnt;
            }
        }

        inline Board get_representative_board(Board b, int *idx){
            Board res = b;
            *idx = 0;
            int cnt = 0;
            first_update_representative_board(&res, &b, idx, &cnt);
            b.board_black_line_mirror();
            update_representative_board(&res, &b, idx, &cnt);
            b.board_horizontal_mirror();
            update_representative_board(&res, &b, idx, &cnt);
            b.board_white_line_mirror();
            update_representative_board(&res, &b, idx, &cnt);
            return res;
        }

        inline int convert_coord_from_representative_board(int cell, int idx){
            int res;
            int y = cell / HW;
            int x = cell % HW;
            switch (idx){
                case 0:
                    res = cell;
                    break;
                case 1:
                    res = (HW_M1 - y) * HW + x; // vertical
                    break;
                case 2:
                    res = (HW_M1 - x) * HW + (HW_M1 - y); // black line
                    break;
                case 3:
                    res = (HW_M1 - x) * HW + y; // black line + vertical ( = rotate 90 clockwise)
                    break;
                case 4:
                    res = x * HW + (HW_M1 - y); // black line + horizontal ( = rotate 90 counterclockwise)
                    break;
                case 5:
                    res = x * HW + y; // black line + horizontal + vertical ( = white line)
                    break;
                case 6:
                    res = y * HW + (HW_M1 - x); // horizontal
                    break;
                case 7:
                    res = (HW_M1 - y) * HW + (HW_M1 - x); // horizontal + vertical ( = rotate180)
                    break;
                default:
                    std::cerr << "converting coord error in book" << std::endl;
                    break;
            }
            return res;
        }

        inline int convert_coord_to_representative_board(int cell, int idx){
            int res;
            int y = cell / HW;
            int x = cell % HW;
            switch (idx){
                case 0:
                    res = cell;
                    break;
                case 1:
                    res = (HW_M1 - y) * HW + x; // vertical
                    break;
                case 2:
                    res = (HW_M1 - x) * HW + (HW_M1 - y); // black line
                    break;
                case 3:
                    res = x * HW + (HW_M1 - y); // black line + vertical ( = rotate 90 clockwise)
                    break;
                case 4:
                    res = (HW_M1 - x) * HW + y; // black line + horizontal ( = rotate 90 counterclockwise)
                    break;
                case 5:
                    res = x * HW + y; // black line + horizontal + vertical ( = white line)
                    break;
                case 6:
                    res = y * HW + (HW_M1 - x); // horizontal
                    break;
                case 7:
                    res = (HW_M1 - y) * HW + (HW_M1 - x); // horizontal + vertical ( = rotate180)
                    break;
                default:
                    std::cerr << "converting coord error in book" << std::endl;
                    break;
            }
            return res;
        }

        inline Board get_representative_board(Board *b){
            return get_representative_board(b->copy());
        }

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book(Board b, Book_elem elem){
            int idx;
            Board representive_board = get_representative_board(b, &idx);
            if (elem.leaf.move != MOVE_UNDEFINED)
                elem.leaf.move = convert_coord_to_representative_board(elem.leaf.move, idx);
            return register_representative(representive_board, elem);
        }

        /*
            @brief delete a board with checking all symmetry boards

            @param b                    a board to delete
            @return 1 if board is deleted (board found) else 0
        */
        inline int delete_symmetric_book(Board b){
            Board representive_board = get_representative_board(b);
            return delete_representative_board(representive_board);
        }

        inline int merge(Board b, Book_elem elem){
            if (!contain(b))
                return register_symmetric_book(b, elem);
            Book_elem book_elem = get(b);
            if (elem.value != SCORE_UNDEFINED)
                book_elem.value = elem.value;
            if (elem.leaf.value != SCORE_UNDEFINED){
                book_elem.leaf.value = elem.leaf.value;
                book_elem.leaf.move = elem.leaf.move;
            }
            return register_symmetric_book(b, book_elem);
        }

        void reset_seen(){
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                boards.emplace_back(itr->first);
            Flip flip;
            for (Board &board: boards)
                book[board].seen = false;
        }

        void flag_book_elem(Board board){
            book[get_representative_board(board)].seen = true;
        }
};








Book book;

bool book_init(std::string file, bool show_log){
    //book_hash_init(show_log);
    book_hash_init_rand();
    bool stop_loading = false;
    return book.init(file, show_log, &stop_loading);
}

bool book_init(std::string file, bool show_log, bool *stop_loading){
    //book_hash_init(show_log);
    book_hash_init_rand();
    return book.init(file, show_log, stop_loading);
}

void book_save_as_edax(std::string file, int level){
    book.save_bin_edax(file, level);
}

void book_fix(bool *stop){
    book.fix(stop);
}

void book_reduce(Board board, int depth, int max_error_per_move, int max_error_sum, bool *doing){
    book.reduce_book(board, depth, max_error_per_move, max_error_sum, doing);
}

void book_recalculate_leaf_all(int level, bool *stop){
    book.recalculate_leaf_all(level, stop);
}
