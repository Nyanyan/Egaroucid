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

#define BOOK_N_ACCEPT_LEVEL 11
#define BOOK_ACCURACY_LEVEL_INF 10

#define LEVEL_UNDEFINED -1

/*
    @brief book result structure

    @param policy               selected best move
    @param value                registered score
*/
struct Book_value{
    int policy;
    int value;
};

/*
    @brief book element structure

    @param value                registered score
    @param level                AI level
    @param moves                each moves and values
*/
struct Book_elem{
    int value;
    int level;
    std::vector<Book_value> moves;

    Book_elem(){
        value = SCORE_UNDEFINED;
        level = LEVEL_UNDEFINED;
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

class Book_old{
    private:
        std::unordered_map<Board, int, Book_hash> book;
        int n_book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading){
            delete_all();
            return import_file_bin(file, show_log, stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_bin_egbk(std::string file, bool show_log, bool *stop_loading){
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Egaroucid book (old version .egbk) " << file << std::endl;
                return false;
            }
            Board b;
            int n_boards, i, value;
            uint64_t p, o;
            uint8_t elem;
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            for (i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                if (i % 32768 == 0 && show_log)
                    std::cerr << "loading book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                value = elem - HW2;
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << value << " " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                b.player = p;
                b.opponent = o;
                n_book += register_symmetric_book(b, value);
            }
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "book imported " << n_book << " boards" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_bin(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_bin_egbk(file, show_log, &stop_loading);
        }

        /*
            @brief register a board to book

            @param b                    a board to register
            @param value                score of the board
        */
        inline void reg(Board b, int value){
            n_book += register_symmetric_book(b, value);
        }

        /*
            @brief register a board to book

            @param b                    a board pointer to register
            @param value                score of the board
        */
        inline void reg(Board *b, int value){
            Board b1 = b->copy();
            n_book += register_symmetric_book(b1, value);
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline int get_onebook(Board b){
            if (book.find(b) == book.end())
                return -INF;
            return book[b];
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board pointer to find
            @return registered value (if not registered, returns -INF)
        */
        inline int get(Board *b){
            Board representive_board = get_representative_board(b);
            return get_onebook(representive_board);
        }

        /*
            @brief get all best moves

            @param b                    a board pointer to find
            @return vector of best moves
        */
        inline std::vector<int> get_all_best_moves(Board *b){
            std::vector<int> policies;
            if (get(b) == -INF)
                return policies;
            Board nb;
            int max_value = -INF;
            uint64_t legal = b->get_legal();
            Flip flip;
            int value;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, b, cell);
                nb = b->move_copy(&flip);
                value = get(&nb);
                if (value != -INF){
                    if (max_value < value){
                        max_value = value;
                        policies.clear();
                        policies.push_back(cell);
                    } else if (value == max_value)
                        policies.push_back(cell);
                }
            }
            return policies;
        }

        /*
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Search_result> get_all_moves_with_value(Board *b){
            std::vector<Search_result> policies;
            if (get(b) == -INF)
                return policies;
            Board nb;
            uint64_t legal = b->get_legal();
            Flip flip;
            int value;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, b, cell);
                nb = b->move_copy(&flip);
                value = get(&nb);
                if (value != -INF){
                    Search_result elem;
                    elem.policy = cell;
                    elem.value = value;
                    elem.depth = SEARCH_BOOK;
                    elem.time = 0;
                    elem.nodes = 0;
                    elem.nps = 0;
                    policies.emplace_back(elem);
                }
            }
            return policies;
        }

        /*
            @brief get a best move

            @param b                    a board pointer to find
            @param accept_value         an error to allow
            @return best move and value as Book_value structure
        */
        inline Book_value get_random(Board *b, int acc_level){
            std::vector<std::pair<double, int>> value_policies;
            int best_score = -INF;
            uint64_t legal = b->get_legal();
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, b, cell);
                Board nb = b->move_copy(&flip);
                int value = get(&nb);
                if (value != -INF){
                    if (value > best_score){
                        best_score = value;
                    }
                    value_policies.emplace_back(std::make_pair((double)value, (int)cell));
                }
            }
            Book_value res;
            if (value_policies.size() == 0){
                res.policy = -1;
                res.value = -INF;
                return res;
            }
            double sum_exp_values = 0.0;
            for (std::pair<double, int> &elem: value_policies){
                if (acc_level == BOOK_ACCURACY_LEVEL_INF && elem.first < (double)best_score - 0.5)
                    elem.first = 0.0;
                else{
                    double exp_val = (exp(elem.first - (double)best_score) + 2.0) / 3.0;
                    elem.first = pow(exp_val, acc_level);
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
                    calc_flip(&flip, b, res.policy);
                    Board nb = b->move_copy(&flip);
                    res.value = get(&nb);
                    res_got = true;
                    break;
                }
            }
            if (!res_got){
                res.policy = value_policies.back().second;
                calc_flip(&flip, b, res.policy);
                Board nb = b->move_copy(&flip);
                res.value = get(&nb);
            }
            return res;
        }

        /*
            @brief get how many boards registered in this book

            @return number of registered boards
        */
        inline int get_n_book(){
            return n_book;
        }

        /*
            @brief change or register a board

            @param b                    a board to change or register
            @param value                a value to change or register
        */
        inline void change(Board b, int value){
            if (register_symmetric_book(b, value)){
                n_book++;
                std::cerr << "book registered " << n_book << std::endl;
            } else
                std::cerr << "book changed " << n_book << std::endl;
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
            @brief delete all board in this book
        */
        inline void delete_all(){
            book.clear();
            reg_first_board();
            n_book = 1;
        }

    private:
        void reg_first_board(){
            Board board;
            board.reset();
            book[board] = 0;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_book(Board b, int value){
            int f_size = book.size();
            book[b] = value;
            return book.size() - f_size > 0;
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

        inline Board get_representative_board(Board *b){
            return get_representative_board(b->copy());
        }

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book(Board b, int value){
            Board representive_board = get_representative_board(b);
            return register_book(representive_board, value);
        }
};

/*
    @brief book data

    @param book                 book data
    @param n_book               number of boards registered
*/
class Book{
    private:
        std::unordered_map<Board, Book_elem, Book_hash> book;
        int n_book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading){
            delete_all();
            return import_file_bin(file, show_log, stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk2 file)
            @return book completely imported?
        */
        inline bool import_file_bin(std::string file, bool show_log, bool *stop_loading){
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")){
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board b;
            int n_boards, i, value;
            uint64_t p, o;
            uint8_t elem;
            uint8_t level, n_moves, val, mov;
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            for (i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                if (i % 32768 == 0 && show_log)
                    std::cerr << "loading book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                value = elem - HW2;
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << value << " " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                std::vector<Book_value> moves;
                if (fread(&n_moves, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (uint8_t i = 0; i < n_moves; ++i){
                    if (fread(&val, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    if (fread(&mov, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << n_book << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    moves.emplace_back(Book_value{(int)mov, (int)val});
                }
                b.player = p;
                b.opponent = o;
                n_book += register_symmetric_book(b, Book_elem{value, level, moves});
            }
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "book imported " << n_book << " boards" << std::endl;
            fclose(fp);
            return true;
        }

        /*
            @brief import Egaroucid-formatted book (old version .egbk file)

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_bin_egbk(std::string file, bool show_log, bool *stop_loading){
            Book_old book_old;
            if (!book_old.init(file, show_log, *stop_loading))
                return false;
            
        }

        inline bool import_file_bin(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_bin(file, show_log, &stop_loading);
        }

        /*
            @brief import Edax-formatted book

            @param file                 book file (.dat file)
            @return book completely imported?
        */
        inline bool import_edax_book(std::string file, bool show_log) {
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
            int i, j;
            for (i = 0; i < 38; ++i){
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&elem_int, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            int n_boards = elem_int;
            uint64_t player, opponent;
            int16_t value;
            char link = 0, link_value, link_move, level;
            Board b;
            Flip flip;
            for (i = 0; i < n_boards; ++i){
                if (i % 32768 == 0 && show_log)
                    std::cerr << "loading edax book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&player, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&opponent, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 4; ++j) {
                    if (fread(&elem_int, 4, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&value, 2, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 2; ++j) {
                    if (fread(&elem_short, 2, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&link, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                b.player = player;
                b.opponent = opponent;
                value *=- -1;
                std::vector<Book_value> moves;
                for (j = 0; j < (int)link + 1; ++j) {
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
                    if (link_move < HW2) {
                        link_value *=- -1;
                        moves.emplace_back(Book_value{(int)link_move, (int)link_value});
                    }
                }
                n_book += register_symmetric_book(b, Book_elem{(int)value, (int)level, moves});
            }
            if (show_log)
                std::cerr << "book imported " << n_book << " boards" << std::endl;
            return true;
        }

        /*
            @brief register a board to book

            @param b                    a board to register
            @param elem                 book element
        */
        inline void reg(Board b, Book_elem elem){
            n_book += register_symmetric_book(b, elem);
        }

        /*
            @brief register a board to book

            @param b                    a board pointer to register
            @param elem                 book element
        */
        inline void reg(Board *b, Book_elem elem){
            n_book += register_symmetric_book(b->copy(), elem);
        }

        /*
            @brief check if book has a board

            @param b                    a board to find
            @return if contains, true, else false
        */
        inline bool contain(Board b){
            return book.find(b) != book.end();
        }

        /*
            @brief check if book has a board

            @param b                    a board pointer to find
            @return if contains, true, else false
        */
        inline bool contain(Board *b){
            return book.find(b->copy()) != book.end();
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get_onebook(Board b){
            Book_elem res;
            if (book.find(b) == book.end())
                return res;
            res = book[b];
            return res;
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board pointer to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board *b){
            Board representive_board = get_representative_board(b);
            return get_onebook(representive_board);
        }

        /*
            @brief get all best moves

            @param b                    a board pointer to find
            @return vector of best moves
        */
        inline std::vector<int> get_all_best_moves(Board *b){
            std::vector<int> policies;
            Book_elem board_elem = get(b);
            int max_value = -INF;
            for (Book_value elem: board_elem.moves){
                if (elem.value > max_value){
                    max_value = elem.value;
                    policies.clear();
                }
                if (elem.value == max_value)
                    policies.emplace_back(elem.policy);
            }
            return policies;
        }

        /*
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Search_result> get_all_moves_with_value(Board *b){
            std::vector<Search_result> policies;
            Book_elem board_elem = get(b);
            for (Book_value elem: board_elem.moves){
                Search_result search_result;
                search_result.policy = elem.policy;
                search_result.value = elem.value;
                search_result.depth = SEARCH_BOOK;
                search_result.time = 0;
                search_result.nodes = 0;
                search_result.nps = 0;
                policies.emplace_back(elem);
            }
            return policies;
        }

        /*
            @brief get a best move

            @param b                    a board pointer to find
            @param acc_level            accuracy level, 0 is very bad, 10 is very good
            @return best move and value as Book_value structure
        */
        inline Book_value get_random(Board *b, int acc_level){
            std::vector<std::pair<double, int>> value_policies;
            Book_elem board_elem = get(b);
            double best_score = -INF;
            for (Book_value elem: board_elem.moves){
                if (elem.value > best_score)
                    best_score = (double)elem.value;
                if (elem.value == best_score)
                    value_policies.emplace_back(std::make_pair((double)elem.value, elem.policy));
            }
            Book_value res;
            if (value_policies.size() == 0){
                res.policy = -1;
                res.value = -INF;
                return res;
            }
            double acceptable_min_value = best_score - 2.0 * acc_level;
            double sum_exp_values = 0.0;
            for (std::pair<double, int> &elem: value_policies){
                if (acc_level == BOOK_ACCURACY_LEVEL_INF && elem.first < (double)best_score - 0.5)
                    elem.first = 0.0;
                else if (elem.first)
                else{
                    double exp_val = (exp(elem.first - best_score) + 2.0) / 3.0;
                    elem.first = pow(exp_val, acc_level);
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
                    calc_flip(&flip, b, res.policy);
                    Board nb = b->move_copy(&flip);
                    res.value = get(&nb);
                    res_got = true;
                    break;
                }
            }
            if (!res_got){
                res.policy = value_policies.back().second;
                calc_flip(&flip, b, res.policy);
                Board nb = b->move_copy(&flip);
                res.value = get(&nb);
            }
            return res;
        }

        /*
            @brief get how many boards registered in this book

            @return number of registered boards
        */
        inline int get_n_book(){
            return n_book;
        }

        /*
            @brief change or register a board

            @param b                    a board to change or register
            @param value                a value to change or register
        */
        inline void change(Board b, int value){
            if (register_symmetric_book(b, value)){
                n_book++;
                std::cerr << "book registered " << n_book << std::endl;
            } else
                std::cerr << "book changed " << n_book << std::endl;
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
            if (delete_symmetric_book(b)){
                n_book--;
                std::cerr << "deleted book elem " << n_book << std::endl;
            } else
                std::cerr << "book elem NOT deleted " << n_book << std::endl;
        }

        /*
            @brief delete all board in this book
        */
        inline void delete_all(){
            book.clear();
            reg_first_board();
            n_book = 1;
        }

        /*
            @brief save as Egaroucid-formatted book (.egbk)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin(std::string file, std::string bak_file){
            if (remove(bak_file.c_str()) == -1)
                std::cerr << "cannot delete backup. you can ignore this." << std::endl;
            rename(file.c_str(), bak_file.c_str());
            std::ofstream fout;
            fout.open(file.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
            if (!fout){
                std::cerr << "can't open " << file << std::endl;
                return;
            }
            uint8_t elem;
            std::cerr << "saving book..." << std::endl;
            fout.write((char*)&n_book, 4);
            int t = 0;
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                ++t;
                if (t % 65536 == 0)
                    std::cerr << "saving book " << (t * 100 / (int)book.size()) << "%" << std::endl;
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                elem = std::max(0, std::min(HW2 * 2, itr->second + HW2));
                fout.write((char*)&elem, 1);
            }
            fout.close();
            std::cerr << "saved " << t << " boards" << std::endl;
        }

        /*
            @brief save as Edax-formatted book (.dat)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin_edax(std::string file){
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
            int level = 21; // fixed
            fout.write((char*)&level, 4);
            int n_empties = HW2;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                n_empties = std::min(n_empties, HW2 - itr->first.n_discs());
            fout.write((char*)&n_empties, 4);
            int err_mid = 0;
            fout.write((char*)&err_mid, 4);
            int err_end = 0;
            fout.write((char*)&err_end, 4);
            int verb = 0;
            fout.write((char*)&verb, 4);
            int n_positions = 0;
            std::unordered_set<Board, Book_hash> positions;
            int t = 0;
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                ++t;
                if (t % 16384 == 0)
                    std::cerr << "converting book " << (t * 100 / (int)book.size()) << "%" << std::endl;
                Board board = itr->first;
                uint64_t legal = board.get_legal();
                Flip flip;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                        if (get(&board) != -INF){
                            ++n_positions;
                            board.undo_board(&flip);
                            positions.emplace(get_representative_board(board));
                            break;
                        }
                    board.undo_board(&flip);
                }
            }
            std::cerr << "Edax formatted positions " << n_positions << std::endl;
            fout.write((char*)&n_positions, 4);
            t = 0;
            int n_win = 0;
            int n_draw = 0;
            int n_lose = 0;
            int n_line = 0;
            int16_t short_val;
            char char_level = 21; // fixed
            for (Board board: positions){
                ++t;
                if (t % 8192 == 0)
                    std::cerr << "saving book " << (t * 100 / (int)positions.size()) << "%" << std::endl;
                char n_link = 0;
                uint64_t legal = board.get_legal();
                Flip flip;
                std::vector<std::pair<char, char>> links;
                char leaf_val = 65;
                char leaf_move = 65;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &board, cell);
                    board.move_board(&flip);
                        int nval = get(&board);
                        if (nval != -INF){
                            if (positions.find(get_representative_board(board)) != positions.end()){ // is link
                                ++n_link;
                                links.emplace_back(std::make_pair((char)nval, (char)cell));
                            } else{ // is leaf
                                if (nval < leaf_val){
                                    leaf_val = nval;
                                    leaf_move = cell;
                                }
                            }
                        }
                    board.undo_board(&flip);
                }
                if (n_link){
                    fout.write((char*)&board.player, 8);
                    fout.write((char*)&board.opponent, 8);
                    fout.write((char*)&n_win, 4);
                    fout.write((char*)&n_draw, 4);
                    fout.write((char*)&n_lose, 4);
                    fout.write((char*)&n_line, 4);
                    short_val = -get(&board);
                    fout.write((char*)&short_val, 2);
                    fout.write((char*)&short_val, 2);
                    fout.write((char*)&short_val, 2);
                    fout.write((char*)&n_link, 1);
                    fout.write((char*)&char_level, 1);
                    for (std::pair<char, char> &link: links){
                        fout.write((char*)&link.first, 1);
                        fout.write((char*)&link.second, 1);
                    }
                    fout.write((char*)&leaf_val, 1);
                    fout.write((char*)&leaf_move, 1);
                }
            }
            fout.close();
            std::cerr << "saved " << t << " boards as a edax-formatted book" << std::endl;
        }


    private:
        void reg_first_board(){
            Board board;
            board.reset();
            book_old[board] = 0;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_book_old(Board b, int value){
            int f_size = book_old.size();
            book_old[b] = value;
            return book_old.size() - f_size > 0;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_book(Board b, Book_elem elem){
            int f_size = book.size();
            book[b] = elem;
            return book.size() - f_size > 0;
        }

        /*
            @brief delete a board

            @param b                    a board to delete
            @return board deleted?
        */
        inline bool delete_book(Board b){
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

        inline Board get_representative_board(Board *b){
            return get_representative_board(b->copy());
        }

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book_old(Board b, int value){
            Board representive_board = get_representative_board(b);
            return register_book_old(representive_board, value);
        }

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book(Board b, Book_elem elem){
            Board representive_board = get_representative_board(b);
            return register_book(representive_board, elem);
        }

        /*
            @brief delete a board with checking all symmetry boards

            @param b                    a board to delete
            @return 1 if board is deleted (board found) else 0
        */
        inline int delete_symmetric_book(Board b){
            Board representive_board = get_representative_board(b);
            return delete_book(representive_board);
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

void book_save_as_edax(std::string file){
    book.save_bin_edax(file);
}