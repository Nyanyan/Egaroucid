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

#define BOOK_N_ACCEPT_LEVEL 11
#define BOOK_ACCURACY_LEVEL_INF 10

#define LEVEL_UNDEFINED -1
#define LEVEL_HUMAN 70
#define BOOK_LOSS_IGNORE_THRESHOLD 8

#define FORCE_BOOK_LEVEL false
#define FORCE_BOOK_DEPTH false

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

struct Book_negamax{
    int value;
    int level;
};

Book_negamax negamax_book_global(Board board, bool *stop);

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
    public:
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
            return import_file_bin_egbk(file, show_log, stop_loading);
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
            std::cerr << n_boards << " boards" << std::endl;
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
                value = (int)elem - HW2;
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

        inline bool import_file_bin_egbk(std::string file, bool show_log){
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
                return SCORE_UNDEFINED;
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
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Book_value> get_all_moves_with_value(Board b){
            std::vector<Book_value> policies;
            if (get(&b) == -INF)
                return policies;
            Board nb;
            uint64_t legal = b.get_legal();
            Flip flip;
            int value;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &b, cell);
                nb = b.move_copy(&flip);
                value = get(&nb);
                if (value != SCORE_UNDEFINED){
                    Book_value elem;
                    elem.policy = cell;
                    elem.value = value;
                    policies.emplace_back(elem);
                }
            }
            return policies;
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
*/
class Book{
    private:
        std::mutex mtx;
        std::unordered_map<Board, Book_elem, Book_hash> book;
        std::unordered_map<Board, int, Book_hash> n_lines;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading){
            delete_all();
            if (!import_file_bin(file, show_log, stop_loading))
                return import_file_bin_egbk(file, show_log, stop_loading);
            return true;
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
            Book_elem book_elem;
            int n_boards, i;
            char value;
            uint64_t p, o;
            char level, n_moves, val, mov;
            char egaroucid_str[10];
            char egaroucid_str_ans[] = "DICUORAGE";
            char elem_char;
            char book_version;
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
                std::cerr << "[ERROR] This is not Egarocuid book version 2, found version" << (int)book_version << std::endl;
                fclose(fp);
                return false;
            }
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << n_boards << " boards to read" << std::endl;
            for (i = 0; i < n_boards; ++i) {
                if (*stop_loading)
                    break;
                if (i % 32768 == 0 && show_log)
                    std::cerr << "loading book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << value << " " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                book_elem.moves.clear();
                if (fread(&n_moves, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (uint8_t i = 0; i < n_moves; ++i){
                    if (fread(&val, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    if (fread(&mov, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    book_elem.moves.emplace_back(Book_value{(int)mov, (int)val});
                }
                b.player = p;
                b.opponent = o;
                #if FORCE_BOOK_DEPTH
                    if (b.n_discs() <= 4 + 30){
                #endif
                        book_elem.value = (int)value;
                        #if FORCE_BOOK_LEVEL
                            book_elem.level = 21;
                        #else
                            book_elem.level = (int)level;
                        #endif
                        merge(b, book_elem);
                #if FORCE_BOOK_DEPTH
                    }
                #endif
            }
            if (*stop_loading){
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log)
                std::cerr << "book imported " << book.size() << " boards" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_bin(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_bin(file, show_log, &stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book (old version .egbk file)

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_bin_egbk(std::string file, bool show_log, bool *stop_loading){
            Book_old book_old;
            if (!book_old.init(file, show_log, stop_loading))
                return false;
            int t = 0;
            for (auto itr = book_old.book.begin(); itr != book_old.book.end(); ++itr){
                ++t;
                if (t % 16384 == 0)
                    std::cerr << "converting book " << (t * 100 / (int)book_old.book.size()) << "%" << std::endl;
                Book_elem book_elem;
                book_elem.value = -itr->second;
                book_elem.level = 21; // fixed
                book_elem.moves = book_old.get_all_moves_with_value(itr->first);
                merge(itr->first, book_elem);
            }
        }

        inline bool import_file_bin_egbk(std::string file, bool show_log){
            bool stop_loading = false;
            return import_file_bin_egbk(file, show_log, &stop_loading);
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
            Book_elem book_elem;
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
                book_elem.moves.clear();
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
                        Book_value book_value;
                        book_value.policy = link_move;
                        book_value.value = link_value;
                        book_elem.moves.emplace_back(book_value);
                    }
                }
                book_elem.value = value;
                book_elem.level = level;
                merge(b, book_elem);
            }
            if (show_log)
                std::cerr << "book imported " << book.size() << " boards" << std::endl;
            return true;
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
        inline bool contain(Board b){
            return book.find(b) != book.end();
        }

        /*
            @brief check if book has a board

            @param b                    a board pointer to find
            @return if contains, true, else false
        */
        inline bool contain(Board *b){
            return contain(b->copy());
        }

        inline bool contain_symmetry(Board b){
            return contain(get_representative_board(b));
        }

        inline bool contain_symmetry(Board *b){
            return contain(get_representative_board(b));
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get_onebook(Board b, int idx){
            Book_elem res;
            if (!contain(b))
                return res;
            res = book[b];
            /*
            uint64_t legal = b.get_legal();
            Flip flip;
            for (Book_value &elem: res.moves)
                legal ^= 1ULL << elem.policy;
            if (legal){
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &b, cell);
                    b.move_board(&flip);
                        if (b.get_legal()){
                            if (contain_symmetry(b)){
                                Book_value move;
                                move.policy = cell;
                                move.value = -get(b).value;
                                res.moves.emplace_back(move);
                            }
                        } else{
                            b.pass();
                                if (contain_symmetry(b)){
                                    Book_value move;
                                    move.policy = cell;
                                    move.value = get(b).value;
                                    res.moves.emplace_back(move);
                                }
                            b.pass();
                        }
                    b.undo_board(&flip);
                }
            }
            */
            for (Book_value &elem: res.moves)
                elem.policy = convert_coord_from_representative_board(elem.policy, idx);
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
            return get_onebook(representive_board, rotate_idx);
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board b){
            int rotate_idx;
            Board representive_board = get_representative_board(b, &rotate_idx);
            return get_onebook(representive_board, rotate_idx);
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
        inline std::vector<Book_value> get_all_moves_with_value(Board *b){
            std::vector<Book_value> policies;
            Book_elem board_elem = get(b);
            for (Book_value elem: board_elem.moves){
                Book_value book_value;
                book_value.policy = elem.policy;
                book_value.value = elem.value;
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
                value_policies.emplace_back(std::make_pair((double)elem.value, elem.policy));
            }
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
                    std::cerr << idx_to_coord(elem.second) << " " << elem.first << std::endl;
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
            Flip flip;
            for (std::pair<double, int> &elem: value_policies){
                s += elem.first;
                if (s >= rnd){
                    res.policy = elem.second;
                    calc_flip(&flip, b, res.policy);
                    Board nb = b->move_copy(&flip);
                    for (Book_value elem: board_elem.moves){
                        if (elem.policy == res.policy)
                            res.value = elem.value;
                    }
                    res_got = true;
                    break;
                }
            }
            if (!res_got){
                res.policy = value_policies.back().second;
                calc_flip(&flip, b, res.policy);
                Board nb = b->move_copy(&flip);
                res.value = get(&nb).value;
            }
            return res;
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
        inline void change(Board b, int value, int level){
            if (-HW2 <= value && value <= HW2){
                if (contain_symmetry(b)){
                    Board bb = get_representative_board(b);
                    book[bb].value = value;
                    book[bb].level = level;
                } else{
                    Book_elem elem;
                    elem.value = value;
                    elem.level = level;
                    register_symmetric_book(b, elem);
                }
            }
        }

        /*
            @brief change or register a board

            @param b                    a board pointer to change or register
            @param value                a value to change or register
        */
        inline void change(Board *b, int value, int level){
            Board nb = b->copy();
            change(nb, value, level);
        }

        /*
            @brief delete a board

            @param b                    a board to delete
        */
        inline void delete_elem(Board b){
            if (delete_symmetric_book(b)){
                std::cerr << "deleted book elem " << book.size() << std::endl;
            } else
                std::cerr << "book elem NOT deleted " << book.size() << std::endl;
        }

        /*
            @brief delete all board in this book
        */
        inline void delete_all(){
            book.clear();
            reg_first_board();
        }

        /*
            @brief save as Egaroucid-formatted book (.egbk)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin(std::string file, std::string bak_file){
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
            char book_version = 2;
            fout.write((char*)&book_version, 1);
            int n_book = (int)book.size();
            fout.write((char*)&n_book, 4);
            int t = 0;
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                ++t;
                if (t % 65536 == 0)
                    std::cerr << "saving book " << (t * 100 / (int)book.size()) << "%" << std::endl;
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                elem = (char)itr->second.value;
                fout.write((char*)&elem, 1);
                elem = (char)itr->second.level;
                fout.write((char*)&elem, 1);
                elem = (char)itr->second.moves.size();
                fout.write((char*)&elem, 1);
                for (Book_value &move: itr->second.moves){
                    elem = (char)move.value;
                    fout.write((char*)&elem, 1);
                    elem = (char)move.policy;
                    fout.write((char*)&elem, 1);
                }
            }
            fout.close();
            int book_size = (int)book.size();
            std::cerr << "saved " << t << " boards , book_size " << book_size << std::endl;
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
            int level = 60;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                level = std::min(level, itr->second.level);
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
            int n_position = 0;
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                if (itr->second.moves.size())
                    ++n_position;
            }
            fout.write((char*)&n_position, 4);
            int t = 0;
            int n_win = 0, n_draw = 0, n_lose = 0;
            int n_line;
            short short_val;
            char char_level;
            Book_elem book_elem;
            char link_value, link_move;
            int max_link_value;
            char leaf_val, leaf_move;
            char n_link;
            Flip flip;
            Board b;
            for (auto itr = book.begin(); itr != book.end(); ++itr){
                book_elem = itr->second;
                if (book_elem.moves.size() == 0)
                    continue;
                ++t;
                if (t % 8192 == 0)
                    std::cerr << "converting book " << (t * 100 / n_position) << "%" << std::endl;
                short_val = book_elem.value;
                char_level = book_elem.level;
                if (char_level > 60)
                    char_level = 60;
                std::vector<Book_value> links;
                leaf_val = -65;
                leaf_move = 65;
                max_link_value = -65;
                b = itr->first;
                for (Book_value &book_value: book_elem.moves){
                    calc_flip(&flip, &b, (uint_fast8_t)book_value.policy);
                    b.move_board(&flip);
                        if (get(b).moves.size()){
                            links.emplace_back(book_value);
                            if (book_value.value > max_link_value)
                                max_link_value = book_value.value;
                        }
                        else if (leaf_val < book_value.value){
                            leaf_val = book_value.value;
                            leaf_move = book_value.policy;
                        }
                    b.undo_board(&flip);
                }
                n_link = (char)links.size();
                if (leaf_move == 65){
                    uint64_t legal = b.get_legal();
                    for (Book_value &link: links)
                        legal ^= 1ULL << link.policy;
                    if (legal){
                        leaf_val = -65;
                        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                            calc_flip(&flip, &b, cell);
                            b.move_board(&flip);
                                int g = -mid_evaluate(&b);
                                if (g > max_link_value)
                                    g = max_link_value;
                            b.undo_board(&flip);
                            if (leaf_val < g){
                                leaf_val = g;
                                leaf_move = cell;
                            }
                        }
                    }
                }
                if (leaf_val == -65)
                    leaf_val = 0;
                n_line = 0; //count_n_line(itr->first);
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_line, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val, 2);
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
            n_lines.clear();
            std::cerr << "saved " << t << " boards as a edax-formatted book " << n_position << " " << book.size() << std::endl;
        }

        /*
            @brief fix book
        */
        inline void fix(bool *stop){
            link_book(stop);
            Board root_board;
            root_board.reset();
            std::cerr << "negamaxing book..." << std::endl;
            negamax_book(root_board, stop);
        }

        /*
            @brief fix book
        */
        inline void fix(){
            bool stop = false;
            link_book(&stop);
            Board root_board;
            root_board.reset();
            std::cerr << "negamaxing book..." << std::endl;
            negamax_book(root_board, &stop);
        }

        void link_book(bool *stop){
            std::cerr << "linking book..." << std::endl;
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr)
                boards.emplace_back(itr->first);
            Book_elem book_elem;
            Flip flip;
            int t = 0;
            Board nb;
            uint64_t legal;
            bool elem_changed;
            for (Board &b: boards){
                if (*stop)
                    break;
                ++t;
                if (t % 16384 == 0)
                    std::cerr << "linking book " << (t * 100 / (int)boards.size()) << "%" << std::endl;
                book_elem = book[b];
                legal = b.get_legal();
                for (Book_value &elem: book_elem.moves)
                    legal ^= 1ULL << elem.policy;
                if (legal){
                    elem_changed = false;
                    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        calc_flip(&flip, &b, cell);
                        b.move_board(&flip);
                            if (b.get_legal()){
                                if (contain_symmetry(b)){
                                    Book_value move;
                                    move.policy = cell;
                                    move.value = -get(b).value;
                                    book_elem.moves.emplace_back(move);
                                    elem_changed = true;
                                }
                            } else{
                                b.pass();
                                    if (contain_symmetry(b)){
                                        Book_value move;
                                        move.policy = cell;
                                        move.value = get(b).value;
                                        book_elem.moves.emplace_back(move);
                                        elem_changed = true;
                                    }
                                b.pass();
                            }
                        b.undo_board(&flip);
                    }
                    if (elem_changed)
                        book[b] = book_elem;
                }
            }
        }

        Book_negamax negamax_book(Board board, bool *stop){
            Book_negamax res;
            Book_elem book_elem = get(board);
            res.value = book_elem.value;
            res.level = book_elem.level;
            if (*stop)
                return res;
            if (book_elem.value == SCORE_UNDEFINED)
                return res;
            if (book_elem.moves.size() == 0)
                return res;
            Flip flip;
            int best_score = -SCORE_INF;
            int best_level = -1;
            Book_negamax child;
            int best_registered_score = -SCORE_INF;
            std::vector<std::pair<int, std::future<Book_negamax>>> parallel_tasks;
            bool pushed;
            int move_idx = 0;
            bool node_updated = false;
            for (Book_value &move: book_elem.moves){
                best_registered_score = std::max(best_registered_score, move.value);
                calc_flip(&flip, &board, move.policy);
                board.move_board(&flip);
                    pushed = false;
                    if (thread_pool.get_n_idle()){
                        parallel_tasks.emplace_back(std::make_pair(move_idx, thread_pool.push(&pushed, std::bind(&negamax_book_global, board, stop))));
                        if (!pushed)
                            parallel_tasks.pop_back();
                    }
                    if (!pushed)
                        child = negamax_book(board, stop);
                board.undo_board(&flip);
                if (!pushed){
                    if (-HW2 <= child.value && child.value <= HW2){
                        if (best_score < -child.value){
                            best_score = -child.value;
                            best_level = child.level;
                        }
                        move.value = -child.value;
                        node_updated = true;
                    }
                }
                ++move_idx;
            }
            for (std::pair<int, std::future<Book_negamax>> &task: parallel_tasks){
                child = task.second.get();
                if (-HW2 <= child.value && child.value <= HW2){
                    if (best_score < -child.value){
                        best_score = -child.value;
                        best_level = child.level;
                    }
                    book_elem.moves[task.first].value = -child.value;
                    node_updated = true;
                }
            }
            bool do_not_update_this_node = best_registered_score < book_elem.value - BOOK_LOSS_IGNORE_THRESHOLD;
            if (best_level >= book_elem.level && !do_not_update_this_node && -HW2 <= best_score && best_score <= HW2){
                res.value = best_score;
                res.level = best_level;
                /*
                for (Book_value &move: book_elem.moves)
                    std::cerr << idx_to_coord(move.policy) << " " << move.value << std::endl;
                board.print();
                std::cerr << best_registered_score << "  " << book_elem.value << " " << book_elem.level << "  " << best_score << " " << best_level << std::endl;
                char e;
                std::cin >> e;
                */
                book_elem.value = best_score;
                book_elem.level = best_level;
                node_updated = true;
            }
            if (node_updated){
                mtx.lock();
                    register_symmetric_book(board, book_elem);
                mtx.unlock();
            }
            return res;
        }


    private:
        void reg_first_board(){
            Board board;
            board.reset();
            Book_elem elem;
            elem.value = 0;
            elem.level = LEVEL_HUMAN;
            Book_value move;
            move.value = 0;
            move.policy = 19;
            elem.moves.emplace_back(move);
            move.policy = 26;
            elem.moves.emplace_back(move);
            move.policy = 37;
            elem.moves.emplace_back(move);
            move.policy = 44;
            elem.moves.emplace_back(move);
            book[board] = elem;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_book(Board b, Book_elem elem){
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

        inline Board get_representative_board(Board *b, int *idx){
            return get_representative_board(b->copy(), idx);
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
            for (Book_value &move: elem.moves)
                move.policy = convert_coord_to_representative_board(move.policy, idx);
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

        inline int merge(Board b, Book_elem elem){
            if (!contain_symmetry(b))
                return register_symmetric_book(b, elem);
            Book_elem book_elem = get(b);
            book_elem.value = elem.value;
            book_elem.level = elem.level;
            for (Book_value &move: elem.moves){
                bool already_registered = false;
                for (int i = 0; i < (int)book_elem.moves.size(); ++i){
                    if (book_elem.moves[i].policy == move.policy){
                        already_registered = true;
                        book_elem.moves[i].value = move.value;
                        break;
                    }
                }
                if (!already_registered){
                    book_elem.moves.emplace_back(move);
                }
            }
            return register_symmetric_book(b, book_elem);
        }

        int count_n_line(Board board){
            auto itr = n_lines.find(board);
            if (itr != n_lines.end())
                return itr->second;
            Flip flip;
            Book_elem book_elem = get(board);
            if (book_elem.moves.size() == 0)
                return 1;
            int res = 0;
            for (Book_value &move: book_elem.moves){
                calc_flip(&flip, &board, move.policy);
                board.move_board(&flip);
                    if (contain_symmetry(&board))
                        res += count_n_line(board);
                board.undo_board(&flip);
            }
            n_lines[board] = res;
            return res;
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

void book_fix(bool *stop){
    book.fix(stop);
}

Book_negamax negamax_book_global(Board board, bool *stop){
    return book.negamax_book(board, stop);
}