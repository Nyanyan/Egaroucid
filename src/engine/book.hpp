/*
    Egaroucid Project

    @file book.hpp
        Book class
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "evaluate.hpp"
#include "board.hpp"

#define BOOK_N_ACCEPT_LEVEL 11
#define BOOK_ACCURACY_LEVEL_INF 10

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
    @param n_book               number of boards registered
*/
class Book{
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
            n_book = 0;
            return import_file_bin(file, show_log, stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk file)
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
            char link = 0, link_value, link_move;
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
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                b.player = player;
                b.opponent = opponent;
                n_book += register_symmetric_book(b, -(int)value);
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
                        calc_flip(&flip, &b, (int)link_move);
                        if (flip.flip == 0ULL){
                            std::cerr << "error! illegal move" << std::endl;
                            return false;
                        }
                        b.move_board(&flip);
                            n_book += register_symmetric_book(b, (int)link_value);
                        b.undo_board(&flip);
                    }
                }
            }
            if (show_log)
                std::cerr << "book imported " << n_book << " boards" << std::endl;
            return true;
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
            Board min_board = get_min_board(b);
            return get_onebook(min_board);
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
            for (std::pair<double, int> &elem: value_policies){
                elem.first /= sum_exp_values;
                std::cerr << elem.first << " " << idx_to_coord(elem.second) << std::endl;
            }
            double rnd = myrandom();
            std::cerr << "rnd " << rnd << std::endl;
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
            n_book = 0;
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
                std::cerr << "can't open book.egbk" << std::endl;
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
        

    private:
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

        inline void update_min_board(Board *res, Board *sym){
            if ((res->player | res->opponent) > (sym->player | sym->opponent))
                *res = sym->copy();
        }

        inline Board get_min_board(Board b){
            Board min_board = b;
            b.board_black_line_mirror();
            update_min_board(&min_board, &b);
            b.board_rotate_180();
            update_min_board(&min_board, &b);
            b.board_black_line_mirror();
            update_min_board(&min_board, &b);
            b.board_horizontal_mirror();
            update_min_board(&min_board, &b);
            b.board_black_line_mirror();
            update_min_board(&min_board, &b);
            b.board_rotate_180();
            update_min_board(&min_board, &b);
            b.board_black_line_mirror();
            update_min_board(&min_board, &b);
            return min_board;
        }

        inline Board get_min_board(Board *b){
            return get_min_board(b->copy());
        }

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book(Board b, int value){
            Board min_board = get_min_board(b);
            register_book(min_board, value);
            return 1;
        }

        /*
            @brief delete a board with checking all symmetry boards

            @param b                    a board to delete
            @return 1 if board is deleted (board found) else 0
        */
        inline int delete_symmetric_book(Board b){
            Board min_board = get_min_board(b);
            return delete_book(min_board);
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