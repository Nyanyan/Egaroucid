/*
    Egaroucid Project

    @file book.hpp
        Book class
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "evaluate.hpp"
#include "board.hpp"

/*
    @brief book result structure

    @param policy               selected best move
    @param value                registered score
*/
struct Book_value{
    int policy;
    int value;
};

class Book{
    private:
        /*
            @brief book data

            @param book                 book data
            @param n_book               number of boards registered
        */
        std::unordered_map<Board, int, Board_hash> book;
        int n_book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file){
            n_book = 0;
            return import_file_bin(file);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_bin(std::string file){
            std::cerr << file << std::endl;
            FILE* fp;
            #ifdef _WIN64
                if (fopen_s(&fp, file.c_str(), "rb") != 0) {
                    std::cerr << "can't open " << file << std::endl;
                    return false;
                }
            #else
                fp = fopen(file.c_str(), "rb");
                if (fp == NULL){
                    std::cerr << "can't open " << file << std::endl;
                    return false;
                }
            #endif
            Board b;
            int n_boards, i, value;
            uint64_t p, o;
            uint8_t elem;
            if (fread(&n_boards, 4, 1, fp) < 1){
                std::cerr << "book NOT FULLY imported " << n_book << " boards code 0" << std::endl;
                fclose(fp);
                return false;
            }
            for (i = 0; i < n_boards; ++i) {
                if (i % 32768 == 0)
                    std::cerr << "loading book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "book NOT FULLY imported " << n_book << " boards code 1" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "book NOT FULLY imported " << n_book << " boards code 2" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem, 1, 1, fp) < 1) {
                    std::cerr << "book NOT FULLY imported " << n_book << " boards code 3" << std::endl;
                    fclose(fp);
                    return false;
                }
                value = elem - HW2;
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "book NOT FULLY imported " << n_book << " boards code 4 got value " << value << std::endl;
                    fclose(fp);
                    return false;
                }
                b.player = p;
                b.opponent = o;
                n_book += register_symmetric_book(b, value);
            }
            std::cerr << "book imported " << n_book << " boards" << std::endl;
            fclose(fp);
            return true;
        }

        /*
            @brief import Edax-formatted book

            @param file                 book file (.dat file)
            @return book completely imported?
        */
        inline bool import_edax_book(std::string file) {
            std::cerr << file << std::endl;
            FILE* fp;
            #ifdef _WIN64
                if (fopen_s(&fp, file.c_str(), "rb") != 0) {
                    std::cerr << "can't open " << file << std::endl;
                    return false;
                }
            #else
                fp = fopen(file.c_str(), "rb");
                if (fp == NULL){
                    std::cerr << "can't open " << file << std::endl;
                    return false;
                }
            #endif
            char elem_char;
            int elem_int;
            int16_t elem_short;
            int i, j;
            for (i = 0; i < 38; ++i){
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&elem_int, 4, 1, fp) < 1) {
                std::cerr << "file broken" << std::endl;
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
                if (i % 32768 == 0)
                    std::cerr << "loading edax book " << (i * 100 / n_boards) << "%" << std::endl;
                if (fread(&player, 8, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&opponent, 8, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 4; ++j) {
                    if (fread(&elem_int, 4, 1, fp) < 1) {
                        std::cerr << "file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&value, 2, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 2; ++j) {
                    if (fread(&elem_short, 2, 1, fp) < 1) {
                        std::cerr << "file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&link, 1, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                b.player = player;
                b.opponent = opponent;
                n_book += register_symmetric_book(b, -(int)value);
                for (j = 0; j < (int)link + 1; ++j) {
                    if (fread(&link_value, 1, 1, fp) < 1) {
                        std::cerr << "file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    if (fread(&link_move, 1, 1, fp) < 1) {
                        std::cerr << "file broken" << std::endl;
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
            Board nb = b->copy();
            int res = -INF;
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_horizontal_mirror();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(nb);
            if (res != -INF)
                return res;
            return -INF;
        }

        /*
            @brief get a best move

            @param b                    a board pointer to find
            @param accept_value         an error to allow
            @return best move and value as Book_value structure
        */
        inline Book_value get_random(Board *b, int accept_value){
            std::vector<int> policies;
            std::vector<int> values;
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
                    policies.push_back(cell);
                    values.push_back(value);
                    max_value = std::max(max_value, value);
                }
            }
            Book_value res;
            if (policies.size() == 0){
                res.policy = -1;
                res.value = -INF;
                return res;
            }
            int idx;
            while (true){
                idx = myrandrange(0, (int)policies.size());
                if (values[idx] >= max_value - accept_value){
                    res.policy = policies[idx];
                    res.value = values[idx];
                    break;
                }
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
            uint64_t i;
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
            bool res = book.find(b) == book.end();
            book[b] = value;
            return res;
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

        /*
            @brief register a board with checking all symmetry boards

            @param b                    a board to register
            @param value                score of the board
            @return 1 if board is new else 0
        */
        inline int register_symmetric_book(Board b, int value){
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_rotate_180();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_horizontal_mirror();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_rotate_180();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(b) != -INF){
                register_book(b, value);
                return 0;
            }
            register_book(b, value);
            return 1;
        }

        /*
            @brief delete a board with checking all symmetry boards

            @param b                    a board to delete
            @return 1 if board is deleted (board found) else 0
        */
        inline int delete_symmetric_book(Board b){
            if (delete_book(b)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b)){
                return 1;
            }
            b.board_rotate_180();
            if (delete_book(b)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b)){
                return 1;
            }
            b.board_horizontal_mirror();
            if (delete_book(b)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b)){
                return 1;
            }
            b.board_rotate_180();
            if (delete_book(b)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b)){
                return 1;
            }
            return 0;
        }
};

Book book;

bool book_init(std::string file){
    return book.init(file);
}
