/*
    Egaroucid Project

    @date 2021-2023
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "evaluate.hpp"
#include "board.hpp"
#include "book_const.hpp"

struct Book_value{
    int policy;
    int value;
};

class Book{
    private:
        unordered_map<Board, int, Board_hash> book;
        int n_book;

    public:
        void init(){
            n_book = 0;
            import_book_const();
        }

        inline void import_book_const(){
            Board b;
            int i, value;
            uint64_t p, o;
            uint8_t elem;
            for (i = 0; i < N_EMBED_BOOK; ++i) {
                b.player = embed_book[i].player;
                b.opponent = embed_book[i].opponent;
                n_book += register_symmetric_book(b, embed_book[i].value, n_book);
            }
            cerr << "book imported " << n_book << " boards" << endl;
        }

        inline void reg(Board b, int value){
            n_book += register_symmetric_book(b, value, n_book);
        }

        inline void reg(Board *b, int value){
            Board b1 = b->copy();
            n_book += register_symmetric_book(b1, value, n_book);
        }

        inline int get_onebook(Board b){
            if (book.find(b) == book.end())
                return -INF;
            return book[b];
        }

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

        inline Book_value get_random(Board *b, int accept_value){
            vector<int> policies;
            vector<int> values;
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
                    max_value = max(max_value, value);
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

        inline int get_n_book(){
            return n_book;
        }

    private:
        inline bool register_book(Board b, int value){
            bool res = book.find(b) == book.end();
            book[b] = value;
            return res;
        }

        inline bool delete_book(Board b){
            if (book.find(b) != book.end()){
                book.erase(b);
                return true;
            }
            return false;
        }

        inline int register_symmetric_book(Board b, int value, int line){
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

void book_init(){
    book.init();
}