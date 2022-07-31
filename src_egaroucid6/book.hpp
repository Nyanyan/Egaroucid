#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"

#define BOOK_HASH_TABLE_SIZE 67108864
#define BOOK_HASH_MASK 67108863


struct Node_book{
    uint64_t player;
    uint64_t opponent;
    int value;
    int line;
    Node_book* p_n_node;
    void init(){
        if (p_n_node != NULL)
            p_n_node->init();
        free(this);
    }
};

struct Book_value{
    int policy;
    int value;
};

class Book{
    private:
        Node_book *book[BOOK_HASH_TABLE_SIZE];
        int n_book;
        int n_hash_conflict;
    public:
        bool init(string file){
            for (int i = 0; i < BOOK_HASH_TABLE_SIZE; ++i)
                this->book[i] = NULL;
            n_book = 0;
            n_hash_conflict = 0;
            return import_file_bin(file);
        }

        inline bool import_file_bin(string file){
            FILE* fp;
            #ifdef _WIN64
                if (fopen_s(&fp, file.c_str(), "rb") != 0) {
                    cerr << "can't open " << file << endl;
                    return false;
                }
            #else
                fp = fopen(file.c_str(), "rb");
                if (fp == NULL){
                    cerr << "can't open " << file << endl;
                    return false;
                }
            #endif
            Board b;
            int n_boards, i, value;
            uint64_t p, o;
            unsigned char elem;
            if (fread(&n_boards, 4, 1, fp) < 1){
                cerr << "book NOT FULLY imported " << n_book << " boards code 0" << endl;
                fclose(fp);
                return false;
            }
            for (i = 0; i < n_boards; ++i) {
                if (i % 32768 == 0)
                    cerr << "loading book " << (i * 100 / n_boards) << "%" << endl;
                if (fread(&p, 8, 1, fp) < 1) {
                    cerr << "book NOT FULLY imported " << n_book << " boards code 1" << endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&o, 8, 1, fp) < 1) {
                    cerr << "book NOT FULLY imported " << n_book << " boards code 2" << endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem, 1, 1, fp) < 1) {
                    cerr << "book NOT FULLY imported " << n_book << " boards code 3" << endl;
                    fclose(fp);
                    return false;
                }
                value = elem - HW2;
                if (value < -HW2 || HW2 < value) {
                    cerr << "book NOT FULLY imported " << n_book << " boards code 4 got value " << value << endl;
                    fclose(fp);
                    return false;
                }
                b.player = p;
                b.opponent = o;
                n_book += register_symmetric_book(b, value, n_book);
            }
            cerr << "book imported " << n_book << " boards hash conflict " << n_hash_conflict << endl;
            fclose(fp);
            return true;
        }
        
        inline bool import_edax_book(string file) {
            FILE* fp;
            #ifdef _WIN64
                if (fopen_s(&fp, file.c_str(), "rb") != 0) {
                    cerr << "can't open " << file << endl;
                    return false;
                }
            #else
                fp = fopen(file.c_str(), "rb");
                if (fp == NULL){
                    cerr << "can't open " << file << endl;
                    return false;
                }
            #endif
            char elem_char;
            int elem_int;
            short elem_short;
            int i, j;
            for (i = 0; i < 38; ++i){
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&elem_int, 4, 1, fp) < 1) {
                cerr << "file broken" << endl;
                fclose(fp);
                return false;
            }
            int n_boards = elem_int;
            uint64_t player, opponent;
            short value;
            char link = 0, link_value, link_move;
            Board b;
            Flip flip;
            for (i = 0; i < n_boards; ++i){
                if (i % 32768 == 0)
                    cerr << "loading edax book " << (i * 100 / n_boards) << "%" << endl;
                if (fread(&player, 8, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&opponent, 8, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 4; ++j) {
                    if (fread(&elem_int, 4, 1, fp) < 1) {
                        cerr << "file broken" << endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&value, 2, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
                for (j = 0; j < 2; ++j) {
                    if (fread(&elem_short, 2, 1, fp) < 1) {
                        cerr << "file broken" << endl;
                        fclose(fp);
                        return false;
                    }
                }
                if (fread(&link, 1, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    cerr << "file broken" << endl;
                    fclose(fp);
                    return false;
                }
                b.player = player;
                b.opponent = opponent;
                n_book += register_symmetric_book(b, -(int)value, n_book);
                for (j = 0; j < (int)link + 1; ++j) {
                    if (fread(&link_value, 1, 1, fp) < 1) {
                        cerr << "file broken" << endl;
                        fclose(fp);
                        return false;
                    }
                    if (fread(&link_move, 1, 1, fp) < 1) {
                        cerr << "file broken" << endl;
                        fclose(fp);
                        return false;
                    }
                    if (link_move < HW2) {
                        calc_flip(&flip, &b, (int)link_move);
                        if (flip.flip == 0ULL){
                            cerr << "error! illegal move" << endl;
                            return false;
                        }
                        b.move_board(&flip);
                            n_book += register_symmetric_book(b, (int)link_value, n_book);
                        b.undo_board(&flip);
                    }
                }
            }
            cerr << "book imported " << n_book << " boards hash conflict " << n_hash_conflict << endl;
            return true;
        }

        inline void reg(Board b, int value){
            n_book += register_symmetric_book(b, value, n_book);
        }

        inline void reg(Board *b, int value){
            Board b1 = b->copy();
            n_book += register_symmetric_book(b1, value, n_book);
        }

        inline int get_onebook(Board *b){
            Node_book *p_node = this->book[b->hash() & BOOK_HASH_MASK];
            while(p_node != NULL){
                if(compare_key(b, p_node)){
                    return p_node->value;
                }
                p_node = p_node->p_n_node;
            }
            return -INF;
        }

        inline int get(Board *b){
            Board nb = b->copy();
            int res = -INF;
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_horizontal_mirror();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_onebook(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_onebook(&nb);
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

        inline int get_oneline(Board *b){
            Node_book *p_node = this->book[b->hash() & BOOK_HASH_MASK];
            while(p_node != NULL){
                if(compare_key(b, p_node)){
                    return p_node->line;
                }
                p_node = p_node->p_n_node;
            }
            return -INF;
        }

        inline int get_line(Board *b){
            Board nb = b->copy();
            int res = -INF;
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_horizontal_mirror();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_rotate_180();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            nb.board_black_line_mirror();
            res = get_oneline(&nb);
            if (res != -INF)
                return res;
            return -INF;
        }

        inline void change(Board b, int value){
            if (register_symmetric_book(b, value, n_book)){
                n_book++;
                cerr << "book registered " << n_book << endl;
            } else
                cerr << "book changed " << n_book << endl;
        }

        inline void change(Board *b, int value){
            Board nb = b->copy();
            if (register_symmetric_book(nb, value, n_book)){
                n_book++;
                cerr << "book registered " << n_book << endl;
            } else
                cerr << "book changed " << n_book << endl;
        }

        inline void delete_elem(Board b){
            if (delete_symmetric_book(b)){
                n_book--;
                cerr << "deleted book elem " << n_book << endl;
            } else
                cerr << "book elem NOT deleted " << n_book << endl;
        }

        inline void delete_all(){
            int t = 0;
            for (uint64_t i = 0; i < BOOK_HASH_TABLE_SIZE; ++i){
                if (i % 1048576 == 0)
                    cerr << "clearing book " << (i * 100 / BOOK_HASH_TABLE_SIZE) << "%" << endl;
                if (this->book[i] != NULL)
                    this->book[i]->init();
                this->book[i] = NULL;
            }
            cerr << "deleted " << t << " boards" << endl;
        }

        inline void save_bin(string file, string bak_file){
            if (remove(bak_file.c_str()) == -1)
                cerr << "cannot delete book_backup.egbk" << endl;
            rename(file.c_str(), bak_file.c_str());
            ofstream fout;
            fout.open(file.c_str(), ios::out|ios::binary|ios::trunc);
            if (!fout){
                cerr << "can't open book.egbk" << endl;
                return;
            }
            uint64_t i;
            unsigned char elem;
            fout.write((char*)&n_book, 4);
            int t = 0;
            for (i = 0; i < BOOK_HASH_TABLE_SIZE; ++i){
                if (i % 1048576 == 0)
                    cerr << "saving book " << (i * 100 / BOOK_HASH_TABLE_SIZE) << "%" << endl;
                Node_book *p_node = this->book[i];
                while(p_node != NULL){
                    fout.write((char*)&p_node->player, 8);
                    fout.write((char*)&p_node->opponent, 8);
                    elem = max(0, min(HW2 * 2, p_node->value + HW2));
                    fout.write((char*)&elem, 1);
                    ++t;
                    p_node = p_node->p_n_node;
                }
            }
            fout.close();
            cerr << "saved " << t << " boards" << endl;
        }
        

    private:
        inline bool compare_key(const Board *a, const Node_book *b){
            return a->player == b->player && a->opponent == b->opponent;
        }

        inline Node_book* Node_book_init(Board b, int value, int line){
            Node_book* p_node = NULL;
            p_node = (Node_book*)malloc(sizeof(Node_book));
            p_node->player = b.player;
            p_node->opponent = b.opponent;
            p_node->value = value;
            p_node->line = line;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline bool register_book(Board b, int hash, int value, int line){
            if(this->book[hash] == NULL){
                this->book[hash] = Node_book_init(b, value, line);
            } else {
                Node_book *p_node = this->book[hash];
                Node_book *p_pre_node = NULL;
                p_pre_node = p_node;
                int delta_conflict = 0;
                while(p_node != NULL){
                    if(compare_key(&b, p_node)){
                        p_node->value = value;
                        return false;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                    ++delta_conflict;
                }
                n_hash_conflict += delta_conflict;
                p_pre_node->p_n_node = Node_book_init(b, value, line);
            }
            return true;
        }

        inline bool delete_book(Board b, int hash){
            if(this->book[hash] != NULL){
                Node_book *p_node = this->book[hash];
                Node_book *p_pre_node = NULL;
                while(p_node != NULL){
                    if(compare_key(&b, p_node)){
                        if (p_pre_node != NULL){
                            cerr << "pre node exist" << endl;
                            p_pre_node->p_n_node = p_node->p_n_node;
                            free(p_node);
                        } else{
                            cerr << "first node" << endl;
                            free(p_node);
                            this->book[hash] = NULL;
                        }
                        return true;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
            }
            return false;
        }

        inline int register_symmetric_book(Board b, int value, int line){
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_rotate_180();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_horizontal_mirror();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_rotate_180();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            b.board_black_line_mirror();
            if (get_onebook(&b) != -INF){
                register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
                return 0;
            }
            register_book(b, b.hash() & BOOK_HASH_MASK, value, line);
            return 1;
        }

        inline int delete_symmetric_book(Board b){
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_rotate_180();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_horizontal_mirror();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_rotate_180();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            b.board_black_line_mirror();
            if (delete_book(b, b.hash() & BOOK_HASH_MASK)){
                return 1;
            }
            return 0;
        }

        inline void create_arr(Node_book *node, int arr[], int p){
            Board b;
            b.player = node->player;
            b.opponent = node->opponent;
            b.translate_to_arr_player(arr);
        }
};

Book book;

bool book_init(string file){
    return book.init(file);
}
