#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "evaluate.hpp"
#include "board.hpp"

#define book_hash_table_size 8192
constexpr int book_hash_mask = book_hash_table_size - 1;
#define book_stones 64

struct book_node{
    uint_fast16_t k[hw];
    int value;
    book_node* p_n_node;
    int line;
};

struct book_value{
    int policy;
    int value;
};

class book{
    private:
        book_node *book[book_hash_table_size];
    public:
        void init(){
            int i, j, k;
            unsigned long long bk, wt;
            string book_str;
            char elem;
            ifstream ifs("resources/book.txt");
            if (ifs.fail()){
                cerr << "book file not exist" << endl;
                exit(1);
            }
            string book_line;
            int n_book = 0;
            board b;
            double value;
            for(i = 0; i < book_hash_table_size; ++i)
                this->book[i] = NULL;
            while (getline(ifs, book_line)){
                bk = 0;
                wt = 0;
                for (j = 0; j < hw2; ++j){
                    elem = book_line[j];
                    if (elem != '.'){
                        bk |= (unsigned long long)(elem == '0') << j;
                        wt |= (unsigned long long)(elem == '1') << j;
                    }
                }
                for (j = 0; j < b_idx_num; ++j){
                    b.b[j] = n_line - 1;
                    for (k = 0; k < idx_n_cell[j]; ++k){
                        if (1 & (bk >> global_place[j][k]))
                            b.b[j] -= pow3[hw_m1 - k] * 2;
                        else if (1 & (wt >> global_place[j][k]))
                            b.b[j] -= pow3[hw_m1 - k];
                    }
                }
                string value_str = "";
                for (j = hw2 + 1; j < (int)book_line.size(); ++j)
                    value_str += book_line[j];
                value = stoi(value_str);
                register_symmetric_book(b, value, n_book);
                ++n_book;
            }
            cerr << "book initialized " << n_book << " boards in book" << endl;
        }

        inline int get(board *b){
            book_node *p_node = this->book[b->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b->b, p_node->k)){
                    return (b->p ? -1 : 1) * p_node->value * step;
                }
                p_node = p_node->p_n_node;
            }
            return -inf;
        }

        inline book_value get_random(board *b, int accept_value){
            vector<int> policies;
            vector<int> values;
            board nb;
            int max_value = -inf;
            for (int coord = 0; coord < hw2; ++coord){
                if (b->legal(coord)){
                    b->move(coord, &nb);
                    book_node *p_node = this->book[nb.hash() & book_hash_mask];
                    while(p_node != NULL){
                        if(compare_key(nb.b, p_node->k)){
                            policies.push_back(coord);
                            values.push_back((b->p ? -1 : 1) * p_node->value);
                            max_value = max(max_value, (b->p ? -1 : 1) * p_node->value);
                            break;
                        }
                        p_node = p_node->p_n_node;
                    }
                }
            }
            book_value res;
            if (policies.size() == 0){
                res.policy = -1;
                res.value = -inf;
                return res;
            }
            int idx;
            while (true){
                idx = myrandrange(0, (int)policies.size());
                if (values[idx] >= max_value - accept_value){
                    res.policy = policies[idx];
                    res.value = values[idx] * step;
                    break;
                }
            }
            return res;
        }

        inline void change(board b, int value){
            if (b.p)
                value = -value;
            book_node *p_node = this->book[b.hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b.b, p_node->k)){
                    register_symmetric_book(b, value, p_node->line);
                    save_book(b, value, p_node->line);
                    return;
                }
                p_node = p_node->p_n_node;
            }
            register_symmetric_book(b, value, -1);
            save_book(b, value, -1);
        }

    private:
        inline bool compare_key(const uint_fast16_t a[], const uint_fast16_t b[]){
            return
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && 
                a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7];
        }

        inline book_node* book_node_init(const uint_fast16_t key[], int value, int line){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            for (int i = 0; i < hw; ++i)
                p_node->k[i] = key[i];
            p_node->value = value;
            p_node->line = line;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline void register_book(const uint_fast16_t key[], int hash, int value, int line){
            if(this->book[hash] == NULL){
                this->book[hash] = book_node_init(key, value, line);
            } else {
                book_node *p_node = this->book[hash];
                book_node *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(compare_key(key, p_node->k)){
                        p_node->value = value;
                        p_node->line = line;
                        return;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = book_node_init(key, value, line);
            }
        }

        inline void register_symmetric_book(board b, int value, int line){
            int i;
            int tmp[b_idx_num];
            register_book(b.b, b.hash() & book_hash_mask, value, line);
            for (i = 0; i < 8; ++i)
                swap(b.b[i], b.b[8 + i]);
            register_book(b.b, b.hash() & book_hash_mask, value, line);
            for (i = 0; i < 16; ++i)
                tmp[i] = b.b[i];
            for (i = 0; i < 8; ++i)
                b.b[i] = reverse_board[tmp[7 - i]];
            for (i = 0; i < 8; ++i)
                b.b[8 + i] = reverse_board[tmp[15 - i]];
            register_book(b.b, b.hash() & book_hash_mask, value, line);
            for (i = 0; i < 8; ++i)
                swap(b.b[i], b.b[8 + i]);
            register_book(b.b, b.hash() & book_hash_mask, value, line);
        }

        inline string create_book_data(board b, int value){
            string res = "";
            int arr[hw2];
            b.translate_to_arr(arr);
            for (int i = 0; i < hw2; ++i){
                if (arr[i] == black)
                    res += to_string(black);
                else if (arr[i] == white)
                    res += to_string(white);
                else
                    res += ".";
            }
            res += " ";
            res += to_string(value);
            return res;
        }

        inline void save_book(board b, int value, int line){
            remove("resources/book_backup.txt");
            rename("resources/book.txt", "resources/book_backup.txt");
            ifstream ifs("resources/book_backup.txt");
            if (ifs.fail()){
                cerr << "book file not exist" << endl;
                exit(1);
            }
            ofstream ofs("resources/book.txt");
            if (ofs.fail()){
                cerr << "book file not exist" << endl;
                exit(1);
            }
            int idx = 0;
            string book_line;
            while (getline(ifs, book_line)){
                if (idx == line)
                    ofs << create_book_data(b, value) << endl;
                else
                    ofs << book_line << endl;
                ++idx;
            }
            if (line == -1)
                ofs << create_book_data(b, value) << endl;
        }
};

book book;

inline void book_init(){
    book.init();
}