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
    double value;
    book_node* p_n_node;
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
            int tmp[b_idx_num];
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
                value = stof(value_str);
                register_book(b.b, b.hash() & book_hash_mask, value);
                for (i = 0; i < 8; ++i)
                    swap(b.b[i], b.b[8 + i]);
                register_book(b.b, b.hash() & book_hash_mask, value);
                for (i = 0; i < 16; ++i)
                    tmp[i] = b.b[i];
                for (i = 0; i < 8; ++i)
                    b.b[i] = reverse_board[tmp[7 - i]];
                for (i = 0; i < 8; ++i)
                    b.b[8 + i] = reverse_board[tmp[15 - i]];
                register_book(b.b, b.hash() & book_hash_mask, value);
                for (i = 0; i < 8; ++i)
                    swap(b.b[i], b.b[8 + i]);
                register_book(b.b, b.hash() & book_hash_mask, value);
                ++n_book;
            }
            cerr << "book initialized " << n_book << " boards in book" << endl;
        }

        inline int get(board *b){
            book_node *p_node = this->book[b->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b->b, p_node->k)){
                    return (b->p ? -1.0 : 1.0) * p_node->value * step;
                }
                p_node = p_node->p_n_node;
            }
            return -inf;
        }

        inline book_value get_random(board *b, double accept_value){
            vector<int> policies;
            vector<double> values;
            board nb;
            double max_value = -inf;
            for (int coord = 0; coord < hw2; ++coord){
                if (b->legal(coord)){
                    b->move(coord, &nb);
                    book_node *p_node = this->book[nb.hash() & book_hash_mask];
                    while(p_node != NULL){
                        if(compare_key(nb.b, p_node->k)){
                            policies.push_back(coord);
                            values.push_back((b->p ? -1.0 : 1.0) * p_node->value);
                            max_value = max(max_value, (b->p ? -1.0 : 1.0) * p_node->value);
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
                idx = myrandrange(0, policies.size());
                if (values[idx] >= max_value - accept_value){
                    res.policy = policies[idx];
                    res.value = values[idx] * step;
                    break;
                }
            }
            return res;
        }

        inline book_value get_half_random(board *b){
            vector<int> policies;
            vector<double> values;
            board nb;
            int max_val = -inf, value;
            for (int coord = 0; coord < hw2; ++coord){
                if (b->legal(coord)){
                    b->move(coord, &nb);
                    book_node *p_node = this->book[nb.hash() & book_hash_mask];
                    while(p_node != NULL){
                        if(compare_key(nb.b, p_node->k)){
                            value = round((b->p ? -1.0 : 1.0) * p_node->value);
                            if (value == max_val){
                                policies.push_back(coord);
                                values.push_back((b->p ? -1.0 : 1.0) * p_node->value);
                            } else if (value > max_val){
                                max_val = value;
                                policies.clear();
                                values.clear();
                                policies.push_back(coord);
                                values.push_back((b->p ? -1.0 : 1.0) * p_node->value);
                            }
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
            int idx = myrandrange(0, policies.size());
            res.policy = policies[idx];
            res.value = values[idx] * step;
            return res;
        }

        inline book_value get_exact(board *b){
            book_value res;
            board nb;
            double max_val = -inf, value;
            for (int coord = 0; coord < hw2; ++coord){
                if (b->legal(coord)){
                    b->move(coord, &nb);
                    book_node *p_node = this->book[nb.hash() & book_hash_mask];
                    while(p_node != NULL){
                        if(compare_key(nb.b, p_node->k)){
                            value = (b->p ? -1.0 : 1.0) * p_node->value;
                            if (value > max_val){
                                max_val = value;
                                res.policy = coord;
                                res.value = value * step;
                            }
                            break;
                        }
                        p_node = p_node->p_n_node;
                    }
                }
            }
            return res;
        }

    private:
        inline bool compare_key(const uint_fast16_t a[], const uint_fast16_t b[]){
            return
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && 
                a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7];
        }

        inline book_node* book_node_init(const uint_fast16_t key[], double value){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            for (int i = 0; i < hw; ++i)
                p_node->k[i] = key[i];
            p_node->value = value;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline void register_book(const uint_fast16_t key[], int hash, double value){
            if(this->book[hash] == NULL){
                this->book[hash] = book_node_init(key, value);
            } else {
                book_node *p_node = this->book[hash];
                book_node *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(compare_key(key, p_node->k)){
                        p_node->value = value;
                        return;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = book_node_init(key, value);
            }
        }
};

book book;

inline void book_init(){
    book.init();
}