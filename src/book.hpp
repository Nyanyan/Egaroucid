#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "board.hpp"

#define book_hash_table_size 8192
constexpr int book_hash_mask = book_hash_table_size - 1;
#define book_stones 64

struct book_node{
    uint_fast16_t k[hw];
    int policies[35];
    int size;
    book_node* p_n_node;
};

class book{
    private:
        book_node *book[book_hash_table_size];
    public:
        void init(){
            int i;
            unordered_map<char, int> char_keys;
            const string book_chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abc";
            string param_compressed1;
            for (i = 0; i < hw2; ++i)
                char_keys[book_chars[i]] = i;
            ifstream ifs("resources/book.txt");
            if (ifs.fail()){
                cerr << "book file not exist" << endl;
                exit(1);
            }
            getline(ifs, param_compressed1);
            int ln = param_compressed1.length();
            int coord;
            board fb;
            const int first_board[b_idx_num] = {6560, 6560, 6560, 6425, 6326, 6560, 6560, 6560, 6560, 6560, 6560, 6425, 6344, 6506, 6560, 6560, 6560, 6560, 6560, 6560, 6344, 6425, 6398, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6479, 6344, 6398, 6074, 6560, 6560, 6560};
            for(i = 0; i < book_hash_table_size; ++i)
                this->book[i] = NULL;
            int data_idx = 0;
            int n_book = 0;
            int y, x;
            int tmp[16];
            while (data_idx < ln){
                fb.p = 1;
                for (i = 0; i < b_idx_num; ++i)
                    fb.b[i] = first_board[i];
                while (true){
                    if (param_compressed1[data_idx] == ' '){
                        ++data_idx;
                        break;
                    }
                    coord = char_keys[param_compressed1[data_idx++]];
                    fb = fb.move(coord);
                }
                coord = char_keys[param_compressed1[data_idx++]];
                y = coord / hw;
                x = coord % hw;
                register_book(fb.b, fb.hash() & book_hash_mask, y * hw + x);
                for (i = 0; i < 8; ++i)
                    swap(fb.b[i], fb.b[8 + i]);
                register_book(fb.b, fb.hash() & book_hash_mask, x * hw + y);
                for (i = 0; i < 16; ++i)
                    tmp[i] = fb.b[i];
                for (i = 0; i < 8; ++i)
                    fb.b[i] = reverse_board[tmp[7 - i]];
                for (i = 0; i < 8; ++i)
                    fb.b[8 + i] = reverse_board[tmp[15 - i]];
                register_book(fb.b, fb.hash() & book_hash_mask, (hw_m1 - x) * hw + (hw_m1 - y));
                for (i = 0; i < 8; ++i)
                    swap(fb.b[i], fb.b[8 + i]);
                register_book(fb.b, fb.hash() & book_hash_mask, (hw_m1 - y) * hw + (hw_m1 - x));
                n_book += 4;
            }
            cerr << "book initialized " << n_book << " boards in book" << endl;
        }

        inline int get(board *b){
            book_node *p_node = this->book[b->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b->b, p_node->k))
                    return p_node->policies[myrandrange(0, p_node->size)];
                p_node = p_node->p_n_node;
            }
            return -1;
        }

    private:
        inline bool compare_key(const uint_fast16_t a[], const uint_fast16_t b[]){
            return
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && 
                a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7];
        }

        inline book_node* book_node_init(const uint_fast16_t key[], int policy){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            for (int i = 0; i < hw; ++i)
                p_node->k[i] = key[i];
            p_node->policies[0] = policy;
            p_node->size = 1;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline void register_book(const uint_fast16_t key[], int hash, int policy){
            if(this->book[hash] == NULL){
                this->book[hash] = book_node_init(key, policy);
            } else {
                book_node *p_node = this->book[hash];
                book_node *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(compare_key(key, p_node->k)){
                        p_node->policies[p_node->size++] = policy;
                        return;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = book_node_init(key, policy);
            }
        }
};

book book;

inline void book_init(){
    book.init();
}