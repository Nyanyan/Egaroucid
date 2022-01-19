#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"

#define book_hash_table_size 32768
constexpr int book_hash_mask = book_hash_table_size - 1;
#define book_stones 64

struct book_node{
    uint_fast16_t k[hw];
    int p;
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
		int n_book;
    public:
        bool init(){
			for (int i = 0; i < book_hash_table_size; ++i)
				this->book[i] = NULL;
			n_book = 0;
            return import_file("resources/book.txt");
        }

        inline bool import_file(string file){
            int j;
            int board_arr[hw2];
            string book_str;
            char elem;
            ifstream ifs(file);
            if (ifs.fail()){
                cerr << "book file not exist" << endl;
				return false;
            }
            string book_line;
            board b;
            double value;
            int p;
            while (getline(ifs, book_line)){
                if (book_line.size() < hw2 + 4){
                    cerr << "book import error 0" << endl;
                    return false;
                }
                for (j = 0; j < hw2; ++j){
                    elem = book_line[j];
                    if (elem == '0')
                        board_arr[j] = 0;
                    else if (elem == '1')
                        board_arr[j] = 1;
                    else if (elem == '.')
                        board_arr[j] = 2;
                    else{
                        cerr << "book import error 1" << endl;
                        return false;
                    }
                }
                p = (int)book_line[hw2 + 1] - (int)'0';
                if (p != 0 && p != 1){
                    cerr << "book import error 2" << endl;
                    return false;
                }
                b.translate_from_arr_fast(board_arr, p);
                string value_str = book_line.substr(hw2 + 3, book_line.size() - hw2 - 3);
                try{
                    value = stoi(value_str);
                } catch (const invalid_argument& err){
                    cerr << "book import error 3" << endl;
                    return false;
                }
                n_book += register_symmetric_book(b, value, n_book);
            }
            cerr << "book initialized " << n_book << " boards in book" << endl;
            return true;
        }

        inline void reg(board b, int value){
            n_book += register_symmetric_book(b, (b.p ? -1 : 1) * value, n_book);
        }

        inline int get(board *b){
            book_node *p_node = this->book[b->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b->b, p_node->k)){
                    return (b->p ? -1 : 1) * p_node->value;
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
                    res.value = values[idx];
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
                    //save_book(b, value, p_node->line);
                    return;
                }
                p_node = p_node->p_n_node;
            }
            n_book += register_symmetric_book(b, value, n_book);
            //save_book(b, value, -1);
        }

        inline void save(){
            remove("resources/book_backup.txt");
            rename("resources/book.txt", "resources/book_backup.txt");
            ofstream ofs("resources/book.txt");
            if (ofs.fail()){
                cerr << "book file not exist" << endl;
                return;
            }
            unordered_set<int> saved_idxes;
            for (int i = 0; i < book_hash_table_size; ++i){
                book_node *p_node = this->book[i];
                while(p_node != NULL){
					if (saved_idxes.find(p_node->line) == saved_idxes.end()) {
						saved_idxes.emplace(p_node->line);
						ofs << create_book_data(p_node->k, p_node->p, p_node->value) << endl;
					}
                    p_node = p_node->p_n_node;
                }
            }
            cerr << "saved" << endl;
        }

    private:
        inline bool compare_key(const uint_fast16_t a[], const uint_fast16_t b[]){
            return
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && 
                a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7];
        }

        inline book_node* book_node_init(board b, int value, int line){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            for (int i = 0; i < hw; ++i)
                p_node->k[i] = b.b[i];
            p_node->p = b.p;
            p_node->value = value;
            p_node->line = line;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline bool register_book(board b, int hash, int value, int line){
            if(this->book[hash] == NULL){
                this->book[hash] = book_node_init(b, value, line);
            } else {
                book_node *p_node = this->book[hash];
                book_node *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(p_node->p == b.p && compare_key(b.b, p_node->k)){
                        p_node->value = value;
                        return false;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = book_node_init(b, value, line);
            }
			return true;
        }

        inline int register_symmetric_book(board b, int value, int line){
            int i, res = 1;
            int tmp[b_idx_num];
			if (!register_book(b, b.hash() & book_hash_mask, value, line))
				res = 0;
            for (i = 0; i < 8; ++i)
                swap(b.b[i], b.b[8 + i]);
            register_book(b, b.hash() & book_hash_mask, value, line);
            for (i = 0; i < 16; ++i)
                tmp[i] = b.b[i];
            for (i = 0; i < 8; ++i)
                b.b[i] = reverse_board[tmp[7 - i]];
            for (i = 0; i < 8; ++i)
                b.b[8 + i] = reverse_board[tmp[15 - i]];
            register_book(b, b.hash() & book_hash_mask, value, line);
            for (i = 0; i < 8; ++i)
                swap(b.b[i], b.b[8 + i]);
            register_book(b, b.hash() & book_hash_mask, value, line);
			return res;
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
            res += to_string(b.p);
            res += " ";
            res += to_string(value);
            return res;
        }

        inline string create_book_data(uint_fast16_t key[], int p, int value){
            board b;
            for (int i = 0; i < hw; ++i)
                b.b[i] = key[i];
            b.p = p;
            return create_book_data(b, value);
        }

        inline void save_book(board b, int value, int line){
            remove("resources/book_backup.txt");
            rename("resources/book.txt", "resources/book_backup.txt");
            ifstream ifs("resources/book_backup.txt");
            if (ifs.fail()){
                cerr << "book file not exist" << endl;
                return;
            }
            ofstream ofs("resources/book.txt");
            if (ofs.fail()){
                cerr << "book file not exist" << endl;
                return;
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

bool book_init(){
    return book.init();
}
