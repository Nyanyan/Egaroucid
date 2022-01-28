#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <io.h>
#include "evaluate.hpp"
#include "board.hpp"

#define book_hash_table_size 67108864
constexpr int book_hash_mask = book_hash_table_size - 1;


struct book_node{
    unsigned long long player;
    unsigned long long opponent;
    int value;
    int line;
    book_node* p_n_node;
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
            return import_file_bin("resources/book.ebok");
			//return import_file("resources/book.txt");
        }

        inline bool import_file_bin(string file){
			FILE* fp;
            if (fopen_s(&fp, file.c_str(), "rb") != 0){
                cerr << "can't open book.ebok" << endl;
                return false;
            }
            int p, value, i;
			unsigned char elem;
            int arr[hw2];
            board b;
			for (;;) {
				if (n_book % 1024 == 0)
					cerr << "loading " << n_book << " boards" << endl;
				for (i = 0; i < hw2; i += 4) {
					if (fread(&elem, 1, 1, fp) < 1) {
						if (i == 0) {
							cerr << "book imported " << n_book << " boards" << endl;
							fclose(fp);
							return true;
						}
						cerr << "book NOT FULLY imported " << n_book << " boards 0 " << i << endl;
						fclose(fp);
						return false;
					}
					arr[i + 3] = elem % p31;
					elem /= p31;
					arr[i + 2] = elem % p31;
					elem /= p31;
					arr[i + 1] = elem % p31;
					elem /= p31;
					arr[i] = elem % p31;
					if (elem / p31) {
						cerr << "book NOT FULLY imported " << n_book << " boards 1" << endl;
						fclose(fp);
						return false;
					}
				}
				if (fread(&elem, 1, 1, fp) < 1) {
					cerr << "book NOT FULLY imported " << n_book << " boards 2" << endl;
					fclose(fp);
					return false;
				}
				value = elem % (hw2 * 2) - hw2;
				elem /= hw2 * 2;
				p = elem % 2;
				if (elem / 2) {
					cerr << "book NOT FULLY imported " << n_book << " boards 3" << endl;
					fclose(fp);
					return false;
				}
				b.translate_from_arr(arr, p);
				n_book += register_symmetric_book(b, value, n_book);
			}
			cerr << "book imported " << n_book << " boards" << endl;
			fclose(fp);
			return true;
        }

		inline bool import_edax_book(string file) {
			FILE* fp;
			if (fopen_s(&fp, file.c_str(), "rb") != 0) {
				cerr << "can't open " << file << endl;
				return false;
			}
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
            unsigned long long player, opponent;
            short value;
			char link = 0, link_value, link_move;
            board b;
            mobility mob;
            for (i = 0; i < n_boards; ++i){
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
				n_book += register_symmetric_book(b, (int)value, n_book);
				if (n_book % 1024 == 0)
					cerr << "loading " << n_book << " boards" << endl;
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
					if (link_move != hw2_p1) {
                        mob.calc_flip(player, opponent, (int)link_move);
                        b.move(&mob);
                        n_book += register_symmetric_book(b, -(int)link_value, n_book);
                        b.undo(&mob);
                        if (n_book % 1024 == 0)
                            cerr << "loading " << n_book << " boards" << endl;
					}
				}
            }
			cerr << "book imported " << n_book << " boards" << endl;
            return true;
        }

        inline bool import_file(string file){
            int j;
            int board_arr[hw2];
			FILE* fp;
            if (fopen_s(&fp, file.c_str(), "r") != 0) {
                cerr << "can't open book.txt" << endl;
				return false;
            }
            board b;
            double value;
            int p;
			bool flag = true;
			char elem;
            while (flag){
				for (j = 0; j < hw2; ++j) {
					if ((elem = fgetc(fp)) == EOF){
						flag = false;
						break;
					}
					board_arr[j] = (int)elem - (int)'0';
                    if (board_arr[j] < 0 || board_arr[j] > 2){
                        cerr << "book import error 1 char " << elem << " found in line " << n_book + 1 << endl;
                        return false;
                    }
                }
				if (flag) {
					if ((elem = fgetc(fp)) == EOF) {
						flag = false;
						break;
					}
					p = (int)elem - (int)'0';
					if (p != 0 && p != 1) {
						cerr << "book import error 2" << endl;
						return false;
					}
					b.translate_from_arr(board_arr, p);
				}
				if (flag) {
					if ((elem = fgetc(fp)) == EOF) {
						flag = false;
						break;
					}
					value = (int)elem - (int)'!';
					if (value < 0 || value > hw2 * 2 / 16) {
						cerr << "book import error 3 value=" << value << " char " << elem << " found in line " << n_book + 1 << endl;
						return false;
					}
				}
				if (flag) {
					if ((elem = fgetc(fp)) == EOF) {
						flag = false;
						break;
					}
					value *= 16;
					value += (int)elem - (int)'!';
					if (value < 0 || value > hw2 * 2) {
						cerr << "book import error 4 value=" << value << " char " << elem << " found in line " << n_book + 1 << endl;
						return false;
					}
					value -= hw2;
					n_book += register_symmetric_book(b, value, n_book);
				}
				while (elem != '\n') {
					if ((elem = fgetc(fp)) == EOF) {
						flag = false;
						break;
					}
				}
            }
			fclose(fp);
            cerr << "book initialized " << n_book << " boards in book" << endl;
            return true;
        }

        inline void reg(board b, int value){
            n_book += register_symmetric_book(b, value, n_book);
        }

        inline int get(board *b){
            book_node *p_node = this->book[b->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(b, p_node)){
                    return p_node->value;
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
            unsigned long long legal = b->mobility_ull();
            mobility mob;
            for (int coord = 0; coord < hw2; ++coord){
                if (1 & (legal >> coord)){
                    calc_flip(&mob, b, coord);
                    nb = b->move_copy(&mob);
                    book_node *p_node = this->book[nb.hash() & book_hash_mask];
                    while(p_node != NULL){
                        if(compare_key(&nb, p_node)){
                            policies.push_back(coord);
                            values.push_back(p_node->value);
                            max_value = max(max_value, p_node->value);
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
            book_node *p_node = this->book[b.hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(&b, p_node)){
                    int result = register_symmetric_book(b, value, p_node->line);
					cerr << "value changed " << result << endl;
                    return;
                }
                p_node = p_node->p_n_node;
            }
            n_book += register_symmetric_book(b, value, n_book);
			cerr << "new value registered" << endl;
        }
        /*
        inline void save(){
			if (_access_s("resources/book_backup.txt", 0) == 0)
				remove("resources/book_backup.txt");
            rename("resources/book.txt", "resources/book_backup.txt");
            ofstream ofs("resources/book.txt");
            if (ofs.fail()){
                cerr << "can't open book.ebok" << endl;
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
        */
        /*
        inline void save_bin(){
			if (_access_s("resources/book_backup.ebok", 0) == 0)
				remove("resources/book_backup.ebok");
			rename("resources/book.ebok", "resources/book_backup.ebok");
            ofstream fout;
            fout.open("resources/book.ebok", ios::out|ios::binary|ios::trunc);
            if (!fout){
                cerr << "can't open book.ebok" << endl;
                return;
            }
            unordered_set<int> saved_idxes;
            int i, j;
            int arr[hw2];
			unsigned char elem;
			int t = 0;
            for (i = 0; i < book_hash_table_size; ++i){
                book_node *p_node = this->book[i];
                while(p_node != NULL){
					if (saved_idxes.find(p_node->line) == saved_idxes.end()) {
						saved_idxes.emplace(p_node->line);
                        create_arr(p_node, arr, black);
						for (j = 0; j < hw2; j += 4) {
							elem = arr[j] * p33 + arr[j + 1] * p32 + arr[j + 2] * p31 + arr[j + 3];
							fout.write((char*)&elem, 1);
						}
						elem = p_node->p * hw2 * 2 + max(0, min(hw2 * 2, p_node->value + hw2));
                        fout.write((char*)&elem, 1);
						++t;
					}
                    p_node = p_node->p_n_node;
                }
            }
            cerr << "saved " << t << " boards" << endl;
        }
        */

    private:
        inline bool compare_key(const board *a, const book_node *b){
            if (a->p == black)
                return a->b == b->player && a->w == b->opponent;
            return a->w == b->player && a->b == b->opponent;
        }

        inline book_node* book_node_init(board b, int value, int line){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            if (b.p == black){
                p_node->player = b.b;
                p_node->opponent = b.w;
            } else{
                p_node->player = b.w;
                p_node->opponent = b.b;
            }
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
                    if(compare_key(&b, p_node)){
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
            int res = 1;
			if (!register_book(b, b.hash() & book_hash_mask, value, line))
				res = 0;
            b.white_mirror();
            register_book(b, b.hash() & book_hash_mask, value, line);
            b.black_mirror();
            register_book(b, b.hash() & book_hash_mask, value, line);
            b.white_mirror();
            register_book(b, b.hash() & book_hash_mask, value, line);
			return res;
        }
        /*
        inline string create_book_data(board b, int value){
            string res = "";
            int arr[hw2];
            b.translate_to_arr(arr);
            for (int i = 0; i < hw2; ++i)
                res += (char)(arr[i] + (int)'0');
            value = max(0, min(hw2 * 2, value + hw2));
            res += (char)(b.p + (int)'0');
            res += (char)((int)'!' + value / 16);
			res += (char)((int)'!' + value % 16);
            return res;
        }

        inline string create_book_data(book_node *node, int p, int value){
            board b;
            for (int i = 0; i < hw; ++i)
                b.b[i] = key[i];
            b.p = p;
            return create_book_data(b, value);
        }
        */

        inline void create_arr(book_node *node, int arr[], int p){
            board b;
            b.p = p;
            if (p == black){
                b.b = node->player;
                b.w = node->opponent;
            } else{
                b.w = node->player;
                b.b = node->opponent;
            }
			b.translate_to_arr(arr);
        }

		/*
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
		*/
};

book book;

bool book_init(){
    return book.init();
}
