#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <io.h>
#include "evaluate.hpp"
#include "board.hpp"

#define BOOK_HASH_TABLE_SIZE 67108864
#define BOOK_HASH_MASK 67108863


struct Node_book{
    unsigned long long player;
    unsigned long long opponent;
    int value;
    int line;
    Node_book* p_n_node;
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
        bool init(){
			for (int i = 0; i < BOOK_HASH_TABLE_SIZE; ++i)
				this->book[i] = NULL;
			n_book = 0;
            n_hash_conflict = 0;
            return import_file_bin("resources/book.egbk");
			//return import_file("resources/book.txt");
        }

        inline bool import_file_bin(string file){
			FILE* fp;
            if (fopen_s(&fp, file.c_str(), "rb") != 0){
                cerr << "can't open book.egbk" << endl;
                return false;
            }
            Board b;
            int n_boards, i, value;
            unsigned long long p, o;
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
				b.p = BLACK;
                b.b = p;
                b.w = o;
				n_book += register_symmetric_book(b, value, n_book);
			}
			cerr << "book imported " << n_book << " boards hash conflict " << n_hash_conflict << endl;
			fclose(fp);
			return true;
        }
        /*
        inline bool import_file_bin(string file){
			FILE* fp;
            if (fopen_s(&fp, file.c_str(), "rb") != 0){
                cerr << "can't open book.egbk" << endl;
                return false;
            }
            int p, value, i;
			unsigned char elem;
            int arr[HW2];
            Board b;
			for (;;) {
				if (n_book % 32768 == 0)
					cerr << "loading " << n_book << " boards" << endl;
				for (i = 0; i < HW2; i += 4) {
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
				value = elem % (HW2 * 2) - HW2;
				elem /= HW2 * 2;
				p = elem % 2;
				if (elem / 2) {
					cerr << "book NOT FULLY imported " << n_book << " boards 3" << endl;
					fclose(fp);
					return false;
				}
				b.translate_from_arr(arr, p);
				n_book += register_symmetric_book(b, (p ? -1 : 1) * value, n_book);
			}
			cerr << "book imported " << n_book << " boards" << endl;
			fclose(fp);
			return true;
        }
        */

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
            Board b;
            Mobility mob;
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
                b.p = BLACK;
                b.b = player;
                b.w = opponent;
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
                        calc_flip(&mob, &b, (int)link_move);
                        if (mob.flip == 0ULL){
                            cerr << "error! illegal move" << endl;
                            return false;
                        }
                        b.move(&mob);
                            n_book += register_symmetric_book(b, (int)link_value, n_book);
                        b.undo(&mob);
					}
				}
            }
			cerr << "book imported " << n_book << " boards hash conflict " << n_hash_conflict << endl;
            return true;
        }

        /*
        inline bool import_file(string file){
            int j;
            int board_arr[HW2];
			FILE* fp;
            if (fopen_s(&fp, file.c_str(), "r") != 0) {
                cerr << "can't open book.txt" << endl;
				return false;
            }
            Board b;
            double value;
            int p;
			bool flag = true;
			char elem;
            while (flag){
				for (j = 0; j < HW2; ++j) {
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
					if (value < 0 || value > HW2 * 2 / 16) {
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
					if (value < 0 || value > HW2 * 2) {
						cerr << "book import error 4 value=" << value << " char " << elem << " found in line " << n_book + 1 << endl;
						return false;
					}
					value -= HW2;
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
        */

        inline void reg(Board b, int value){
            n_book += register_symmetric_book(b, value, n_book);
        }

        inline int get(Board *b){
            Node_book *p_node = this->book[b->hash_player() & BOOK_HASH_MASK];
            while(p_node != NULL){
                if(compare_key(b, p_node)){
                    return p_node->value;
                }
                p_node = p_node->p_n_node;
            }
            return -INF;
        }

        inline Book_value get_random(Board *b, int accept_value){
            vector<int> policies;
            vector<int> values;
            Board nb;
            int max_value = -INF;
            unsigned long long legal = b->mobility_ull();
            Mobility mob;
            for (int coord = 0; coord < HW2; ++coord){
                if (1 & (legal >> coord)){
                    calc_flip(&mob, b, coord);
                    nb = b->move_copy(&mob);
                    Node_book *p_node = this->book[nb.hash_player() & BOOK_HASH_MASK];
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

        inline void change(Board b, int value){
            n_book += register_symmetric_book(b, value, n_book);
			cerr << "book changed" << endl;
        }
        /*
        inline void save(){
			if (_access_s("resources/book_backup.txt", 0) == 0)
				remove("resources/book_backup.txt");
            rename("resources/book.txt", "resources/book_backup.txt");
            ofstream ofs("resources/book.txt");
            if (ofs.fail()){
                cerr << "can't open book.egbk" << endl;
                return;
            }
            unordered_set<int> saved_idxes;
            for (int i = 0; i < BOOK_HASH_TABLE_SIZE; ++i){
                Node_book *p_node = this->book[i];
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
        
        inline void save_bin(){
			if (_access_s("resources/book_backup.egbk", 0) == 0)
				remove("resources/book_backup.egbk");
			rename("resources/book.egbk", "resources/book_backup.egbk");
            ofstream fout;
            fout.open("resources/book.egbk", ios::out|ios::binary|ios::trunc);
            if (!fout){
                cerr << "can't open book.egbk" << endl;
                return;
            }
            unordered_set<int> saved_idxes;
            unsigned long long i;
			unsigned char elem;
            fout.write((char*)&n_book, 4);
			int t = 0;
            for (i = 0; i < BOOK_HASH_TABLE_SIZE; ++i){
                if (i % 1048576 == 0)
                    cerr << "saving book " << (i * 100 / BOOK_HASH_TABLE_SIZE) << "%" << endl;
                Node_book *p_node = this->book[i];
                while(p_node != NULL){
					if (saved_idxes.find(p_node->line) == saved_idxes.end()) {
						saved_idxes.emplace(p_node->line);
                        fout.write((char*)&p_node->player, 8);
                        fout.write((char*)&p_node->opponent, 8);
						elem = max(0, min(HW2 * 2, p_node->value + HW2));
                        fout.write((char*)&elem, 1);
						++t;
					}
                    p_node = p_node->p_n_node;
                }
            }
            fout.close();
            cerr << "saved " << t << " boards" << endl;
        }
        

    private:
        inline bool compare_key(const Board *a, const Node_book *b){
            if (a->p == BLACK)
                return a->b == b->player && a->w == b->opponent;
            return a->w == b->player && a->b == b->opponent;
        }

        inline Node_book* Node_book_init(Board b, int value, int line){
            Node_book* p_node = NULL;
            p_node = (Node_book*)malloc(sizeof(Node_book));
            if (b.p == BLACK){
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

        inline int register_symmetric_book(Board b, int value, int line){
            int res = 1;
			if (!register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line))
				res = 0;
            b.white_mirror();
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.black_mirror();
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.white_mirror();
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.b = horizontal_mirror(b.b);
            b.w = horizontal_mirror(b.w);
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.white_mirror();
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.black_mirror();
            register_book(b, b.hash_player() & BOOK_HASH_MASK, value, line);
            b.white_mirror();
			return res;
        }
        /*
        inline string create_book_data(Board b, int value){
            string res = "";
            int arr[HW2];
            b.translate_to_arr(arr);
            for (int i = 0; i < HW2; ++i)
                res += (char)(arr[i] + (int)'0');
            value = max(0, min(HW2 * 2, value + HW2));
            res += (char)(b.p + (int)'0');
            res += (char)((int)'!' + value / 16);
			res += (char)((int)'!' + value % 16);
            return res;
        }

        inline string create_book_data(Node_book *node, int p, int value){
            Board b;
            for (int i = 0; i < hw; ++i)
                b.b[i] = key[i];
            b.p = p;
            return create_book_data(b, value);
        }
        */

        inline void create_arr(Node_book *node, int arr[], int p){
            Board b;
            b.p = p;
            if (p == BLACK){
                b.b = node->player;
                b.w = node->opponent;
            } else{
                b.w = node->player;
                b.b = node->opponent;
            }
			b.translate_to_arr(arr);
        }

		/*
        inline void save_book(Board b, int value, int line){
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

Book book;

bool book_init(){
    return book.init();
}

int modify_book(Board b){
    unsigned long long legal = b.mobility_ull();
    Mobility mob;
    bool has_child = false;
    int v = -INF;
    int vbook = book.get(&b);
    for (int cell = 0; cell < HW2; ++cell){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            b.move(&mob);
                if (book.get(&b) != -INF){
                    has_child = true;
                    v = max(v, modify_book(b));
                }
            b.undo(&mob);
        }
    }
    if (!has_child)
        return vbook;
    if (-v != vbook)
        book.change(b, -v);
    return -v;
}