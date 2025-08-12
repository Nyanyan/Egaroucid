﻿/*
    Egaroucid Project

    @file book.hpp
        Book class
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "search.hpp"
#include "ai.hpp"

inline Search_result tree_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, bool use_multi_thread);
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);
Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal);
void search_new_leaf(Board board, int level, int book_elem_value, bool use_multi_thread);

#define FORCE_BOOK_DEPTH false

constexpr int BOOK_N_ACCEPT_LEVEL = 11;
constexpr int BOOK_ACCURACY_LEVEL_INF = 10;
constexpr int BOOK_LEAF_LEVEL = 5;

constexpr int LEVEL_UNDEFINED = -1;
constexpr int LEVEL_HUMAN = 70;
constexpr int BOOK_LOSS_IGNORE_THRESHOLD = 8;
constexpr int LEAF_CALCULATE_LEVEL = 5;

#define BOOK_EXTENSION ".egbk3"
#define BOOK_EXTENSION_NODOT "egbk3"

constexpr int ADD_LEAF_SPECIAL_LEVEL = -1;

constexpr uint64_t MAX_N_LINES = 4000000000; // < 2^32

/*
    @brief book result structure

    @param policy               selected best move
    @param value                registered score
*/
struct Book_value {
    int policy;
    int value;

    Search_result to_search_result() {
        Search_result res;
        res.policy = policy;
        res.value = value;
        res.depth = SEARCH_BOOK;
        res.time = 0;
        res.nodes = 0;
        res.clog_time = 0;
        res.clog_nodes = 0;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = -1;
        return res;
    }
};

struct Leaf {
    int8_t value;
    int8_t move;
    int8_t level;

    Leaf()
        : value(SCORE_UNDEFINED), move(MOVE_UNDEFINED), level(LEVEL_UNDEFINED) {}
};

/*
    @brief book element structure

    @param value                registered score
    @param level                AI level
    @param moves                each moves and values
*/
struct Book_elem {
    int8_t value;
    int8_t level;
    Leaf leaf;
    uint32_t n_lines;
    bool seen; // used in various situation

    Book_elem()
        : value(SCORE_UNDEFINED), level(LEVEL_UNDEFINED), n_lines(0), seen(false) {}
};

struct Book_info {
    uint64_t n_boards;
    uint64_t n_boards_in_level[LEVEL_HUMAN + 1];
    uint64_t n_boards_in_ply[HW2 - 4 + 1];
    uint64_t n_leaves_in_level[LEVEL_HUMAN + 1];
    uint64_t n_leaves_in_ply[HW2 - 4 + 1];

    Book_info() 
        : n_boards(0) {
        for (int i = 0; i < LEVEL_HUMAN + 1; ++i) {
            n_boards_in_level[i] = 0;
            n_leaves_in_level[i] = 0;
        }
        for (int i = 0; i < HW2 - 4 + 1; ++i) {
            n_boards_in_ply[i] = 0;
            n_leaves_in_ply[i] = 0;
        }
    }
};

/*
    @brief array for calculating hash code for book
*/
size_t hash_rand_player_book[4][65536];
size_t hash_rand_opponent_book[4][65536];

/*
    @brief initialize hash array for book randomly
*/
void book_hash_init_rand() {
    int i, j;
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 65536; ++j) {
            hash_rand_player_book[i][j] = 0;
            while (pop_count_uint(hash_rand_player_book[i][j]) < 9) {
                hash_rand_player_book[i][j] = myrand_ull();
            }
            hash_rand_opponent_book[i][j] = 0;
            while (pop_count_uint(hash_rand_opponent_book[i][j]) < 9) {
                hash_rand_opponent_book[i][j] = myrand_ull();
            }
        }
    }
}

/*
    @brief initialize hash array for book
*/
void book_hash_init(bool show_log) {
    FILE* fp;
    if (!file_open(&fp, "resources/hash_book.eghs", "rb")) {
        std::cerr << "[ERROR] can't open hash_book.eghs" << std::endl;
        book_hash_init_rand();
        return;
    }
    for (int i = 0; i < 4; ++i) {
        if (fread(hash_rand_player_book[i], 8, 65536, fp) < 65536) {
            std::cerr << "[ERROR] hash_book.eghs broken" << std::endl;
            book_hash_init_rand();
            return;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (fread(hash_rand_opponent_book[i], 8, 65536, fp) < 65536) {
            std::cerr << "[ERROR] hash_book.eghs broken" << std::endl;
            book_hash_init_rand();
            return;
        }
    }
    if (show_log) {
        std::cerr << "hash for book initialized" << std::endl;
    }
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
*/
class Book {
    private:
        std::mutex mtx;
        std::unordered_map<Board, Book_elem, Book_hash> book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading) {
            delete_all();
            if (!import_file_egbk3(file, show_log, stop_loading)) { // try egbk3 format
                std::cerr << "failed egbk3 formatted book. trying egbk2 format." << std::endl;
                if (!import_file_egbk2(file, show_log, stop_loading)) { // try egbk2 format
                    std::cerr << "failed egbk2 formatted book. trying egbk format." << std::endl;
                    return import_file_egbk(file, 1, show_log, stop_loading); // try egbk format
                }
            }
            return true;
        }

        inline bool import_book_extension_determination(std::string file, int level, bool *stop) {
            bool result = false;
            std::vector<std::string> lst;
            auto offset = std::string::size_type(0);
            while (1) {
                auto pos = file.find(".", offset);
                if (pos == std::string::npos) {
                    lst.push_back(file.substr(offset));
                    break;
                }
                lst.push_back(file.substr(offset, pos - offset));
                offset = pos + 1;
            }
            if (lst[lst.size() - 1] == "egbk3") {
                std::cerr << "importing Egaroucid book (.egbk3)" << std::endl;
                result = import_file_egbk3(file, true, stop);
            }
            if (lst[lst.size() - 1] == "egbk2") {
                std::cerr << "importing Egaroucid legacy book (.egbk2)" << std::endl;
                result = import_file_egbk2(file, true, stop);
            } else if (lst[lst.size() - 1] == "egbk") {
                std::cerr << "importing Egaroucid legacy book (.egbk)" << std::endl;
                result = import_file_egbk(file, level, true, stop);
            } else if (lst[lst.size() - 1] == "dat") {
                std::cerr << "importing Edax book" << std::endl;
                result = import_file_edax(file, true, stop);
            } else {
                std::cerr << "this is not a book" << std::endl;
            }
            return result;
        }

        inline bool import_book_extension_determination(std::string file, int level) {
            bool stop = false;
            return import_book_extension_determination(file, level, &stop);
        }

        inline bool import_book_extension_determination(std::string file) {
            bool stop = false;
            return import_book_extension_determination(file, LEVEL_UNDEFINED, &stop);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk3 file)
            @return book completely imported?
        */
        inline bool import_file_egbk3(std::string file, bool show_log, bool *stop_loading) {
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value, level, leaf_value, leaf_move, leaf_level;
            uint32_t n_lines;
            uint64_t p, o;
            char egaroucid_str[10];
            char egaroucid_str_ans[] = "DICUORAGE";
            char elem_char;
            char book_version;
            // Header
            if (fread(egaroucid_str, 1, 9, fp) < 9) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            for (int i = 0; i < 9; ++i) {
                if (egaroucid_str[i] != egaroucid_str_ans[i]) {
                    std::cerr << "[ERROR] This is not Egarocuid book, found " << egaroucid_str[i] << ", " << (int)egaroucid_str[i] << " at char " << i << ", expected " << egaroucid_str_ans[i] << " , " << (int)egaroucid_str_ans[i] << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&book_version, 1, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            if (book_version != 3) {
                std::cerr << "[ERROR] This is not Egarocuid book version 3, found version " << (int)book_version << std::endl;
                fclose(fp);
                return false;
            }
            // Book Information
            if (fread(&n_boards, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] book broken at n_book data" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log) {
                std::cerr << n_boards << " boards to read" << std::endl;
            }
            // for each board
            int percent = -1;
            char datum[25];
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading) {
                    break;
                }
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log) {
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board, player
                if (fread(datum, 1, 25, fp) < 25) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                p = ((uint64_t*)datum)[0];
                o = ((uint64_t*)datum)[1];
                value = datum[16];
                level = datum[17];
                n_lines = ((uint32_t*)(datum + 18))[0];
                leaf_value = datum[22];
                leaf_move = datum[23];
                leaf_level = datum[24];
                /*
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 0 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board, opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 1 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 3 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read level
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 3 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                //if (value < -HW2 || HW2 < value) {
                //    std::cerr << "[ERROR] book NOT FULLY imported 4 got value " << (int)value << " " << book.size() << " boards" << std::endl;
                //    fclose(fp);
                //    return false;
                //    //std::cerr << "[WARNING] value error found " << (int)value << " " << book.size() << " boards" << std::endl;
                //    //value = SCORE_UNDEFINED;
                //}
                // read n_lines
                if (fread(&n_lines, 4, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 5 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf value
                if (fread(&leaf_value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 6 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf move
                if (fread(&leaf_move, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 7 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf level
                if (fread(&leaf_level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported 3 " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                */
                // push elem
                if (-HW2 <= value && value <= HW2 && (p & o) == 0) {
                    board.player = p;
                    board.opponent = o;
#if FORCE_BOOK_DEPTH
                    if (board.n_discs() <= 4 + 30) {
#endif
                        book_elem.value = value;
                        book_elem.level = level;
                        //if (board.n_discs() >= 32 + 4)
                        //    book_elem.level = 27;
                        //else
                        //    book_elem.level = 17;
                        book_elem.n_lines = n_lines;
                        book_elem.leaf.value = leaf_value;
                        book_elem.leaf.move = leaf_move;
                        book_elem.leaf.level = leaf_level;
                        merge(board, book_elem);
#if FORCE_BOOK_DEPTH
                    }
#endif
                }
            }
            if (*stop_loading) {
                std::cerr << "stop loading book" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log) {
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            }
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk3(std::string file, bool show_log) {
            bool stop_loading = false;
            return import_file_egbk3(file, show_log, &stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book (old)

            @param file                 book file (.egbk2 file)
            @return book completely imported?
        */
        inline bool import_file_egbk2(std::string file, bool show_log, bool *stop_loading) {
            if (show_log) {
                std::cerr << "importing " << file << std::endl;
            }
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value;
            uint64_t p, o;
            char level, n_moves, val, mov;
            char egaroucid_str[10];
            char egaroucid_str_ans[] = "DICUORAGE";
            char book_version;
            // Header
            if (fread(egaroucid_str, 1, 9, fp) < 9) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            for (int i = 0; i < 9; ++i) {
                if (egaroucid_str[i] != egaroucid_str_ans[i]) {
                    std::cerr << "[ERROR] This is not Egarocuid book version 2, found " << egaroucid_str[i] << ", " << (int)egaroucid_str[i] << " at char " << i << ", expected " << egaroucid_str_ans[i] << " , " << (int)egaroucid_str_ans[i] << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(&book_version, 1, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            if (book_version != 2) {
                std::cerr << "[ERROR] This is not Egarocuid book version 2, found version " << (int)book_version << std::endl;
                fclose(fp);
                return false;
            }
            // Book Information
            if (fread(&n_boards, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log) {
                std::cerr << n_boards << " boards to read" << std::endl;
            }
            // for each board
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading) {
                    break;
                }
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log) {
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read level
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read n_links
                if (fread(&n_moves, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // for each link
                for (uint8_t i = 0; i < n_moves; ++i) {
                    // read value
                    if (fread(&val, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    // read move
                    if (fread(&mov, 1, 1, fp) < 1) {
                        std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                        fclose(fp);
                        return false;
                    }
                    // ignore links...
                }
                board.player = p;
                board.opponent = o;
#if FORCE_BOOK_DEPTH
                if (b.n_discs() <= 4 + 30) {
#endif
                    book_elem.value = value;
                    book_elem.level = level;
                    book_elem.leaf.value = SCORE_UNDEFINED;
                    book_elem.leaf.move = MOVE_UNDEFINED;
                    merge(board, book_elem);
#if FORCE_BOOK_DEPTH
                }
#endif
            }
            // check_add_leaf_all_undefined();
            // bool stop = false;
            // check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            // if (*stop_loading) {
            //     std::cerr << "stop loading book" << std::endl;
            //     fclose(fp);
            //     return false;
            // }
            if (show_log)
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk2(std::string file, bool show_log) {
            bool stop_loading = false;
            return import_file_egbk2(file, show_log, &stop_loading);
        }



        /*
            @brief import Egaroucid-formatted book (old)

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        inline bool import_file_egbk(std::string file, int level, bool show_log, bool *stop_loading) {
            if (show_log)
                std::cerr << "importing " << file << std::endl;
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            Board board;
            Book_elem book_elem;
            int n_boards;
            char value;
            uint64_t p, o;
            uint8_t value_raw;
            // Book Information
            if (fread(&n_boards, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                fclose(fp);
                return false;
            }
            if (show_log) {
                std::cerr << n_boards << " boards to read" << std::endl;
            }
            // for each board
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop_loading) {
                    break;
                }
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log) {
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&p, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&o, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value_raw, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                value = -((int8_t)value_raw - HW2);
                if (value < -HW2 || HW2 < value) {
                    std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    fclose(fp);
                    return false;
                }
                board.player = p;
                board.opponent = o;
#if FORCE_BOOK_DEPTH
                if (b.n_discs() <= 4 + 30) {
#endif
                    book_elem.value = value;
                    if (level != LEVEL_UNDEFINED) {
                        book_elem.level = level;
                    } else {
                        book_elem.level = 1;
                    }
                    book_elem.leaf.value = SCORE_UNDEFINED;
                    book_elem.leaf.move = MOVE_UNDEFINED;
                    merge(board, book_elem);
#if FORCE_BOOK_DEPTH
                }
#endif
            }
            // check_add_leaf_all_undefined();
            // bool stop = false;
            // check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            // if (*stop_loading) {
            //     std::cerr << "stop loading book" << std::endl;
            //     fclose(fp);
            //     return false;
            // }
            if (show_log) {
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            }
            fclose(fp);
            return true;
        }

        inline bool import_file_egbk(std::string file, int level, bool show_log) {
            bool stop_loading = false;
            return import_file_egbk(file, level, show_log, &stop_loading);
        }

        /*
            @brief import Edax-formatted book

            @param file                 book file (.dat file)
            @return book completely imported?
        */
        inline bool import_file_edax(std::string file, bool show_log, bool *stop) {
            if (show_log) {
                std::cerr << "importing " << file << std::endl;
            }
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                std::cerr << "[ERROR] can't open Edax book " << file << std::endl;
                return false;
            }
            char elem_char;
            int elem_int;
            int16_t elem_short;
            // Header
            for (int i = 0; i < 38; ++i) {
                if (fread(&elem_char, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            // Book Information
            if (fread(&elem_int, 4, 1, fp) < 1) {
                std::cerr << "[ERROR] file broken" << std::endl;
                fclose(fp);
                return false;
            }
            int n_boards = elem_int;
            std::cerr << n_boards << " boards found" << std::endl;
            uint64_t player, opponent;
            int16_t value;
            uint32_t n_lines;
            char link = 0, link_value, link_move, level, leaf_value, leaf_move;
            Board board;
            Flip flip;
            Book_elem book_elem;
            int percent = -1;
            for (int i = 0; i < n_boards; ++i) {
                if (*stop) {
                    return false;
                }
                int n_percent = (double)i / n_boards * 100;
                if (n_percent > percent && show_log) {
                    percent = n_percent;
                    std::cerr << "loading book " << percent << "%" << std::endl;
                }
                // read board player
                if (fread(&player, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read board opponent
                if (fread(&opponent, 8, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read additional data (w/d/l)
                for (int j = 0; j < 3; ++j) {
                    if (fread(&elem_int, 4, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                // read n_lines
                if (fread(&n_lines, 4, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read value
                if (fread(&value, 2, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (value < -HW2 || HW2 < value) {
                    //std::cerr << "[ERROR] book NOT FULLY imported got value " << (int)value << " " << book.size() << " boards" << std::endl;
                    //fclose(fp);
                    //return false;
                    //std::cerr << "[WARNING] value error found " << (int)value << " " << book.size() << " boards" << std::endl;
                    value = SCORE_UNDEFINED;
                }
                // read additional data
                for (int j = 0; j < 2; ++j) {
                    if (fread(&elem_short, 2, 1, fp) < 1) {
                        std::cerr << "[ERROR] file broken" << std::endl;
                        fclose(fp);
                        return false;
                    }
                }
                // read link data
                if (fread(&link, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read level
                if (fread(&level, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // for each link (ignore)
                for (int j = 0; j < (int)link; ++j) {
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
                }
                // read leaf value
                if (fread(&leaf_value, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                // read leaf move
                if (fread(&leaf_move, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (leaf_value < -HW2 || HW2 < leaf_value || leaf_move < 0 || HW2 <= leaf_move) {
                    leaf_value = SCORE_UNDEFINED;
                    leaf_move = MOVE_UNDEFINED;
                }
                if (value != SCORE_UNDEFINED && (player & opponent) == 0ULL && calc_legal(player, opponent)) {
                    board.player = player;
                    board.opponent = opponent;
                    book_elem.value = value;
                    book_elem.level = level;
                    book_elem.leaf.value = leaf_value;
                    book_elem.leaf.move = leaf_move;
                    book_elem.leaf.level = level;
                    book_elem.n_lines = n_lines;
                    merge(board, book_elem);
                }
            }
            if (show_log) {
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            }
            return true;
        }

        /*
            @brief save as Egaroucid-formatted book (.egbk3)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_egbk3(std::string file, std::string bak_file, bool use_backup, int level) {
            if (use_backup) {
                if (remove(bak_file.c_str()) == -1) {
                    std::cerr << "cannot delete backup. you can ignore this error." << std::endl;
                }
                rename(file.c_str(), bak_file.c_str());
            }
            std::ofstream fout;
            fout.open(file.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
            if (!fout) {
                std::cerr << "can't open " << file << std::endl;
                return;
            }
            char elem;
            std::cerr << "saving book..." << std::endl;
            char egaroucid_str[] = "DICUORAGE";
            fout.write((char*)&egaroucid_str, 9);
            char book_version = 3;
            fout.write((char*)&book_version, 1);
            int n_book = (int)book.size();
            fout.write((char*)&n_book, 4);
            int t = 0, percent = -1, n_boards = (int)book.size();
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                ++t;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "saving book " << percent << "%" << std::endl;
                }
                char char_level = itr->second.level, char_leaf_level = itr->second.leaf.level;
                if (level != LEVEL_UNDEFINED) {
                    char_level = level;
                    char_leaf_level = level;
                }
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                fout.write((char*)&itr->second.value, 1);
                fout.write((char*)&char_level, 1);
                fout.write((char*)&itr->second.n_lines, 4);
                fout.write((char*)&itr->second.leaf.value, 1);
                fout.write((char*)&itr->second.leaf.move, 1);
                fout.write((char*)&char_leaf_level, 1);
            }
            fout.close();
            int book_size = (int)book.size();
            std::cerr << "saved " << t << " boards , book_size " << book_size << std::endl;
        }

        inline void save_egbk3(std::string file, std::string bak_file) {
            save_egbk3(file, bak_file, true, LEVEL_UNDEFINED);
        }

        inline void save_egbk3(std::string file, int level) {
            save_egbk3(file, "", false, level);
        }

        void get_pass_boards(Board board, std::unordered_set<Board, Book_hash> &pass_boards) {
            board = representative_board(board);
            if (contain_representative(board)) {
                if (book[board].seen) {
                    return;
                }
                book[board].seen = true;
            }
            uint64_t legal = board.get_legal();
            if (legal == 0ULL) {
                if (pass_boards.find(board) != pass_boards.end()) {
                    return;
                }
                Board passed_board = board.copy();
                passed_board.pass();
                if (!contain_representative(board) && contain(passed_board)) {
                    pass_boards.emplace(board);
                    get_pass_boards(passed_board, pass_boards);
                }
            } else {
                std::vector<Book_value> links = get_all_moves_with_value(&board);
                Flip flip;
                for (Book_value &link: links) {
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        get_pass_boards(board, pass_boards);
                    board.undo_board(&flip);
                }
            }
        }

        /*
            @brief save as Edax-formatted book (.dat)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin_edax(std::string file, int level) {
            bool stop = false;
            check_add_leaf_all_search(ADD_LEAF_SPECIAL_LEVEL, &stop);
            std::unordered_set<Board, Book_hash> pass_boards;
            Board root_board;
            root_board.reset();
            std::cerr << "pass board calculating..." << std::endl;
            reset_seen();
            get_pass_boards(root_board, pass_boards);
            reset_seen();
            std::cerr << "pass board calculated " << pass_boards.size() << std::endl;
            std::ofstream fout;
            fout.open(file.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
            if (!fout) {
                std::cerr << "can't open " << file << std::endl;
                return;
            }
            std::cerr << "saving book..." << std::endl;
            char header[] = "XADEKOOB";
            for (int i = 0; i < 8; ++i) {
                fout.write((char*)&header[i], 1);
            }
            char ver = 4;
            fout.write((char*)&ver, 1);
            char rel = 4;
            fout.write((char*)&rel, 1);
            int year, month, day, hour, minute, second;
            calc_date(&year, &month, &day, &hour, &minute, &second);
            fout.write((char*)&year, 2);
            fout.write((char*)&month, 1);
            fout.write((char*)&day, 1);
            fout.write((char*)&hour, 1);
            fout.write((char*)&minute, 1);
            fout.write((char*)&second, 1);
            char dummy = 0;
            fout.write((char*)&dummy, 1);
            fout.write((char*)&level, 4);
            int n_empties = HW2;
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                n_empties = std::min(n_empties, HW2 + 1 - itr->first.n_discs());
            }
            fout.write((char*)&n_empties, 4);
            int err_mid = 0;
            fout.write((char*)&err_mid, 4);
            int err_end = 0;
            fout.write((char*)&err_end, 4);
            int verb = 0;
            fout.write((char*)&verb, 4);
            int n_position = book.size() + pass_boards.size();
            fout.write((char*)&n_position, 4);
            int n_win = 0, n_draw = 0, n_lose = 0;
            uint32_t n_lines;
            short short_val, short_val_min = -HW2, short_val_max = HW2;
            char char_level = (char)level;
            Book_elem book_elem;
            char link_value, link_move;
            int max_link_value, min_link_value;
            char leaf_val, leaf_move;
            char n_link;
            Flip flip;
            Board b;
            bool searching = true;
            int percent = -1;
            int n_boards = (int)book.size();
            int t = 0;
            for (Board pass_board: pass_boards) {
                Board passed_board = pass_board.copy();
                passed_board.pass();
                Book_elem passed_elem = get(passed_board);
                n_lines = passed_elem.n_lines;
                short_val = (short)passed_elem.value;
                if (level == LEVEL_UNDEFINED) {
                    Board b = pass_board.copy();
                    b.pass();
                    if (contain(b)) {
                        char_level = get(b).level;
                    } else {
                        char_level = 1;
                    }
                }
                if (char_level > 60) {
                    char_level = 60;
                }
                n_link = 1;
                link_value = (char)passed_elem.value;
                link_move = MOVE_PASS;
                leaf_val = SCORE_UNDEFINED;
                leaf_move = MOVE_NOMOVE;
                fout.write((char*)&pass_board.player, 8);
                fout.write((char*)&pass_board.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_lines, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val_min, 2);
                fout.write((char*)&short_val_max, 2);
                fout.write((char*)&n_link, 1);
                fout.write((char*)&char_level, 1);
                fout.write((char*)&link_value, 1);
                fout.write((char*)&link_move, 1);
                fout.write((char*)&leaf_val, 1);
                fout.write((char*)&leaf_move, 1);
            }
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                book_elem = itr->second;
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "converting book " << percent << "%" << std::endl;
                }
                ++t;
                short_val = book_elem.value;
                //short_val_min = book_elem.value;
                //short_val_max = book_elem.value;
                b = itr->first;
                std::vector<Book_value> links = get_all_moves_with_value(&b);
                n_link = (char)links.size();
                leaf_val = itr->second.leaf.value;
                leaf_move = itr->second.leaf.move;
                if (leaf_val < -HW2 || HW2 < leaf_val || leaf_move < 0 || HW2 <= leaf_move) {
                    leaf_val = SCORE_UNDEFINED;
                    leaf_move = MOVE_NOMOVE;
                }
                n_lines = itr->second.n_lines;
                if (level == LEVEL_UNDEFINED) {
                    char_level = itr->second.level;
                }
                if (char_level > 60) {
                    char_level = 60;
                }
                fout.write((char*)&itr->first.player, 8);
                fout.write((char*)&itr->first.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_lines, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val_min, 2);
                fout.write((char*)&short_val_max, 2);
                fout.write((char*)&n_link, 1);
                fout.write((char*)&char_level, 1);
                for (Book_value &book_value: links) {
                    link_value = (char)book_value.value;
                    link_move = (char)book_value.policy;
                    fout.write((char*)&link_value, 1);
                    fout.write((char*)&link_move, 1);
                }
                fout.write((char*)&leaf_val, 1);
                fout.write((char*)&leaf_move, 1);
            }
            fout.close();
            std::cerr << "saved " << t << " boards as a edax-formatted book " << n_position << " " << book.size() << std::endl;
        }

        /*
            @brief register a board to book

            @param b                    a board to register
            @param elem                 book element
        */
        inline void reg(Board b, Book_elem elem) {
            register_symmetric_book(b, elem);
        }

        /*
            @brief register a board to book

            @param b                    a board pointer to register
            @param elem                 book element
        */
        inline void reg(Board *b, Book_elem elem) {
            register_symmetric_book(b->copy(), elem);
        }

        /*
            @brief check if book has a board

            @param b                    a board to find
            @return if contains, true, else false
        */
        inline bool contain_representative(Board b) {
            return book.find(b) != book.end();
        }

        /*
            @brief check if book has a board

            @param b                    a board pointer to find
            @return if contains, true, else false
        */
        inline bool contain_representative(Board *b) {
            return contain_representative(b->copy());
        }

        inline bool contain(Board b) {
            return contain_representative(representative_board(b));
        }

        inline bool contain(Board *b) {
            return contain_representative(representative_board(b));
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get_representative(Board b, int idx) {
            Book_elem res;
            if (!contain_representative(b)) {
                return res;
            }
            res = book[b];
            if (is_valid_policy(res.leaf.move)) {
                res.leaf.move = convert_coord_from_representative_board(res.leaf.move, idx);
            }
            return res;
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board pointer to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board *b) {
            int rotate_idx;
            Board representive_board = representative_board(b, &rotate_idx);
            return get_representative(representive_board, rotate_idx);
        }

        /*
            @brief get registered score with all rotation

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get(Board b) {
            int rotate_idx;
            Board representive_board = representative_board(b, &rotate_idx);
            return get_representative(representive_board, rotate_idx);
        }

        /*
            @brief get all best moves

            @param b                    a board pointer to find
            @return vector of best moves
        */
        inline std::vector<int> get_all_best_moves(Board *b) {
            std::vector<int> policies;
            uint64_t legal = b->get_legal();
            int max_value = -INF;
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    int sgn = -1;
                    if (b->get_legal() == 0ULL) {
                        sgn = 1;
                        b->pass();
                    }
                    if (contain(b)) {
                        Book_elem elem = get(b);
                        if (sgn * elem.value > max_value) {
                            max_value = sgn * elem.value;
                            policies.clear();
                        }
                        if (sgn * elem.value == max_value) {
                            policies.emplace_back(cell);
                        }
                    }
                    if (sgn == 1) {
                        b->pass();
                    }
                b->undo_board(&flip);
            }
            return policies;
        }

        /*
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Book_value> get_all_moves_with_value(Board *b) {
            std::lock_guard<std::mutex> lock(mtx);
            std::vector<Book_value> policies;
            uint64_t legal = b->get_legal();
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    if (b->is_end()) {
                        if (contain(b)) {
                            Book_value book_value;
                            book_value.policy = cell;
                            book_value.value = -get(b).value;
                            policies.emplace_back(book_value);
                        } else {
                            b->pass();
                                if (contain(b)) {
                                    Book_value book_value;
                                    book_value.policy = cell;
                                    book_value.value = get(b).value;
                                    policies.emplace_back(book_value);
                                }
                            b->pass();
                        }
                    } else if (b->get_legal() == 0) {
                        b->pass();
                            if (contain(b)) {
                                Book_value book_value;
                                book_value.policy = cell;
                                book_value.value = get(b).value;
                                policies.emplace_back(book_value);
                            }
                        b->pass();
                    } else {
                        if (contain(b)) {
                            Book_value book_value;
                            book_value.policy = cell;
                            book_value.value = -get(b).value;
                            policies.emplace_back(book_value);
                        }
                    }
                b->undo_board(&flip);
            }
            return policies;
        }

        inline Book_value get_random(Board *b, int acc_level, uint64_t use_legal) {
            std::vector<std::pair<double, int>> value_policies;
            std::vector<std::pair<int, int>> value_policies_memo;
            uint64_t legal = b->get_legal();
            double best_score = -INF;
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                if ((1ULL << cell) & use_legal) {
                    calc_flip(&flip, b, cell);
                    b->move_board(&flip);
                        int sgn = -1;
                        if (b->get_legal() == 0ULL) {
                            sgn = 1;
                            b->pass();
                        }
                        if (contain(b)) {
                            Book_value book_value;
                            book_value.policy = cell;
                            book_value.value = sgn * get(b).value;
                            if (book_value.value > best_score) {
                                best_score = (double)book_value.value;
                            }
                            value_policies.emplace_back(std::make_pair((double)book_value.value, cell));
                            value_policies_memo.emplace_back(std::make_pair(book_value.value, cell));
                        }
                        if (sgn == 1) {
                            b->pass();
                        }
                    b->undo_board(&flip);
                }
            }
            Book_elem board_elem = get(b);
            Book_value res;
            if (value_policies.size() == 0 || best_score < board_elem.value - BOOK_LOSS_IGNORE_THRESHOLD) {
                res.policy = -1;
                res.value = -INF;
                return res;
            }
            double acceptable_min_value = best_score - 2.0 * acc_level - 0.5; // acc_level: 0 is best
            double sum_exp_values = 0.0;
            for (std::pair<double, int> &elem: value_policies) {
                if (elem.first < acceptable_min_value) {
                    elem.first = 0.0;
                } else {
                    double exp_val = (exp(elem.first - best_score) + 1.5) / 3.0;
                    elem.first = pow(exp_val, BOOK_ACCURACY_LEVEL_INF - acc_level);
                }
                sum_exp_values += elem.first;
            }
            for (std::pair<double, int> &elem: value_policies)
                elem.first /= sum_exp_values;
            double rnd = myrandom();
            double s = 0.0;
            bool res_got = false;
            for (std::pair<double, int> &elem: value_policies) {
                s += elem.first;
                if (s >= rnd) {
                    res.policy = elem.second;
                    for (std::pair<int, int> elem: value_policies_memo) {
                        if (elem.second == res.policy) {
                            res.value = elem.first;
                        }
                    }
                    res_got = true;
                    break;
                }
            }
            if (!res_got) {
                res.policy = value_policies.back().second;
                for (std::pair<int, int> elem: value_policies_memo) {
                    if (elem.second == res.policy) {
                        res.value = elem.first;
                    }
                }
            }
            return res;
        }

        /*
            @brief get a best move

            @param b                    a board pointer to find
            @param acc_level            accuracy level, 0 is very good, 10 is very bad
            @return best move and value as Book_value structure
        */
        inline Book_value get_random(Board *b, int acc_level) {
            return get_random(b, acc_level, b->get_legal());
        }

        inline Book_value get_specified_best_move(Board *b, uint64_t use_legal) {
            Book_value res;
            res.policy = MOVE_UNDEFINED;
            res.value = -INF;
            Flip flip;
            for (uint_fast8_t cell = first_bit(&use_legal); use_legal; cell = next_bit(&use_legal)) {
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    int sgn = -1;
                    if (b->get_legal() == 0ULL) {
                        sgn = 1;
                        b->pass();
                    }
                    if (contain(b)) {
                        Book_value book_value;
                        book_value.policy = cell;
                        book_value.value = sgn * get(b).value;
                        if (book_value.value > res.value) {
                            res = book_value;
                        }
                    }
                    if (sgn == 1) {
                        b->pass();
                    }
                b->undo_board(&flip);
            }
            return res;
        }

        /*
            @brief get how many boards registered in this book

            @return number of registered boards
        */
        inline int get_n_book() {
            return (int)book.size();
        }

        /*
            @brief change or register a board

            @param b                    a board to change or register
            @param value                a value to change or register
        */
        inline void change(Board b, int value, int level) {
            std::lock_guard<std::mutex> lock(mtx);
            if (-HW2 <= value && value <= HW2) {
                if (b.is_end()) { // game over
                    if (contain(b)) {
                        Board bb = representative_board(b);
                        book[bb].value = value;
                        book[bb].level = level;
                    } else {
                        b.pass();
                        if (contain(b)) {
                            Board bb = representative_board(b);
                            book[bb].value = -value;
                            book[bb].level = level;
                        } else {
                            b.pass();
                            Book_elem elem;
                            elem.value = value;
                            elem.level = level;
                            elem.leaf.move = MOVE_UNDEFINED;
                            elem.leaf.value = SCORE_UNDEFINED;
                            elem.leaf.level = LEVEL_UNDEFINED;
                            register_symmetric_book(b, elem);
                        }
                    }
                } else {
                    if (b.get_legal() == 0) { // just pass
                        b.pass();
                        value *= -1;
                    }
                    if (contain(b)) {
                        Board bb = representative_board(b);
                        book[bb].value = value;
                        book[bb].level = level;
                    } else {
                        Book_elem elem;
                        elem.value = value;
                        elem.level = level;
                        elem.leaf.move = MOVE_UNDEFINED;
                        elem.leaf.value = SCORE_UNDEFINED;
                        elem.leaf.level = LEVEL_UNDEFINED;
                        register_symmetric_book(b, elem);
                    }
                }
            }
        }

        /*
            @brief change or register a board

            @param b                    a board pointer to change or register
            @param value                a value to change or register
        */
        inline void change(Board *b, int value, int level) {
            Board nb = b->copy();
            change(nb, value, level);
        }

        /*
            @brief delete a board

            @param b                    a board to delete
        */
        inline void delete_elem(Board b) {
            delete_symmetric_book(b);
            //if (delete_symmetric_book(b)) {
            //    std::cerr << "deleted book elem " << book.size() << std::endl;
            //} else
            //    std::cerr << "book elem NOT deleted " << book.size() << std::endl;
        }

        /*
            @brief delete all board in this book
        */
        inline void delete_all() {
            //std::cerr << "delete book" << std::endl;
            book.clear();
            reg_first_board();
        }

        /*
            @brief fix book
        */
        inline void fix(bool edax_compliant, bool *stop) {
            negamax_book(edax_compliant, stop);
            check_add_leaf_all_undefined();
        }

        /*
            @brief fix book
        */
        inline void fix(bool edax_compliant) {
            bool stop = false;
            fix(edax_compliant, &stop);
        }

        Book_elem negamax_book_p(Board board, int64_t *n_seen, int64_t *n_fix, int *percent, bool edax_compliant, bool *stop) {
            if (*stop) {
                Book_elem stop_res;
                stop_res.value = SCORE_UNDEFINED;
                stop_res.n_lines = 0;
                return stop_res;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) { // game over
                    if (contain(&board)) {
                        return get(board);
                    } else {
                        board.pass();
                        if (contain(&board)) {
                            return get(board);
                        } else {
                            Book_elem stop_res;
                            stop_res.value = SCORE_UNDEFINED;
                            stop_res.n_lines = 0;
                            return stop_res;
                        }
                    }
                } else { // just pass
                    Book_elem res = negamax_book_p(board, n_seen, n_fix, percent, edax_compliant, stop);
                    if (res.value != SCORE_UNDEFINED) {
                        res.value *= -1;
                    }
                    return res;
                }
            }
            board = representative_board(&board);
            Book_elem res = book[board];
            if (res.seen) {
                return res;
            }
            res.seen = true;
            book[board].seen = true;
            if (res.value < -HW2 || HW2 < res.value) {
                //std::cerr << "value error found " << (int)res.value << std::endl;
                //board.print();
                Book_elem stop_res;
                stop_res.value = SCORE_UNDEFINED;
                stop_res.n_lines = 0;
                return stop_res;
            }
            ++(*n_seen);
            int n_percent = (double)(*n_seen) / book.size() * 100;
            if (n_percent > (*percent)) {
                *percent = n_percent;
                std::cerr << "negamaxing book... " << (*percent) << "%" << " fixed " << (*n_fix) << std::endl;
            }
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            uint64_t n_lines = 1;
            Flip flip;
            //int v = -INF, child_level = -INF;
            int v = -INF;
            if (edax_compliant) {
                v = res.leaf.value;
            }
            //if (res.leaf.value != SCORE_UNDEFINED) {
            //    v = -res.leaf.value;
            //    child_level = res.leaf.level;
            //}
            Book_elem child_res;
            for (Book_value &link: links) {
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    child_res = negamax_book_p(board, n_seen, n_fix, percent, edax_compliant, stop);
                    if (child_res.value != SCORE_UNDEFINED) {
                        if (v < -child_res.value /*&& res.level <= child_res.level*/) { // update parent value
                            v = -child_res.value;
                            //child_level = child_res.level;
                        }
                        n_lines += child_res.n_lines;
                    }
                board.undo_board(&flip);
            }
            if (v != -INF /*&& child_level >= res.level*/) {
                //res.level = child_level;
                if (v != res.value) {
                    res.value = v;
                    ++(*n_fix);
                }
            }
            res.n_lines = (uint32_t)std::min((uint64_t)MAX_N_LINES, n_lines);
            book[board] = res;
            return res;
        }

        void negamax_book(bool edax_compliant, bool *stop) {
            Board root_board;
            root_board.reset();
            int64_t n_seen = 0, n_fix = 0;
            int percent = -1;
            reset_seen();
            negamax_book_p(root_board, &n_seen, &n_fix, &percent, edax_compliant, stop);
            reset_seen();
            std::cerr << "negamaxed book fixed " << n_fix << " boards seen " << n_seen << " boards size " << book.size() << std::endl;
        }

        // flag keeped boards
        void reduce_book_flag_moves(Board board, int max_depth, int max_error_per_move, int remaining_error, uint64_t *n_flags, std::unordered_set<Board, Book_hash> &keep_list, bool *doing) {
            if (!*(doing)) {
                return;
            }
            // depth threshold
            if (board.n_discs() > 4 + max_depth) {
                return;
            }
            // pass
            if (board.get_legal() == 0) { // just pass
                board.pass();
                if (board.get_legal() == 0) { // game over
                    if (keep_list.find(representative_board(board)) != keep_list.end()) {
                        return;
                    }
                    if (contain(&board)) {
                        Book_elem book_elem = get(board);
                        if (book_elem.seen) {
                            return;
                        }
                    } else {
                        board.pass();
                        if (keep_list.find(representative_board(board)) != keep_list.end()) {
                            return;
                        }
                        Book_elem book_elem = get(board);
                        if (book_elem.seen) {
                            return;
                        }
                    }
                    flag_book_elem(board);
                    keep_list.emplace(representative_board(board));
                    ++(*n_flags);
                    return;
                }
            }
            Board unique_board = representative_board(board);
            // already seen
            if (keep_list.find(unique_board) != keep_list.end()) {
                return;
            }
            // not in book
            if (!contain(&board)) {
                return;
            }
            Book_elem book_elem = get(board);
            // already seen?
            if (book_elem.seen) {
                return;
            }
            flag_book_elem(board);
            keep_list.emplace(unique_board);
            ++(*n_flags);
            if ((*n_flags) % 100 == 0) {
                std::cerr << "keep " << (*n_flags) << " boards of " << book.size() << std::endl;
            }
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            for (Book_value &link: links) {
                int link_error = book_elem.value - link.value;
                if (link_error <= max_error_per_move && link_error <= remaining_error) {
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        reduce_book_flag_moves(board, max_depth, max_error_per_move, remaining_error - std::max(0, link_error), n_flags, keep_list, doing);
                    board.undo_board(&flip);
                }
            }
        }

        void update_flagged_leaves(Board board, std::unordered_set<Board, Book_hash> &keep_list, bool *doing) {
            if (!(*doing)) {
                return;
            }
            if (!contain(board)) {
                return;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) {
                    return;
                }
            }
            Book_elem book_elem = get(board);
            // already seen
            if (book_elem.seen) {
                return;
            }
            flag_book_elem(board);
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            if (keep_list.find(representative_board(board)) != keep_list.end()) {
                bool leaf_updated = false;
                for (Book_value &link: links) {
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        bool is_end = board.is_end();
                        bool passed = board.get_legal() == 0;
                        bool will_be_deleted = keep_list.find(representative_board(board)) == keep_list.end();
                        if (is_end) {
                            will_be_deleted = keep_list.find(representative_board(board)) == keep_list.end();
                            board.pass();
                                will_be_deleted |= keep_list.find(representative_board(board)) == keep_list.end();
                            board.pass();
                        } else if (passed) {
                            board.pass();
                                will_be_deleted = keep_list.find(representative_board(board)) == keep_list.end();
                            board.pass();
                        }
                        if (will_be_deleted) {
                            if (book_elem.leaf.value < link.value) {
                                book_elem.leaf.value = link.value;
                                book_elem.leaf.move = link.policy;
                                if (is_end) {
                                    book_elem.leaf.level = get(board).level;
                                    if (book_elem.leaf.level == LEVEL_UNDEFINED) {
                                        board.pass();
                                            book_elem.leaf.level = get(board).level;
                                        board.pass();
                                    }
                                } else if (passed) {
                                    board.pass();
                                        book_elem.leaf.level = get(board).level;
                                    board.pass();
                                } else {
                                    book_elem.leaf.level = get(board).level;
                                }
                                leaf_updated = true;
                            }
                        } else {
                            update_flagged_leaves(board, keep_list, doing);
                        }
                    board.undo_board(&flip);
                }
                if (leaf_updated) {
                    reg(&board, book_elem);
                }
            }
        }

        void delete_unflagged_moves(Board board, uint64_t *n_delete, std::unordered_set<Board, Book_hash> &keep_list, bool *doing) {
            if (!(*doing)) {
                return;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) { // game over
                    bool keep_board = keep_list.find(representative_board(board)) != keep_list.end();
                    board.pass();
                    keep_board |= keep_list.find(representative_board(board)) != keep_list.end();
                    if (!keep_board) {
                        delete_elem(board);
                        board.pass();
                        delete_elem(board);
                        ++(*n_delete);
                    }
                    return;
                }
            }
            if (!contain(board)) {
                return;
            }
            Book_elem book_elem = get(board);
            // already seen
            if (book_elem.seen) {
                return;
            }
            flag_book_elem(board);
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            for (Book_value &link: links) {
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    delete_unflagged_moves(board, n_delete, keep_list, doing);
                board.undo_board(&flip);
            }
            if (keep_list.find(representative_board(board)) == keep_list.end()) {
                delete_elem(board);
                ++(*n_delete);
                if ((*n_delete) % 100 == 0) {
                    std::cerr << "deleting " << (*n_delete) << " boards" << std::endl;
                }
            }
        }

        void reduce_book(Board root_board, int max_depth, int max_error_per_move, int max_line_error, bool *doing) {
            Book_elem book_elem = get(root_board);
            if (book_elem.value == SCORE_UNDEFINED) {
                *doing = false;
                return;
            }
            uint64_t n_flags = 0, n_delete = 0;
            uint64_t book_size = book.size();
            std::unordered_set<Board, Book_hash> keep_list;
            reset_seen();
            reduce_book_flag_moves(root_board, max_depth, max_error_per_move, max_line_error, &n_flags, keep_list, doing);
            reset_seen();
            std::cerr << "updating leaves" << std::endl;
            update_flagged_leaves(root_board, keep_list, doing);
            reset_seen();
            delete_unflagged_moves(root_board, &n_delete, keep_list, doing);
            reset_seen();
            std::cerr << "book reduced size before " << book_size << " n_keep " << n_flags << " n_delete " << n_delete << std::endl;
            *doing = false;
        }

        void delete_terminal_midsearch_rec(Board board, bool *doing) {
            if (!(*doing)) {
                return;
            }
            if (!contain(board)) {
                return;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) {
                    return;
                }
            }
            Book_elem book_elem = get(board);
            // already seen
            if (book_elem.seen) {
                return;
            }
            flag_book_elem(board);
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            if (links.size() == 0) {
                int end_depth = get_level_endsearch_depth(book_elem.level);
                if (HW2 - board.n_discs() > end_depth) {
                    delete_elem(board);
                }
            } else{
                Flip flip;
                for (Book_value &link: links) {
                    calc_flip(&flip, &board, link.policy);
                    board.move_board(&flip);
                        delete_terminal_midsearch_rec(board, doing);
                    board.undo_board(&flip);
                }
            }
        }

        void delete_terminal_midsearch(Board root_board, bool *doing) {
            reset_seen();
            delete_terminal_midsearch_rec(root_board, doing);
            reset_seen();
        }

        uint32_t recalculate_n_lines_rec(Board board, bool *stop) {
            if (*stop) {
                return 0;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) {
                    return 1;
                }
            }
            board = representative_board(&board);
            Book_elem book_elem = get(board);
            // already seen
            if (book_elem.seen) {
                return book_elem.n_lines;
            }
            flag_book_elem(board);
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            uint64_t n_lines = 1;
            Flip flip;
            for (Book_value &link: links) {
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    n_lines += recalculate_n_lines_rec(board, stop);
                board.undo_board(&flip);
            }
            n_lines = (uint32_t)std::min((uint64_t)MAX_N_LINES, n_lines);
            book[board].n_lines = (uint32_t)n_lines;
            return (uint32_t)n_lines;
        }

        uint64_t upgrade_better_leaves_rec(Board board, bool *stop) {
            if (*stop) {
                return 0;
            }
            if (board.get_legal() == 0) {
                board.pass();
                if (board.get_legal() == 0) {
                    return 0;
                }
            }
            uint64_t res = 0;
            board = representative_board(&board);
            Book_elem book_elem = get(board);
            // already seen
            if (book_elem.seen) {
                return 0;
            }
            flag_book_elem(board);
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            if (links.size() && (board.get_legal() & (1ULL << book_elem.leaf.move))) {
                int link_max = -SCORE_INF;
                for (Book_value &link: links) {
                    link_max = std::max(link_max, link.value);
                }
                if (book_elem.leaf.value > link_max) { // upgrade
                    //board.print();
                    //std::cerr << (int)book_elem.value << " " << link_max << " " << idx_to_coord(book_elem.leaf.move) << " " << (int)book_elem.leaf.value << std::endl;
                    calc_flip(&flip, &board, book_elem.leaf.move);
                    board.move_board(&flip);
                        Book_elem new_elem;
                        new_elem.level = book_elem.leaf.level;
                        new_elem.seen = false;
                        new_elem.value = -book_elem.leaf.value;
                        reg(&board, new_elem);
                        ++res;
                        res += upgrade_better_leaves_rec(board, stop);
                    board.undo_board(&flip);
                }
            }
            for (Book_value &link: links) {
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    res += upgrade_better_leaves_rec(board, stop);
                board.undo_board(&flip);
            }
            return res;
        }

        void recalculate_n_lines(Board root_board, bool *stop) {
            std::cerr << "recalculating n_lines..." << std::endl;
            reset_seen();
            recalculate_n_lines_rec(root_board, stop);
            reset_seen();
            std::cerr << "n_lines recalculated" << std::endl;
        }

        void upgrade_better_leaves(Board root_board, bool *stop) {
            std::cerr << "upgrading better leaves..." << std::endl;
            reset_seen();
            uint64_t n = upgrade_better_leaves_rec(root_board, stop);
            reset_seen();
            std::cerr << "upgraded " << n << " leaves" << std::endl;
        }

        uint64_t size() {
            return book.size();
        }

        void reset_seen() {
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                boards.emplace_back(itr->first);
            }
            Flip flip;
            for (Board &board: boards) {
                book[board].seen = false;
            }
        }

        void flag_book_elem(Board board) {
            book[representative_board(board)].seen = true;
        }

        void add_leaf(Board *board, int8_t value, int8_t policy, int8_t level) {
            std::lock_guard<std::mutex> lock(mtx);
            int rotate_idx;
            Board representive_board = representative_board(board, &rotate_idx);
            int8_t rotated_policy = policy;
            if (is_valid_policy(policy)) {
                rotated_policy = convert_coord_from_representative_board((int)policy, rotate_idx);
            }
            Leaf leaf;
            leaf.value = value;
            leaf.move = rotated_policy;
            leaf.level = level;
            book[representive_board].leaf = leaf;
        }

        void search_leaf(Board board, int level, bool use_multi_thread) {
            mtx.lock();
                Book_elem book_elem = book[board];
            mtx.unlock();
            int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            uint64_t remaining_legal = board.get_legal();
            for (Book_value &link: links)
                remaining_legal &= ~(1ULL << link.policy);
            if (remaining_legal) {
                Search_result ai_result = ai_legal(board, level, false, 0, use_multi_thread, false, remaining_legal);
                if (ai_result.value != SCORE_UNDEFINED) {
                    new_leaf_value = ai_result.value;
                    if (level == ADD_LEAF_SPECIAL_LEVEL)
                        new_leaf_value = book_elem.value;
                    new_leaf_move = ai_result.policy;
                }
            } else { // all move seen
                new_leaf_move = MOVE_NOMOVE;
            }
            //std::cerr << (int)new_leaf_value << " " << idx_to_coord(new_leaf_move) << std::endl;
            add_leaf(&board, new_leaf_value, new_leaf_move, level);
        }

        void search_leaf(Board board, int level) {
            search_leaf(board, level, true);
        }

        void check_add_leaf_all_undefined() {
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                boards.emplace_back(itr->first);
            }
            Flip flip;
            for (Board &board: boards) {
                int leaf_move = book[board].leaf.move;
                bool need_to_rewrite_leaf = leaf_move < 0 || MOVE_UNDEFINED <= leaf_move;
                if (!need_to_rewrite_leaf) {
                    calc_flip(&flip, &board, leaf_move);
                    board.move_board(&flip);
                        need_to_rewrite_leaf = contain(&board);
                    board.undo_board(&flip);
                }
                if (need_to_rewrite_leaf) {
                    int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
                    add_leaf(&board, new_leaf_value, new_leaf_move, LEVEL_UNDEFINED);
                }
            }
        }

        void check_add_leaf_all_search(int level, bool *stop) {
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                boards.emplace_back(itr->first);
            }
            Flip flip;
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1, t = 0, n_boards = (int)boards.size();
            Book_elem book_elem;
            std::vector<std::future<void>> tasks;
            int n_doing = 0, n_done = 0;
            for (Board &board: boards) {
                if (!global_searching || *stop) {
                    break;
                }
                int n_percent = (double)t++ / n_boards * 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << " n_recalculated " << n_done << std::endl;
                }
                book_elem = get(board);
                int leaf_move = book_elem.leaf.move;
                bool need_to_rewrite_leaf = leaf_move < 0 || MOVE_UNDEFINED <= leaf_move || (board.get_legal() & (1ULL << leaf_move)) == 0;
                if (!need_to_rewrite_leaf) {
                    calc_flip(&flip, &board, leaf_move);
                    board.move_board(&flip);
                        if (board.get_legal() == 0) {
                            board.pass();
                                need_to_rewrite_leaf = contain(&board);
                            board.pass();
                        } else {
                            need_to_rewrite_leaf = contain(&board);
                        }
                    board.undo_board(&flip);
                }
                if (need_to_rewrite_leaf) {
                    bool use_multi_thread = (n_boards - n_doing) < thread_pool.get_n_idle();
                    bool pushed;
                    ++n_doing;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&search_new_leaf, board, level, book_elem.value, use_multi_thread)));
                    if (!pushed) {
                        tasks.pop_back();
                        search_new_leaf(board, level, book_elem.value, true);
                        ++n_done;
                    }
                }
                int tasks_size = tasks.size();
                for (int i = 0; i < tasks_size; ++i) {
                    if (tasks[i].valid()) {
                        if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            tasks[i].get();
                            ++n_done;
                        }
                    }
                }
                for (int i = 0; i < tasks_size; ++i) {
                    if (i >= tasks.size()) {
                        break;
                    }
                    if (!tasks[i].valid()) {
                        tasks.erase(tasks.begin() + i);
                        --i;
                    }
                }
            }
            int tasks_size = tasks.size();
            for (int i = 0; i < tasks_size; ++i) {
                if (tasks[i].valid()) {
                    tasks[i].get();
                    ++n_done;
                }
            }
            std::cerr << "leaf search finished" << std::endl;
        }

        void recalculate_leaf_all(int level, bool *stop) {
            std::vector<Board> boards;
            for (auto itr = book.begin(); itr != book.end(); ++itr) {
                boards.emplace_back(itr->first);
            }
            Flip flip;
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1, t = 0, n_boards = (int)boards.size();
            for (Board &board: boards) {
                if (!global_searching || *stop) {
                    break;
                }
                int n_percent = (double)t / n_boards * 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << std::endl;
                }
                ++t;
                search_leaf(board, level);
            }
        }

        Book_info calculate_book_info(bool *calculating) {
            Book_info res;
            for (auto itr = book.begin(); itr != book.end() && *calculating; ++itr) {
                int level = itr->second.level;
                int ply = itr->first.n_discs() - 4;
                int leaf_level = itr->second.leaf.level;
                int leaf_ply = ply + 1;
                ++res.n_boards;
                if (0 <= level && level <= LEVEL_HUMAN) {
                    ++res.n_boards_in_level[level];
                }
                ++res.n_boards_in_ply[ply];
                if (0 <= leaf_level && leaf_level <= LEVEL_HUMAN) {
                    ++res.n_leaves_in_level[leaf_level];
                    ++res.n_leaves_in_ply[leaf_ply];
                }
            }
            return res;
        }

    private:
        void reg_first_board() {
            Board board;
            board.reset();
            Book_elem elem;
            elem.value = 0;
            elem.leaf.value = 0;
            elem.leaf.move = 19;
            book[board] = elem;
        }

        /*
            @brief register a board

            @param b                    a board to register
            @param value                score of the board
            @return is this board new?
        */
        inline bool register_representative(Board b, Book_elem elem) {
            /*
            if (elem.value < -HW2 && HW2 < elem.value)
                return false;
            std::vector<Book_value> moves;
            for (Book_value &move: elem.moves) {
                if (-HW2 <= move.value && move.value <= HW2)
                    moves.emplace_back(move);
            }
            elem.moves = moves;
            */
            int f_size = book.size();
            book[b] = elem;
            return book.size() - f_size > 0;
        }

        /*
            @brief delete a board

            @param b                    a board to delete
            @return board deleted?
        */
        inline bool delete_representative_board(Board b) {
            if (book.find(b) != book.end()) {
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
        inline int register_symmetric_book(Board b, Book_elem elem) {
            int idx;
            Board representive_board = representative_board(b, &idx);
            if (elem.leaf.move != MOVE_UNDEFINED) {
                elem.leaf.move = convert_coord_to_representative_board(elem.leaf.move, idx);
            }
            return register_representative(representive_board, elem);
        }

        /*
            @brief delete a board with checking all symmetry boards

            @param b                    a board to delete
            @return 1 if board is deleted (board found) else 0
        */
        inline int delete_symmetric_book(Board b) {
            Board representive_board = representative_board(b);
            return delete_representative_board(representive_board);
        }

        inline int merge(Board b, Book_elem elem) {
            if (!contain(b)) {
                return register_symmetric_book(b, elem);
            }
            Book_elem book_elem = get(b);
            if (elem.value != SCORE_UNDEFINED && book_elem.level <= elem.level) {
                book_elem.value = elem.value;
                book_elem.level = elem.level;
            }
            if (elem.leaf.value != SCORE_UNDEFINED && book_elem.leaf.level <= elem.leaf.level) {
                book_elem.leaf.value = elem.leaf.value;
                book_elem.leaf.move = elem.leaf.move;
                book_elem.leaf.level = elem.leaf.level;
            }
            return register_symmetric_book(b, book_elem);
        }
};








Book book;

bool book_init(std::string file, bool show_log) {
    //book_hash_init(show_log);
    book_hash_init_rand();
    bool stop_loading = false;
    return book.init(file, show_log, &stop_loading);
}

bool book_init(std::string file, bool show_log, bool *stop_loading) {
    //book_hash_init(show_log);
    book_hash_init_rand();
    return book.init(file, show_log, stop_loading);
}

void book_save_as_egaroucid(std::string file, int level) {
    book.save_egbk3(file, level);
}

void book_save_as_edax(std::string file, int level) {
    book.save_bin_edax(file, level);
}

void book_fix(bool *stop) {
    book.fix(false, stop);
}

void book_fix_edax(bool *stop) {
    book.fix(true, stop);
}

void book_reduce(Board board, int depth, int max_error_per_move, int max_error_sum, bool *doing) {
    book.reduce_book(board, depth, max_error_per_move, max_error_sum, doing);
}

void book_recalculate_leaf_all(int level, bool *stop) {
    book.recalculate_leaf_all(level, stop);
}

void book_recalculate_n_lines_all(bool *stop) {
    Board root_board;
    root_board.reset();
    book.recalculate_n_lines(root_board, stop);
}

void book_upgrade_better_leaves_all(bool *stop) {
    Board root_board;
    root_board.reset();
    book.upgrade_better_leaves(root_board, stop);
}

void search_new_leaf(Board board, int level, int book_elem_value, bool use_multi_thread) {
    int8_t new_leaf_value = SCORE_UNDEFINED, new_leaf_move = MOVE_UNDEFINED;
    std::vector<Book_value> links = book.get_all_moves_with_value(&board);
    uint64_t legal = board.get_legal();
    for (Book_value &link: links) {
        legal &= ~(1ULL << link.policy);
    }
    if (legal) {
        int use_level = level;
        if (level == ADD_LEAF_SPECIAL_LEVEL) {
            use_level = 1;
        }
        Search_result ai_result = ai_legal(board, use_level, false, 0, use_multi_thread, false, legal);
        if (ai_result.value != SCORE_UNDEFINED) {
            new_leaf_value = ai_result.value;
            if (level == ADD_LEAF_SPECIAL_LEVEL) {
                new_leaf_value = book_elem_value;
            }
            new_leaf_move = ai_result.policy;
            //std::cerr << "recalc leaf " << (int)new_leaf_value << " " << (int)new_leaf_move << " " << idx_to_coord(new_leaf_move) << std::endl;
            //for (Book_value &link: links)
            //    std::cerr << "link " << idx_to_coord(link.policy) << std::endl;
            //board.print();
        }
    } else {
        new_leaf_move = MOVE_NOMOVE;
    }
    book.add_leaf(&board, new_leaf_value, new_leaf_move, level);
}

Book_info calculate_book_info(bool *calculating) {
    Book_info res = book.calculate_book_info(calculating);
    return res;
}