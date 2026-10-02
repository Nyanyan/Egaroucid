/*
    Egaroucid Project

    @file book.hpp
        Book class
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <future>
#include <cstdio>
#include <cstring>
#include "evaluate.hpp"
#include "board.hpp"
#include "search.hpp"
#include "ai.hpp"
#include "book_table.hpp"

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
constexpr int ADD_LEAF_SPECIAL_LEVEL_SEARCH_LEVEL = 5;

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

    The size of this structure determines the memory usage of book: keep it 12 bytes.

    @param n_lines              number of lines
    @param value                registered score
    @param level                AI level
    @param leaf                 best move which is not registered in book
    @param seen                 used in various situation
    @param exported             used in Edax export (the board is written to the Edax book)
*/
struct Book_elem {
    uint32_t n_lines;
    int8_t value;
    int8_t level;
    Leaf leaf;
    bool seen; // used in various situation
    bool exported; // used in Edax export

    Book_elem()
        : n_lines(0), value(SCORE_UNDEFINED), level(LEVEL_UNDEFINED), seen(false), exported(false) {}
};

static_assert(sizeof(Book_elem) == 12, "Book_elem must be 12 bytes");

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
    @brief initialize hash for book

    Hash function for book (Book_hash in book_table.hpp) needs no table.
    These functions are kept for compatibility.
*/
inline void book_hash_init_rand() {
}

inline void book_hash_init(bool show_log) {
}

/*
    @brief get the size of a file

    @param fp                   file opened in binary mode (the position is not changed)
    @return file size in bytes (-1 if failed)
*/
inline int64_t book_get_file_size(FILE *fp) {
#ifdef _WIN32
    int64_t pos = _ftelli64(fp);
    if (pos < 0 || _fseeki64(fp, 0, SEEK_END) != 0) {
        return -1;
    }
    int64_t res = _ftelli64(fp);
    _fseeki64(fp, pos, SEEK_SET);
#else
    off_t pos = ftello(fp);
    if (pos < 0 || fseeko(fp, 0, SEEK_END) != 0) {
        return -1;
    }
    int64_t res = (int64_t)ftello(fp);
    fseeko(fp, pos, SEEK_SET);
#endif
    return res;
}

/*
    @brief buffered writer for book files

    Book files consist of many small records. Writing them one by one through std::ofstream is slow.
*/
class Book_file_writer {
    private:
        static constexpr size_t BUFFER_SIZE = 1 << 20;
        FILE *fp;
        std::vector<char> buffer;
        size_t pos;
        bool failed;

    public:
        Book_file_writer()
            : fp(nullptr), buffer(BUFFER_SIZE), pos(0), failed(false) {}

        ~Book_file_writer() {
            close();
        }

        bool open(const std::string &file) {
            failed = !file_open(&fp, file.c_str(), "wb");
            if (failed) {
                fp = nullptr;
            }
            pos = 0;
            return !failed;
        }

        inline void write(const void *data, size_t size) {
            if (pos + size > BUFFER_SIZE) {
                flush();
                if (size > BUFFER_SIZE) { // larger than the buffer: write directly
                    if (fp != nullptr && fwrite(data, 1, size, fp) != size) {
                        failed = true;
                    }
                    return;
                }
            }
            memcpy(buffer.data() + pos, data, size);
            pos += size;
        }

        void flush() {
            if (fp != nullptr && pos && fwrite(buffer.data(), 1, pos, fp) != pos) {
                failed = true;
            }
            pos = 0;
        }

        // overwrite data written before (file position returns to the end)
        void write_at(long offset, const void *data, size_t size) {
            flush();
            if (fp == nullptr) {
                return;
            }
            if (fseek(fp, offset, SEEK_SET) != 0 || fwrite(data, 1, size, fp) != size || fseek(fp, 0, SEEK_END) != 0) {
                failed = true;
            }
        }

        // returns true if everything is written
        bool close() {
            if (fp != nullptr) {
                flush();
                if (fclose(fp) != 0) {
                    failed = true;
                }
                fp = nullptr;
            }
            return !failed;
        }
};






/*
    @brief book data

    @param book                 book data
*/
class Book {
    private:
        std::mutex mtx;
        Board_table<Book_elem> book;

    public:
        /*
            @brief initialize book

            @param file                 book file (.egbk file)
            @return book completely imported?
        */
        bool init(std::string file, bool show_log, bool *stop_loading) {
            delete_all();
            // The format is determined with the header.
            // A broken egbk3 / egbk2 book must not be read as egbk format, which has no header (garbage would be imported).
            int book_version = get_egaroucid_book_version(file);
            if (book_version == 3 || book_version == BOOK_VERSION_CANT_OPEN) {
                return import_file_egbk3(file, show_log, stop_loading);
            }
            if (book_version == 2) {
                std::cerr << "not egbk3 formatted book. trying egbk2 format." << std::endl;
                return import_file_egbk2(file, show_log, stop_loading);
            }
            if (book_version == BOOK_VERSION_NO_HEADER) {
                std::cerr << "not egbk3 / egbk2 formatted book. trying egbk format." << std::endl;
                return import_file_egbk(file, 1, show_log, stop_loading);
            }
            std::cerr << "[ERROR] unknown Egaroucid book version " << book_version << std::endl;
            return false;
        }

        /*
            @brief get the version of Egaroucid-formatted book written in the header

            @param file                 book file
            @return version, BOOK_VERSION_NO_HEADER if the file has no header, BOOK_VERSION_CANT_OPEN if the file can't be opened
        */
        static constexpr int BOOK_VERSION_NO_HEADER = 0;
        static constexpr int BOOK_VERSION_CANT_OPEN = -1;
        inline int get_egaroucid_book_version(const std::string &file) {
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                return BOOK_VERSION_CANT_OPEN;
            }
            char header[10];
            size_t n_read = fread(header, 1, 10, fp);
            fclose(fp);
            if (n_read < 10 || memcmp(header, "DICUORAGE", 9) != 0) {
                return BOOK_VERSION_NO_HEADER;
            }
            return (unsigned char)header[9];
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
            } else if (lst[lst.size() - 1] == "egbk2") {
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
            @brief merge a book element into a registered one

            The one searched with higher level is used.
        */
        static inline void merge_book_elem(Book_elem &registered, const Book_elem &elem) {
            if (elem.value != SCORE_UNDEFINED && registered.level <= elem.level) {
                registered.value = elem.value;
                registered.level = elem.level;
            }
            if (elem.leaf.value != SCORE_UNDEFINED && registered.leaf.level <= elem.leaf.level) {
                registered.leaf.value = elem.leaf.value;
                registered.leaf.move = elem.leaf.move;
                registered.leaf.level = elem.leaf.level;
            }
        }

        /*
            @brief Parallel import of the boards in Egaroucid-formatted book (.egbk3)

            This thread reads the file and worker threads register boards.
        */
        inline bool import_egbk3_parallel(FILE* fp, uint32_t n_boards, bool show_log, bool *stop_loading) {
            constexpr uint32_t n_chunk = 65536;
            constexpr size_t n_bytes_per_board = 25;

            const int n_workers = thread_pool.size();
            const int n_buffers = std::max(1, n_workers);
            if (show_log) {
                std::cerr << "book import parallelized with " << n_buffers << " threads" << std::endl;
            }

            // Reserve capacity for the book to avoid resizing
            // (the number of boards written in the header may be broken: do not trust it over the file size)
            uint64_t n_boards_reserve = n_boards;
            int64_t file_size = book_get_file_size(fp);
#ifdef _WIN32
            int64_t file_pos = _ftelli64(fp);
#else
            int64_t file_pos = (int64_t)ftello(fp);
#endif
            if (file_size >= 0 && file_pos >= 0) {
                n_boards_reserve = std::min(n_boards_reserve, (uint64_t)(file_size - file_pos) / n_bytes_per_board);
            }
            book.reserve(book.size() + n_boards_reserve);

            std::vector<std::vector<char>> data_chunks(n_buffers);
            std::unique_ptr<std::atomic<bool>[]> buffer_busy(new std::atomic<bool>[n_buffers]);
            std::vector<std::future<void>> futures(n_buffers);
            for (int i = 0; i < n_buffers; ++i) {
                buffer_busy[i].store(false);
            }

            // register boards in a chunk
            auto process_chunk = [&](int buffer_idx, uint32_t n_boards_in_chunk) {
                const char *data = data_chunks[buffer_idx].data();
                Board board;
                Book_elem book_elem;
                int64_t n_registered = 0;
                for (uint32_t j = 0; j < n_boards_in_chunk; ++j) {
                    if ((j & 1023) == 0 && *stop_loading) {
                        break;
                    }
                    const char *datum = data + (size_t)j * n_bytes_per_board;
                    memcpy(&board.player, datum, 8);
                    memcpy(&board.opponent, datum + 8, 8);
                    int8_t value = (int8_t)datum[16];
                    if (-HW2 <= value && value <= HW2 && (board.player & board.opponent) == 0) {
#if FORCE_BOOK_DEPTH
                        if (board.n_discs() <= 4 + 30) {
#endif
                            book_elem.value = value;
                            book_elem.level = (int8_t)datum[17];
                            memcpy(&book_elem.n_lines, datum + 18, 4);
                            book_elem.leaf.value = (int8_t)datum[22];
                            book_elem.leaf.move = (int8_t)datum[23];
                            book_elem.leaf.level = (int8_t)datum[24];
                            int idx;
                            Board rep_board = representative_board(board, &idx);
                            if (idx != 0 && is_valid_policy(book_elem.leaf.move)) {
                                book_elem.leaf.move = convert_coord_to_representative_board(book_elem.leaf.move, idx);
                            }
                            n_registered += book.upsert(rep_board, book_elem, [&book_elem](Book_elem &registered) {
                                merge_book_elem(registered, book_elem);
                            }, false);
#if FORCE_BOOK_DEPTH
                        }
#endif
                    }
                }
                book.add_size(n_registered);
                buffer_busy[buffer_idx].store(false, std::memory_order_release);
            };

            // This thread: file reading and work distribution
            bool error = false;
            uint64_t n_read = 0;
            int percent = -1;
            while (n_read < n_boards && !(*stop_loading)) {
                // Find a free buffer
                int buffer_idx = -1;
                for (int t = 0; t < n_buffers; ++t) {
                    if (!buffer_busy[t].load(std::memory_order_acquire)) {
                        buffer_idx = t;
                        break;
                    }
                }
                if (buffer_idx == -1) {
                    // No idle thread, yield CPU
                    std::this_thread::yield();
                    continue;
                }
                if (show_log) {
                    int n_percent = (double)n_read / n_boards * 100;
                    if (n_percent > percent) {
                        percent = n_percent;
                        std::cerr << "loading book " << percent << "%" << std::endl;
                    }
                }
                // Load next chunk and assign to the thread
                uint32_t n_boards_chunk = (uint32_t)std::min<uint64_t>(n_chunk, n_boards - n_read);
                size_t n_data = (size_t)n_boards_chunk * n_bytes_per_board;
                if (data_chunks[buffer_idx].empty()) {
                    data_chunks[buffer_idx].resize((size_t)n_chunk * n_bytes_per_board);
                }
                if (fread(data_chunks[buffer_idx].data(), 1, n_data, fp) < n_data) {
                    std::cerr << "[ERROR] book NOT FULLY imported " << book.size() << " boards" << std::endl;
                    error = true;
                    break;
                }
                n_read += n_boards_chunk;
                buffer_busy[buffer_idx].store(true, std::memory_order_release);
                bool pushed = false;
                if (n_workers > 0) {
                    std::future<void> future = thread_pool.push(&pushed, [&process_chunk, buffer_idx, n_boards_chunk]() {
                        process_chunk(buffer_idx, n_boards_chunk);
                    });
                    if (pushed) {
                        futures[buffer_idx] = std::move(future);
                    }
                }
                if (!pushed) {
                    process_chunk(buffer_idx, n_boards_chunk);
                }
            }

            // Wait for all threads to complete
            for (int t = 0; t < n_buffers; ++t) {
                if (futures[t].valid()) {
                    futures[t].wait();
                }
            }

            if (error) {
                return false;
            }
            return !(*stop_loading);
        }

        /*
            @brief import Egaroucid-formatted book

            @param file                 book file (.egbk3 file)
            @return book completely imported?
        */
        inline bool import_file_egbk3(std::string file, bool show_log, bool *stop_loading) {
            uint64_t strt = tim();
            if (show_log) {
                std::cerr << "importing " << file << std::endl;
            }
            FILE* fp;
            if (!file_open(&fp, file.c_str(), "rb")) {
                std::cerr << "[ERROR] can't open Egaroucid book " << file << std::endl;
                return false;
            }
            uint32_t n_boards;
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

            if (!import_egbk3_parallel(fp, n_boards, show_log, stop_loading)) {
                if (*stop_loading) {
                    std::cerr << "stop loading book" << std::endl;
                }
                fclose(fp);
                return false;
            }
            if (show_log) {
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
                std::cerr << "elapsed " << tim() - strt << " ms" << std::endl;
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
                    std::cerr << "stop loading book" << std::endl;
                    fclose(fp);
                    return false;
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
            // this format has no header: check the file size not to import a file which is not a book
            constexpr int64_t n_bytes_per_board = 17;
            int64_t file_size = book_get_file_size(fp);
            if (n_boards < 0 || (file_size >= 0 && file_size != 4 + n_bytes_per_board * n_boards)) {
                std::cerr << "[ERROR] This is not Egaroucid book (egbk format), " << n_boards << " boards expected but file size is " << file_size << std::endl;
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
                    std::cerr << "stop loading book" << std::endl;
                    fclose(fp);
                    return false;
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
                    fclose(fp);
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
                if (value != SCORE_UNDEFINED && (player & opponent) == 0ULL && calc_legal(player, opponent)) {
                    board.player = player;
                    board.opponent = opponent;
                    book_elem.value = value;
                    book_elem.level = level;
                    book_elem.leaf.value = SCORE_UNDEFINED;
                    book_elem.leaf.move = MOVE_UNDEFINED;
                    book_elem.leaf.level = level;
                    book_elem.n_lines = n_lines;
                    merge(board, book_elem);
                }
                // read leaf move (imported as position in Egaroucid book)
                if (fread(&leaf_move, 1, 1, fp) < 1) {
                    std::cerr << "[ERROR] file broken" << std::endl;
                    fclose(fp);
                    return false;
                }
                if (is_valid_score(leaf_value) && is_valid_policy(leaf_move)) {
                    board.player = player;
                    board.opponent = opponent;
                    calc_flip(&flip, &board, leaf_move);
                    board.move_board(&flip);
                    leaf_value = -leaf_value;
                    if (board.get_legal() == 0 && !board.is_end()) {
                        board.pass();
                        leaf_value = -leaf_value;
                    }
                    book_elem.value = leaf_value;
                    book_elem.level = level;
                    book_elem.leaf.value = SCORE_UNDEFINED;
                    book_elem.leaf.move = MOVE_UNDEFINED;
                    book_elem.leaf.level = level;
                    book_elem.n_lines = 1;
                    merge(board, book_elem);
                }
            }
            if (show_log) {
                std::cerr << "imported " << book.size() << " boards to book" << std::endl;
            }
            fclose(fp);
            return true;
        }

        /*
            @brief save as Egaroucid-formatted book (.egbk3)

            With backup, the book is written to a temporary file first,
            so that neither the book file nor its backup is broken if saving fails (disk full etc.).

            @param file                 file name to save
            @param bak_file             backup file name
            @return book completely saved?
        */
        inline bool save_egbk3(std::string file, std::string bak_file, bool use_backup, int level, bool *stop_saving) {
            std::string write_file = file;
            if (use_backup) {
                if (remove(bak_file.c_str()) == -1) {
                    std::cerr << "cannot delete backup. you can ignore this error." << std::endl;
                }
                write_file = file + ".tmp";
            }
            Book_file_writer fout;
            if (!fout.open(write_file)) {
                std::cerr << "can't open " << write_file << std::endl;
                return false;
            }
            std::cerr << "saving book..." << std::endl;
            char egaroucid_str[] = "DICUORAGE";
            fout.write(egaroucid_str, 9);
            char book_version = 3;
            fout.write(&book_version, 1);
            uint32_t n_book = 0; // rewritten after all boards are written
            fout.write(&n_book, 4);
            uint64_t t = 0, n_boards = book.size();
            int percent = -1;
            bool stopped = false;
            // each segment is locked while it is written: other threads may be registering boards (book learning)
            for (int segment_idx = 0; segment_idx < book.N_SEGMENTS && !stopped; ++segment_idx) {
                stopped = !book.for_each_in_segment_locked_while(segment_idx, [&](const Board &board, const Book_elem &book_elem) {
                    if (stop_saving && *stop_saving) {
                        return false;
                    }
                    ++t;
                    int n_percent = n_boards ? (int)std::min(100.0, (double)t / n_boards * 100) : 100;
                    if (n_percent > percent) {
                        percent = n_percent;
                        std::cerr << "saving book " << percent << "%" << std::endl;
                    }
                    char datum[25];
                    memcpy(datum, &board.player, 8);
                    memcpy(datum + 8, &board.opponent, 8);
                    datum[16] = book_elem.value;
                    datum[17] = level != LEVEL_UNDEFINED ? (char)level : book_elem.level;
                    memcpy(datum + 18, &book_elem.n_lines, 4);
                    datum[22] = book_elem.leaf.value;
                    datum[23] = book_elem.leaf.move;
                    datum[24] = level != LEVEL_UNDEFINED ? (char)level : book_elem.leaf.level;
                    fout.write(datum, 25);
                    return true;
                });
            }
            if (t > 0xFFFFFFFFULL) {
                std::cerr << "[ERROR] too many boards to save as egbk3 " << t << std::endl;
                stopped = true;
            }
            n_book = (uint32_t)t;
            fout.write_at(10, &n_book, 4);
            bool saved = fout.close();
            if (stop_saving && *stop_saving) {
                remove(write_file.c_str());
                std::cerr << "book saving stopped" << std::endl;
                return false;
            }
            if (!saved || stopped) {
                std::cerr << "[ERROR] failed to save book " << write_file << std::endl;
                if (use_backup) {
                    remove(write_file.c_str()); // the book file is kept as it is
                }
                return false;
            }
            if (use_backup) {
                rename(file.c_str(), bak_file.c_str());
                if (rename(write_file.c_str(), file.c_str()) != 0) {
                    remove(file.c_str());
                    if (rename(write_file.c_str(), file.c_str()) != 0) {
                        std::cerr << "[ERROR] can't rename " << write_file << " to " << file << std::endl;
                        return false;
                    }
                }
            }
            std::cerr << "saved " << t << " boards , book_size " << book.size() << std::endl;
            return true;
        }

        inline bool save_egbk3(std::string file, std::string bak_file, bool use_backup, int level) {
            return save_egbk3(file, bak_file, use_backup, level, nullptr);
        }

        inline bool save_egbk3(std::string file, std::string bak_file) {
            return save_egbk3(file, bak_file, true, LEVEL_UNDEFINED);
        }

        inline bool save_egbk3(std::string file, int level) {
            return save_egbk3(file, "", false, level);
        }

        inline bool save_egbk3(std::string file, int level, bool *stop_saving) {
            return save_egbk3(file, "", false, level, stop_saving);
        }

        /*
            @brief find the boards where the player must pass, for Edax-formatted book

            Egaroucid's book has only the board after pass, Edax's book needs the board before pass.

            @param only_exported        true: only the boards with exported flag are regarded as registered
        */
        void get_pass_boards(Board board, std::unordered_set<Board, Book_hash> &pass_boards, bool additional_calculation = true, bool only_exported = false, bool *stop_saving = nullptr) {
            // The boards which can be reached from the root with the moves written to Edax book are flagged (seen),
            // and the boards before pass found on the way are collected.
            // Boards with fewer discs are processed first, boards with the same number of discs in parallel
            // (a move always leads to a board with 1 more disc, so the result is the same as a recursive search).
            // Boards must not be registered / deleted by other threads (functions without lock are used).
            if (stop_saving && *stop_saving) {
                return;
            }
            // check the moves of a board: flag the next boards and collect the boards before pass
            auto check_moves = [this, only_exported](const Board &b, std::vector<Board> &found_pass_boards) {
                Moved_board moved_boards[HW2];
                int n_moved_boards = get_moved_boards(&b, moved_boards, true);
                bool flagged = false;
                Book_elem book_elem;
                for (int i = 0; i < n_moved_boards; ++i) {
                    const Moved_board &moved_board = moved_boards[i];
                    if (moved_board.is_end) { // game over
                        if (book.get_unlocked(moved_board.board, moved_board.hash, &book_elem)) {
                            continue; // registered as it is: not a board before pass
                        }
                        if (book.get_unlocked(moved_board.end_board, moved_board.end_hash, &book_elem) && (!only_exported || book_elem.exported)) {
                            found_pass_boards.emplace_back(moved_board.board);
                        }
                    } else if (book.get_unlocked(moved_board.board, moved_board.hash, &book_elem) && (!only_exported || book_elem.exported)) {
                        if (moved_board.passed) { // moved_board.board is the board after pass
                            Board board_before_pass = representative_board(Board(moved_board.board.opponent, moved_board.board.player));
                            if (book.contains(board_before_pass)) {
                                continue; // the board before pass is registered: not followed (same as the recursive search)
                            }
                            found_pass_boards.emplace_back(board_before_pass);
                        }
                        flagged |= book.modify_unlocked(moved_board.board, moved_board.hash, [](Book_elem &child_elem) {
                            if (!child_elem.seen) { // several threads may flag the same board
                                child_elem.seen = true;
                            }
                        });
                    }
                }
                return flagged;
            };
            // root
            std::vector<Board> root_pass_boards;
            if (board.get_legal() == 0ULL) {
                Board passed_board = board.copy();
                passed_board.pass();
                if (!contain(board) && contain_edax_export_board(passed_board, only_exported)) {
                    pass_boards.emplace(representative_board(board));
                    board = passed_board;
                } else {
                    return;
                }
                if (board.get_legal() == 0ULL) { // game over
                    return;
                }
            }
            bool root_registered = book.modify(representative_board(board), [](Book_elem &book_elem) {
                book_elem.seen = true;
            });
            if (!root_registered) { // moves of an unregistered root are also followed
                check_moves(representative_board(board), root_pass_boards);
                pass_boards.insert(root_pass_boards.begin(), root_pass_boards.end());
            }
            for (int n_discs = board.n_discs() + (root_registered ? 0 : 1); n_discs < HW2; ++n_discs) {
                if (stop_saving && *stop_saving) {
                    return;
                }
                std::atomic<bool> flagged(false);
                std::vector<std::vector<Board>> found_pass_boards(book.N_SEGMENTS);
                parallel_for_segments([this, n_discs, stop_saving, &flagged, &found_pass_boards, &check_moves](int segment_idx) {
                    book.for_each_in_segment(segment_idx, [n_discs, stop_saving, &flagged, &found_pass_boards, &check_moves, segment_idx](const Board &b, Book_elem &book_elem) {
                        if (b.n_discs() != n_discs || !book_elem.seen || (stop_saving && *stop_saving)) {
                            return;
                        }
                        if (check_moves(b, found_pass_boards[segment_idx]) && !flagged.load(std::memory_order_relaxed)) {
                            flagged.store(true, std::memory_order_relaxed);
                        }
                    });
                });
                for (const std::vector<Board> &boards: found_pass_boards) {
                    pass_boards.insert(boards.begin(), boards.end());
                }
                if (!flagged.load()) {
                    break;
                }
            }
        }

        inline bool has_book_value_policy(const std::vector<Book_value> &values, int policy) {
            for (const Book_value &value: values) {
                if (value.policy == policy) {
                    return true;
                }
            }
            return false;
        }

        inline int choose_best_book_value_idx(const std::vector<Book_value> &values) {
            int best_idx = 0;
            for (int i = 1; i < (int)values.size(); ++i) {
                if (values[i].value > values[best_idx].value) {
                    best_idx = i;
                }
            }
            return best_idx;
        }

        inline int choose_worst_book_value_idx(const std::vector<Book_value> &values) {
            int worst_idx = 0;
            for (int i = 1; i < (int)values.size(); ++i) {
                if (values[i].value < values[worst_idx].value) {
                    worst_idx = i;
                }
            }
            return worst_idx;
        }

        inline void add_leaf_candidate_unique(std::vector<Book_value> &leaf_candidates, Book_value candidate) {
            for (Book_value &leaf_candidate: leaf_candidates) {
                if (leaf_candidate.policy == candidate.policy) {
                    if (candidate.value > leaf_candidate.value) {
                        leaf_candidate.value = candidate.value;
                    }
                    return;
                }
            }
            leaf_candidates.emplace_back(candidate);
        }

        inline bool get_edax_child_board_after_move(Board board, int policy, Board *child) {
            Flip flip;
            calc_flip(&flip, &board, policy);
            board.move_board(&flip);
            if (board.is_end()) {
                if (contain(board)) {
                    *child = representative_board(board);
                    return true;
                }
                board.pass();
                if (contain(board)) {
                    *child = representative_board(board);
                    return true;
                }
                return false;
            }
            if (board.get_legal() == 0ULL) {
                board.pass();
                if (contain(board)) {
                    *child = representative_board(board);
                    return true;
                }
                return false;
            }
            if (contain(board)) {
                *child = representative_board(board);
                return true;
            }
            return false;
        }

        /*
            @brief find the board to remove from Edax export, so that the given board has a leaf

            @param board                representative board (registered)
            @param elem                 book element of the board
            @param child_to_remove      the board after the move which becomes the leaf
            @return true if a board must be removed
        */
        inline bool find_leaf_demoted_edax_child_without_additional_calculation(Board board, const Book_elem &elem, Board *child_to_remove) {
            uint64_t legal = board.get_legal();
            if (legal == 0ULL) {
                return false;
            }
            std::vector<Book_value> all_links, links;
            get_all_moves_and_edax_links(&board, all_links, links);
            int mobility = pop_count_ull(legal);
            if ((int)links.size() == mobility) {
                return false;
            }
            std::vector<Book_value> leaf_candidates;
            bool registered_leaf_valid = is_valid_score(elem.leaf.value) && is_valid_policy(elem.leaf.move) && ((legal & (1ULL << elem.leaf.move)) != 0ULL);
            if (registered_leaf_valid) {
                Book_value leaf_candidate;
                leaf_candidate.policy = elem.leaf.move;
                leaf_candidate.value = elem.leaf.value;
                add_leaf_candidate_unique(leaf_candidates, leaf_candidate);
            }
            for (const Book_value &link: all_links) {
                if (!has_book_value_policy(links, link.policy)) {
                    add_leaf_candidate_unique(leaf_candidates, link);
                }
            }
            if (!leaf_candidates.empty()) {
                int leaf_idx = choose_best_book_value_idx(leaf_candidates);
                if (has_book_value_policy(links, leaf_candidates[leaf_idx].policy)) {
                    return get_edax_child_board_after_move(board, leaf_candidates[leaf_idx].policy, child_to_remove);
                }
                return false;
            }
            if (links.empty()) {
                return false;
            }
            int worst_link_idx = choose_worst_book_value_idx(links);
            return get_edax_child_board_after_move(board, links[worst_link_idx].policy, child_to_remove);
        }

        /*
            @brief remove boards from Edax export, so that every board has a leaf

            Edax's book needs a leaf (the best move which is not a link) on every board which has a move out of book.
            If a board has no leaf candidate, its worst link is removed from the export and becomes the leaf.

            The result depends on the order of the boards. In v7.8.3 the order was the order of the hash table,
            which changed at every run (the exported book was different each time).
            Now the order is fixed: boards with fewer discs first, then the order of segments, then the order of bits.
            Fewer discs first keeps more boards: a board removed by its parent is skipped, so it does not remove its own link.

            @return number of removed boards
        */
        inline int64_t remove_leaf_demoted_edax_export_boards_without_additional_calculation(bool *stop_saving) {
            int64_t n_removed = 0;
            bool changed = true;
            std::vector<Board> boards;
            while (changed) {
                changed = false;
                for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
                    std::cerr << "checking leaves for Edax export, discs " << n_discs << " removed " << n_removed << std::endl;
                    for (int segment_idx = 0; segment_idx < book.N_SEGMENTS; ++segment_idx) {
                        book.get_keys_in_segment_if(segment_idx, boards, [n_discs](const Board &board, const Book_elem &book_elem) {
                            return book_elem.exported && board.n_discs() == n_discs;
                        });
                        std::sort(boards.begin(), boards.end(), [](const Board &a, const Board &b) {
                            return a.player < b.player || (a.player == b.player && a.opponent < b.opponent);
                        });
                        for (const Board &board: boards) {
                            if (stop_saving && *stop_saving) {
                                return n_removed;
                            }
                            Book_elem book_elem;
                            if (!book.get(board, &book_elem) || !book_elem.exported) { // removed in this loop
                                continue;
                            }
                            Board child_to_remove;
                            if (find_leaf_demoted_edax_child_without_additional_calculation(board, book_elem, &child_to_remove)) {
                                bool removed = false;
                                book.modify(representative_board(child_to_remove), [&removed](Book_elem &book_elem) {
                                    removed = book_elem.exported;
                                    book_elem.exported = false;
                                });
                                if (removed) {
                                    ++n_removed;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
            return n_removed;
        }

        /*
            @brief get the boards to write to Edax-formatted book in a segment, in a fixed order

            @param only_exported        true: only the boards with exported flag
        */
        inline void get_edax_boards_to_write(int segment_idx, bool only_exported, std::vector<Board> &boards) {
            book.get_keys_in_segment_if(segment_idx, boards, [only_exported](const Board &board, const Book_elem &book_elem) {
                return !only_exported || book_elem.exported;
            });
            std::sort(boards.begin(), boards.end(), [](const Board &a, const Board &b) {
                return a.player < b.player || (a.player == b.player && a.opponent < b.opponent);
            });
        }

        /*
            @brief write a board to Edax-formatted book

            @param fout                 object which has write(const void*, size_t)
            @param board_to_write       representative board
            @return false if the board is not registered (nothing is written)
        */
        template<class Writer>
        inline bool write_edax_board(Writer &fout, const Board &board_to_write, bool additional_calculation, bool only_exported, int level, int edax_level, int leaf_recalculation_level) {
            Book_elem book_elem;
            if (!book.get(board_to_write, &book_elem)) {
                return false;
            }
            int n_win = 0, n_draw = 0, n_lose = 0;
            uint32_t n_lines;
            short short_val, short_val_min = -HW2, short_val_max = HW2;
            char char_level;
            char link_value, link_move;
            char leaf_val, leaf_move;
            char n_link;
            Board b;
            short_val = book_elem.value;
            //short_val_min = book_elem.value;
            //short_val_max = book_elem.value;
            b = board_to_write;
            uint64_t legal = b.get_legal();
            if (legal == 0ULL && !b.is_end()) {
                Board passed_board = b.copy();
                passed_board.pass();
                bool has_pass_child = contain_edax_export_board(passed_board, only_exported);
                Book_elem passed_elem;
                if (has_pass_child) {
                    passed_elem = get(passed_board);
                    n_lines = passed_elem.n_lines;
                    short_val = (short)passed_elem.value;
                } else {
                    n_lines = book_elem.n_lines;
                }
                if (!additional_calculation && level == LEVEL_UNDEFINED) {
                    char_level = has_pass_child ? passed_elem.level : book_elem.level;
                } else {
                    char_level = (char)edax_level;
                }
                if (char_level < 1) {
                    char_level = 1;
                }
                if (char_level > 60) {
                    char_level = 60;
                }
                if (has_pass_child) {
                    n_link = 1;
                    link_value = (char)passed_elem.value;
                    link_move = MOVE_PASS;
                    leaf_val = SCORE_UNDEFINED;
                    leaf_move = MOVE_NOMOVE;
                } else {
                    n_link = 0;
                    leaf_val = is_valid_score(short_val) ? (char)short_val : 0;
                    leaf_move = MOVE_PASS;
                }
                fout.write((char*)&board_to_write.player, 8);
                fout.write((char*)&board_to_write.opponent, 8);
                fout.write((char*)&n_win, 4);
                fout.write((char*)&n_draw, 4);
                fout.write((char*)&n_lose, 4);
                fout.write((char*)&n_lines, 4);
                fout.write((char*)&short_val, 2);
                fout.write((char*)&short_val_min, 2);
                fout.write((char*)&short_val_max, 2);
                fout.write((char*)&n_link, 1);
                fout.write((char*)&char_level, 1);
                if (has_pass_child) {
                    fout.write((char*)&link_value, 1);
                    fout.write((char*)&link_move, 1);
                }
                fout.write((char*)&leaf_val, 1);
                fout.write((char*)&leaf_move, 1);
                return true;
            }
            std::vector<Book_value> all_links, links;
            if (additional_calculation) {
                all_links = get_all_moves_with_value(&b);
                links = all_links;
            } else {
                get_all_moves_and_edax_links(&b, all_links, links);
            }
            auto has_policy = [](const std::vector<Book_value> &values, int policy) {
                for (const Book_value &value: values) {
                    if (value.policy == policy) {
                        return true;
                    }
                }
                return false;
            };
            auto choose_best_idx = [](const std::vector<Book_value> &values) {
                int best_idx = 0;
                for (int i = 1; i < (int)values.size(); ++i) {
                    if (values[i].value > values[best_idx].value) {
                        best_idx = i;
                    }
                }
                return best_idx;
            };
            auto choose_worst_idx = [](const std::vector<Book_value> &values) {
                int worst_idx = 0;
                for (int i = 1; i < (int)values.size(); ++i) {
                    if (values[i].value < values[worst_idx].value) {
                        worst_idx = i;
                    }
                }
                return worst_idx;
            };
            auto add_leaf_candidate = [](std::vector<Book_value> &leaf_candidates, Book_value candidate) {
                for (Book_value &leaf_candidate: leaf_candidates) {
                    if (leaf_candidate.policy == candidate.policy) {
                        if (candidate.value > leaf_candidate.value) {
                            leaf_candidate.value = candidate.value;
                        }
                        return;
                    }
                }
                leaf_candidates.emplace_back(candidate);
            };
            std::vector<Book_value> leaf_candidates;
            leaf_val = SCORE_UNDEFINED;
            leaf_move = MOVE_NOMOVE;
            bool registered_leaf_ok = is_valid_score(book_elem.leaf.value) && is_valid_policy(book_elem.leaf.move) && ((legal & (1ULL << book_elem.leaf.move)) != 0ULL) && !has_policy(links, book_elem.leaf.move);
            if (registered_leaf_ok) {
                Book_value leaf_candidate;
                leaf_candidate.policy = book_elem.leaf.move;
                leaf_candidate.value = book_elem.leaf.value;
                add_leaf_candidate(leaf_candidates, leaf_candidate);
            }
            if (!additional_calculation) {
                for (const Book_value &link: all_links) {
                    if (!has_policy(links, link.policy)) {
                        add_leaf_candidate(leaf_candidates, link);
                    }
                }
            }
            int mobility = pop_count_ull(legal);
            if ((int)links.size() != mobility) {
                if (!leaf_candidates.empty()) {
                    int leaf_idx = choose_best_idx(leaf_candidates);
                    leaf_val = (char)leaf_candidates[leaf_idx].value;
                    leaf_move = (char)leaf_candidates[leaf_idx].policy;
                } else if (additional_calculation) {
                    uint64_t missing = legal;
                    for (const Book_value &link: links) {
                        missing &= ~(1ULL << link.policy);
                    }
                    if (missing) {
                        int use_level = leaf_recalculation_level;
                        if (use_level == ADD_LEAF_SPECIAL_LEVEL) {
                            use_level = ADD_LEAF_SPECIAL_LEVEL_SEARCH_LEVEL;
                        } else {
                            use_level = std::clamp(use_level, 1, MAX_LEVEL);
                        }
                        Search_result ai_result = ai_legal(b, use_level, false, 0, true, false, missing);
                        if (ai_result.value != SCORE_UNDEFINED && is_valid_policy(ai_result.policy)) {
                            leaf_val = (char)ai_result.value;
                            if (leaf_recalculation_level == ADD_LEAF_SPECIAL_LEVEL) {
                                leaf_val = is_valid_score(book_elem.value) ? (char)book_elem.value : (char)ai_result.value;
                            }
                            leaf_move = (char)ai_result.policy;
                        } else if (!links.empty()) {
                            int worst_link_idx = choose_worst_idx(links);
                            leaf_val = (char)links[worst_link_idx].value;
                            leaf_move = (char)links[worst_link_idx].policy;
                            links.erase(links.begin() + worst_link_idx);
                        }
                    }
                } else if (!links.empty()) {
                    int worst_link_idx = choose_worst_idx(links);
                    leaf_val = (char)links[worst_link_idx].value;
                    leaf_move = (char)links[worst_link_idx].policy;
                    links.erase(links.begin() + worst_link_idx);
                }
            }
            n_link = (char)links.size();
            n_lines = book_elem.n_lines;
            if (!additional_calculation && level == LEVEL_UNDEFINED) {
                char_level = book_elem.level;
            } else {
                char_level = (char)edax_level;
            }
            if (char_level < 1) {
                char_level = 1;
            }
            if (char_level > 60) {
                char_level = 60;
            }
            fout.write((char*)&board_to_write.player, 8);
            fout.write((char*)&board_to_write.opponent, 8);
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
            return true;
        }

        /*
            @brief save as Edax-formatted book (.dat)

            @param file                 file name to save
            @param bak_file             backup file name
        */
        inline void save_bin_edax(std::string file, int level, bool additional_calculation = true, bool *stop_saving = nullptr, int leaf_recalculation_level = ADD_LEAF_SPECIAL_LEVEL) {
            if (additional_calculation) {
                bool stop = false;
                bool *stop_ptr = stop_saving ? stop_saving : &stop;
                int edax_leaf_recalculation_level = leaf_recalculation_level;
                if (edax_leaf_recalculation_level != ADD_LEAF_SPECIAL_LEVEL) {
                    edax_leaf_recalculation_level = std::clamp(edax_leaf_recalculation_level, 1, MAX_LEVEL);
                }
                check_add_leaf_all_search(edax_leaf_recalculation_level, stop_ptr);
                if (stop_saving && *stop_saving) {
                    return;
                }
            }
            // Boards to write are marked with the exported flag of each element
            // (a set of all boards needs more memory than the book itself with a big book).
            std::unordered_set<Board, Book_hash> pass_boards;
            Board root_board;
            root_board.reset();
            const bool only_exported = !additional_calculation;
            if (only_exported) {
                parallel_for_segments([this, stop_saving](int segment_idx) {
                    book.for_each_in_segment(segment_idx, [this, stop_saving](const Board &board, Book_elem &book_elem) {
                        if (stop_saving && *stop_saving) {
                            return;
                        }
                        bool exported = should_write_edax_position_without_additional_calculation(board);
                        book.modify(board, [exported](Book_elem &registered) {
                            registered.exported = exported;
                        });
                    });
                });
                if (stop_saving && *stop_saving) {
                    return;
                }
                int64_t n_leaf_demoted_removed = remove_leaf_demoted_edax_export_boards_without_additional_calculation(stop_saving);
                if (stop_saving && *stop_saving) {
                    return;
                }
                if (n_leaf_demoted_removed) {
                    std::cerr << "removed " << n_leaf_demoted_removed << " Edax leaf-demoted boards from export" << std::endl;
                }
            }
            std::cerr << "pass board calculating..." << std::endl;
            reset_seen();
            get_pass_boards(root_board, pass_boards, additional_calculation, only_exported, stop_saving);
            reset_seen();
            if (stop_saving && *stop_saving) {
                return;
            }
            // number of boards and the fewest empty squares, for the header
            uint64_t n_boards_to_write = 0;
            int n_empties = HW2;
            int max_book_level = 0;
            book.for_each([&n_boards_to_write, &n_empties, &max_book_level, only_exported](const Board &board, Book_elem &book_elem) {
                max_book_level = std::max(max_book_level, (int)book_elem.level);
                if (!only_exported || book_elem.exported) {
                    ++n_boards_to_write;
                    n_empties = std::min(n_empties, HW2 + 1 - board.n_discs());
                }
            });
            std::cerr << "pass board calculated " << pass_boards.size() << std::endl;
            Book_file_writer fout;
            if (!fout.open(file)) {
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
            int edax_level = level;
            if (edax_level < 1) {
                edax_level = std::max(edax_level, max_book_level);
                if (edax_level < 1) {
                    edax_level = 1;
                }
            }
            int header_level = edax_level;
            fout.write((char*)&header_level, 4);
            for (const Board &board: pass_boards) {
                n_empties = std::min(n_empties, HW2 + 1 - board.n_discs());
            }
            fout.write((char*)&n_empties, 4);
            int err_mid = 0;
            fout.write((char*)&err_mid, 4);
            int err_end = 0;
            fout.write((char*)&err_end, 4);
            int verb = 0;
            fout.write((char*)&verb, 4);
            int n_position = (int)(n_boards_to_write + pass_boards.size());
            fout.write((char*)&n_position, 4);
            int n_win = 0, n_draw = 0, n_lose = 0;
            uint32_t n_lines;
            short short_val, short_val_min = -HW2, short_val_max = HW2;
            char char_level = (char)edax_level;
            Book_elem book_elem;
            char link_value, link_move;
            char leaf_val, leaf_move;
            char n_link;
            Board b;
            int percent = -1;
            uint64_t n_boards = n_boards_to_write;
            uint64_t t = 0;
            // written in a fixed order (the order in the set depends on the order of registration)
            std::vector<Board> sorted_pass_boards(pass_boards.begin(), pass_boards.end());
            std::sort(sorted_pass_boards.begin(), sorted_pass_boards.end(), [](const Board &a, const Board &b) {
                return a.player < b.player || (a.player == b.player && a.opponent < b.opponent);
            });
            for (Board pass_board: sorted_pass_boards) {
                if (stop_saving && *stop_saving) {
                    break;
                }
                Board passed_board = pass_board.copy();
                passed_board.pass();
                Book_elem passed_elem = get(passed_board);
                n_lines = passed_elem.n_lines;
                short_val = (short)passed_elem.value;
                if (!additional_calculation && level == LEVEL_UNDEFINED) {
                    b = pass_board.copy();
                    b.pass();
                    if (contain(b)) {
                        char_level = get(b).level;
                    } else {
                        char_level = 1;
                    }
                } else {
                    char_level = (char)edax_level;
                }
                if (char_level < 1) {
                    char_level = 1;
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
            // Boards are written segment by segment in a fixed order (no array of all boards).
            // Without additional calculation, each record depends only on the book,
            // so the records of several segments are built in parallel and written in order.
            // With additional calculation, a search may run for a record (the search uses the thread pool): one by one.
            struct Edax_record_buffer {
                std::vector<char> data;
                inline void write(const void *src, size_t size) {
                    data.insert(data.end(), (const char*)src, (const char*)src + size);
                }
            };
            const int n_segments_per_block = only_exported ? std::max(1, (thread_pool.size() + 1) * 2) : 1;
            for (int block_start = 0; block_start < book.N_SEGMENTS; block_start += n_segments_per_block) {
                if (stop_saving && *stop_saving) {
                    break;
                }
                int n_percent = n_boards == 0 ? 100 : (int)std::min(100.0, (double)t / n_boards * 100);
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "converting book " << percent << "%" << std::endl;
                }
                const int n_segments_in_block = std::min(n_segments_per_block, book.N_SEGMENTS - block_start);
                if (only_exported) {
                    std::vector<Edax_record_buffer> buffers(n_segments_in_block);
                    std::vector<uint64_t> n_written(n_segments_in_block, 0);
                    parallel_for(n_segments_in_block, [&](int i) {
                        std::vector<Board> boards_to_write;
                        get_edax_boards_to_write(block_start + i, only_exported, boards_to_write);
                        for (const Board &board_to_write: boards_to_write) {
                            if (stop_saving && *stop_saving) {
                                break;
                            }
                            n_written[i] += write_edax_board(buffers[i], board_to_write, additional_calculation, only_exported, level, edax_level, leaf_recalculation_level);
                        }
                    });
                    for (int i = 0; i < n_segments_in_block; ++i) {
                        fout.write(buffers[i].data.data(), buffers[i].data.size());
                        t += n_written[i];
                    }
                } else {
                    std::vector<Board> boards_to_write;
                    get_edax_boards_to_write(block_start, only_exported, boards_to_write);
                    for (const Board &board_to_write: boards_to_write) {
                        if (stop_saving && *stop_saving) {
                            break;
                        }
                        t += write_edax_board(fout, board_to_write, additional_calculation, only_exported, level, edax_level, leaf_recalculation_level);
                    }
                }
            }
            bool saved = fout.close();
            if (stop_saving && *stop_saving) {
                remove(file.c_str());
                std::cerr << "book saving stopped" << std::endl;
                return;
            }
            if (!saved) {
                remove(file.c_str());
                std::cerr << "[ERROR] failed to save book " << file << std::endl;
                return;
            }
            std::cerr << "saved " << t << " boards as a edax-formatted book " << n_position << " " << n_boards_to_write << std::endl;
        }

        inline void save_bin_edax_without_additional_calculation(std::string file, int level) {
            save_bin_edax(file, level, false);
        }

        inline void save_bin_edax_without_additional_calculation(std::string file, int level, bool *stop_saving) {
            save_bin_edax(file, level, false, stop_saving);
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
            return book.contains(b);
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
            @brief check if a board is written to Edax-formatted book

            @param only_exported        true: registered boards with exported flag, false: all registered boards
        */
        inline bool contain_edax_export_board(Board b, bool only_exported) {
            Book_elem book_elem;
            if (!book.get(representative_board(b), &book_elem)) {
                return false;
            }
            return !only_exported || book_elem.exported;
        }

        inline bool contain_edax_export_board(Board *b, bool only_exported) {
            return contain_edax_export_board(b->copy(), only_exported);
        }

        inline bool should_write_edax_position_without_additional_calculation(Board board) {
            board = representative_board(board);
            Book_elem elem;
            if (!book.get(board, &elem)) {
                return false;
            }
            uint64_t legal = board.get_legal();
            if (is_valid_score(elem.leaf.value) && is_valid_policy(elem.leaf.move) && ((legal & (1ULL << elem.leaf.move)) != 0ULL)) {
                return true;
            }
            if (legal == 0ULL && !board.is_end()) {
                Board passed_board = board.copy();
                passed_board.pass();
                return contain(passed_board);
            }
            return !get_all_moves_with_value(&board).empty();
        }

        /*
            @brief get registered score

            @param b                    a board to find
            @return registered value (if not registered, returns -INF)
        */
        inline Book_elem get_representative(Board b, int idx) {
            Book_elem res;
            if (!book.get(b, &res)) {
                return res;
            }
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
            @brief get registered score with all rotation

            Same as contain() + get().value with one search.

            @param b                    a board to find
            @param value                registered value is stored if the board is registered
            @return registered?
        */
        inline bool get_value(Board b, int *value) {
            Book_elem elem;
            if (!book.get(representative_board(b), &elem)) {
                return false;
            }
            *value = elem.value;
            return true;
        }

        /*
            @brief get the value of the move just played

            If the game is over after the move, both of the board and passed board are checked.
            If the next player must pass, passed board is checked.

            @param b                    a board pointer after the move (not modified)
            @param value                value for the player who played the move is stored if registered
            @return registered?
        */
        inline bool get_moved_board_value(Board *b, int *value) {
            int registered_value;
            if (b->get_legal() == 0) {
                if (calc_legal(b->opponent, b->player) == 0) { // game over
                    if (get_value(*b, &registered_value)) {
                        *value = -registered_value;
                        return true;
                    }
                }
                // pass
                Board passed_board(b->opponent, b->player);
                if (get_value(passed_board, &registered_value)) {
                    *value = registered_value;
                    return true;
                }
                return false;
            }
            if (get_value(*b, &registered_value)) {
                *value = -registered_value;
                return true;
            }
            return false;
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
                    int value;
                    if (get_value(*b, &value)) {
                        if (sgn * value > max_value) {
                            max_value = sgn * value;
                            policies.clear();
                        }
                        if (sgn * value == max_value) {
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
            @brief board after a move, to be searched in book

            Same rule as get_moved_board_value():
            if the game is over after the move, the board is searched first and the passed board next.
            If the next player must pass, the passed board is searched.

            @param board                representative board to search first
            @param hash                 hash code of board
            @param end_board            representative board to search next (game over only)
            @param end_hash             hash code of end_board
            @param policy               the move
            @param passed               the next player has no legal move
            @param is_end               game over
        */
        struct Moved_board {
            Board board;
            uint64_t hash;
            Board end_board;
            uint64_t end_hash;
            int policy;
            bool passed;
            bool is_end;
        };

        /*
            @brief calculate the boards after all legal moves and load their memory into the cache

            A big book is much larger than the cache. Searching the boards one by one waits for the memory each time,
            so all boards are calculated and prefetched first.

            @param b                    a board pointer (not modified)
            @param moved_boards         array to store the result (HW2 elements)
            @param unlocked             use functions without lock (only while no thread registers / deletes boards)
            @return number of legal moves
        */
        inline int get_moved_boards(const Board *b, Moved_board moved_boards[], bool unlocked = false) {
            int n_moved_boards = 0;
            uint64_t legal = b->get_legal();
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                flip.calc_flip(b->player, b->opponent, cell);
                Board moved_board(b->opponent ^ flip.flip, b->player ^ flip.flip ^ (1ULL << cell));
                Moved_board &elem = moved_boards[n_moved_boards++];
                elem.policy = cell;
                elem.passed = moved_board.get_legal() == 0;
                elem.is_end = elem.passed && calc_legal(moved_board.opponent, moved_board.player) == 0;
                if (elem.is_end) {
                    elem.board = representative_board(moved_board);
                    moved_board.pass();
                    elem.end_board = representative_board(moved_board);
                    elem.end_hash = book_hash_board(elem.end_board.player, elem.end_board.opponent);
                    if (unlocked) {
                        book.prefetch_unlocked(elem.end_hash);
                    } else {
                        book.prefetch(elem.end_hash);
                    }
                } else {
                    if (elem.passed) {
                        moved_board.pass();
                    }
                    elem.board = representative_board(moved_board);
                }
                elem.hash = book_hash_board(elem.board.player, elem.board.opponent);
                if (unlocked) {
                    book.prefetch_unlocked(elem.hash);
                } else {
                    book.prefetch(elem.hash);
                }
            }
            for (int i = 0; i < n_moved_boards; ++i) {
                if (unlocked) {
                    book.prefetch_entry_unlocked(moved_boards[i].hash);
                    if (moved_boards[i].is_end) {
                        book.prefetch_entry_unlocked(moved_boards[i].end_hash);
                    }
                } else {
                    book.prefetch_entry(moved_boards[i].hash);
                    if (moved_boards[i].is_end) {
                        book.prefetch_entry(moved_boards[i].end_hash);
                    }
                }
            }
            return n_moved_boards;
        }

        /*
            @brief get all registered moves with value

            @param b                    a board pointer to find
            @return vector of moves
        */
        inline std::vector<Book_value> get_all_moves_with_value(Board *b) {
            Moved_board moved_boards[HW2];
            int n_moved_boards = get_moved_boards(b, moved_boards);
            std::vector<Book_value> policies;
            Book_elem book_elem;
            for (int i = 0; i < n_moved_boards; ++i) {
                const Moved_board &elem = moved_boards[i];
                Book_value book_value;
                book_value.policy = elem.policy;
                if (book.get(elem.board, elem.hash, &book_elem)) {
                    book_value.value = (elem.passed && !elem.is_end) ? book_elem.value : -book_elem.value;
                    policies.emplace_back(book_value);
                } else if (elem.is_end && book.get(elem.end_board, elem.end_hash, &book_elem)) {
                    book_value.value = book_elem.value;
                    policies.emplace_back(book_value);
                }
            }
            return policies;
        }

        /*
            @brief get all registered moves with value, and the moves written as links to Edax-formatted book

            @param b                    a board pointer to find
            @param all_links            same as get_all_moves_with_value()
            @param edax_links           moves of all_links whose board has the exported flag
        */
        inline void get_all_moves_and_edax_links(Board *b, std::vector<Book_value> &all_links, std::vector<Book_value> &edax_links) {
            Moved_board moved_boards[HW2];
            int n_moved_boards = get_moved_boards(b, moved_boards);
            Book_elem book_elem;
            for (int i = 0; i < n_moved_boards; ++i) {
                const Moved_board &elem = moved_boards[i];
                Book_value book_value;
                book_value.policy = elem.policy;
                if (book.get(elem.board, elem.hash, &book_elem)) {
                    book_value.value = (elem.passed && !elem.is_end) ? book_elem.value : -book_elem.value;
                } else if (elem.is_end && book.get(elem.end_board, elem.end_hash, &book_elem)) {
                    book_value.value = book_elem.value;
                } else {
                    continue;
                }
                all_links.emplace_back(book_value);
                if (book_elem.exported) {
                    edax_links.emplace_back(book_value);
                }
            }
        }

        /*
            @brief get the moves written as links to Edax-formatted book

            @param only_exported        true: moves to the boards with exported flag, false: moves to the boards which have a registered move
        */
        inline std::vector<Book_value> get_edax_links(Board *b, bool only_exported = false) {
            std::vector<Book_value> all_links, policies;
            if (only_exported) {
                get_all_moves_and_edax_links(b, all_links, policies);
                return policies;
            }
            uint64_t legal = b->get_legal();
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    Book_value book_value;
                    book_value.policy = cell;
                    int registered_value;
                    bool found = false;
                    bool passed = false;
                    if (b->get_legal() == 0) {
                        if (b->is_end() && get_value(*b, &registered_value)) { // game over
                            book_value.value = -registered_value;
                            found = true;
                        } else { // pass
                            b->pass();
                            passed = true;
                            if (get_value(*b, &registered_value)) {
                                book_value.value = registered_value;
                                found = true;
                            }
                        }
                    } else if (get_value(*b, &registered_value)) { // normal move
                        book_value.value = -registered_value;
                        found = true;
                    }
                    if (found && get_all_moves_with_value(b).size() > 0) {
                        policies.emplace_back(book_value);
                    }
                    if (passed) {
                        b->pass();
                    }
                b->undo_board(&flip);
            }
            return policies;
        }

        inline Leaf get_edax_leaf(Board *b, std::vector<Book_value> &edax_links) {
            Leaf leaf;
            leaf.value = -127;
            uint64_t legal = b->get_legal();
            for (Book_value &link: edax_links) {
                legal ^= 1ULL << link.policy;
            }
            Flip flip;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                calc_flip(&flip, b, cell);
                b->move_board(&flip);
                    int v;
                    if (get_moved_board_value(b, &v)) {
                        if (leaf.value < v) {
                            leaf.value = v;
                            leaf.move = cell;
                        }
                    }
                b->undo_board(&flip);
            }
            return leaf;
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
                        int registered_value;
                        if (get_value(*b, &registered_value)) {
                            Book_value book_value;
                            book_value.policy = cell;
                            book_value.value = sgn * registered_value;
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
                    int registered_value;
                    if (get_value(*b, &registered_value)) {
                        Book_value book_value;
                        book_value.policy = cell;
                        book_value.value = sgn * registered_value;
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
            @brief change the value of a registered board

            @param b                    a board to change
            @param value                a value to change
            @return registered?
        */
        inline bool change_registered_value(Board b, int value, int level) {
            return book.modify(representative_board(b), [value, level](Book_elem &elem) {
                elem.value = value;
                elem.level = level;
            });
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
                    if (!change_registered_value(b, value, level)) {
                        b.pass();
                        if (!change_registered_value(b, -value, level)) {
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
                    if (!change_registered_value(b, value, level)) {
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

        inline void change(Board b, int value, int level, int leaf_move, int leaf_value, int leaf_level) {
            std::lock_guard<std::mutex> lock(mtx);
            if (-HW2 <= value && value <= HW2) {
                if (b.is_end()) { // game over
                    if (!change_registered_value(b, value, level)) {
                        b.pass();
                        if (!change_registered_value(b, -value, level)) {
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
                    int idx;
                    Board bb = representative_board(b, &idx);
                    int rotated_leaf_move = leaf_move;
                    if (is_valid_policy(leaf_move)) {
                        rotated_leaf_move = convert_coord_to_representative_board(leaf_move, idx);
                    }
                    bool registered = book.modify(bb, [value, level, rotated_leaf_move, leaf_value, leaf_level](Book_elem &elem) {
                        elem.value = value;
                        elem.level = level;
                        elem.leaf.move = rotated_leaf_move;
                        elem.leaf.value = leaf_value;
                        elem.leaf.level = leaf_level;
                    });
                    if (!registered) {
                        Book_elem elem;
                        elem.value = value;
                        elem.level = level;
                        elem.leaf.move = leaf_move;
                        elem.leaf.value = leaf_value;
                        elem.leaf.level = leaf_level;
                        register_symmetric_book(b, elem);
                    }
                }
            }
        }

        /*
            @brief delete a board

            @param b                    a board to delete
        */
        inline void delete_elem(Board b) {
            delete_symmetric_book(b);
        }

        /*
            @brief delete all board in this book
        */
        inline void delete_all() {
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
        inline void fix(bool *stop) {
            fix(false, stop);
        }

        /*
            @brief fix book
        */
        inline void fix(bool edax_compliant) {
            bool stop = false;
            fix(edax_compliant, &stop);
        }

        /*
            @brief fix book
        */
        inline void fix() {
            fix(false);
        }

        /*
            @brief call func(segment index) for all segments of the book, using the thread pool

            func is called from several threads at the same time.
        */
        template<class F>
        inline void parallel_for_segments(F &&func) {
            parallel_for(Board_table<Book_elem>::N_SEGMENTS, func);
        }

        /*
            @brief call func(i) for i = 0, 1, ..., n - 1, using the thread pool

            func is called from several threads at the same time.
        */
        template<class F>
        inline void parallel_for(int n, F &&func) {
            std::atomic<int> next_idx(0);
            auto worker = [&next_idx, &func, n]() {
                for (;;) {
                    int idx = next_idx.fetch_add(1);
                    if (idx >= n) {
                        break;
                    }
                    func(idx);
                }
            };
            std::vector<std::future<void>> tasks;
            const int n_workers = thread_pool.size();
            for (int i = 0; i < n_workers; ++i) {
                bool pushed = false;
                std::future<void> task = thread_pool.push(&pushed, worker);
                if (pushed) {
                    tasks.emplace_back(std::move(task));
                }
            }
            worker();
            for (std::future<void> &task: tasks) {
                task.get();
            }
        }

        /*
            @brief call func(const Board&, Book_elem&) for all boards with the seen flag and the given number of discs

            Boards must not be registered / deleted in func. func is called from several threads at the same time.
        */
        template<class F>
        inline void parallel_for_each_seen_board(int n_discs, F &&func) {
            parallel_for_segments([this, n_discs, &func](int segment_idx) {
                book.for_each_in_segment(segment_idx, [n_discs, &func](const Board &board, Book_elem &book_elem) {
                    if (board.n_discs() == n_discs && book_elem.seen) {
                        func(board, book_elem);
                    }
                });
            });
        }

        /*
            @brief flag (seen) all boards which can be reached from the root board with registered moves

            Boards with fewer discs are processed first, so that boards with the same number of discs can be processed in parallel.
            Game-over boards are not flagged.
            Boards must not be registered / deleted by other threads (functions without lock are used).

            @param root_board           root board
            @param expand_invalid       expand the boards whose value is out of range?
            @param stop                 flag to stop
            @return the largest number of discs in the flagged boards (-1: nothing flagged)
        */
        inline int flag_reachable_boards(Board root_board, bool expand_invalid, bool *stop) {
            if (root_board.get_legal() == 0) {
                root_board.pass();
                if (root_board.get_legal() == 0) { // game over
                    return -1;
                }
            }
            bool registered = book.modify(representative_board(root_board), [](Book_elem &book_elem) {
                book_elem.seen = true;
            });
            if (!registered) {
                return -1;
            }
            int max_n_discs = root_board.n_discs();
            for (int n_discs = root_board.n_discs(); n_discs < HW2 && !(*stop); ++n_discs) {
                std::atomic<bool> flagged(false);
                parallel_for_each_seen_board(n_discs, [this, expand_invalid, stop, &flagged](const Board &board, Book_elem &book_elem) {
                    if (*stop || (!expand_invalid && (book_elem.value < -HW2 || HW2 < book_elem.value))) {
                        return;
                    }
                    Moved_board moved_boards[HW2];
                    int n_moved_boards = get_moved_boards(&board, moved_boards, true);
                    bool flagged_local = false;
                    for (int i = 0; i < n_moved_boards; ++i) {
                        if (!moved_boards[i].is_end) {
                            flagged_local |= book.modify_unlocked(moved_boards[i].board, moved_boards[i].hash, [](Book_elem &child_elem) {
                                if (!child_elem.seen) { // several threads may flag the same board
                                    child_elem.seen = true;
                                }
                            });
                        }
                    }
                    if (flagged_local && !flagged.load(std::memory_order_relaxed)) {
                        flagged.store(true, std::memory_order_relaxed);
                    }
                });
                if (!flagged.load()) {
                    break;
                }
                max_n_discs = n_discs + 1;
            }
            return max_n_discs;
        }

        /*
            @brief negamax the values of the boards which can be reached from the initial board

            The value of a board is the best value of the registered moves, and n_lines is recalculated.
            Boards with more discs are processed first: when a board is processed, the boards after its moves are already fixed.
            Boards with the same number of discs are processed in parallel.
            If stopped, the boards not processed yet keep their values.

            @param edax_compliant       use the leaf value as a candidate of the value
            @param stop                 flag to stop
        */
        void negamax_book(bool edax_compliant, bool *stop) {
            Board root_board;
            root_board.reset();
            std::atomic<int64_t> n_seen(0), n_fix(0);
            reset_seen();
            int max_n_discs = flag_reachable_boards(root_board, false, stop);
            uint64_t n_boards = book.size();
            for (int n_discs = max_n_discs; n_discs >= 4 && !(*stop); --n_discs) {
                parallel_for_segments([this, n_discs, edax_compliant, stop, &n_seen, &n_fix](int segment_idx) {
                  int64_t n_seen_local = 0, n_fix_local = 0;
                  book.for_each_in_segment(segment_idx, [this, n_discs, edax_compliant, stop, &n_seen_local, &n_fix_local](const Board &board, Book_elem &book_elem) {
                    if (board.n_discs() != n_discs || !book_elem.seen) {
                        return;
                    }
                    if (*stop || book_elem.value < -HW2 || HW2 < book_elem.value) {
                        return;
                    }
                    ++n_seen_local;
                    Moved_board moved_boards[HW2];
                    int n_moved_boards = get_moved_boards(&board, moved_boards, true);
                    uint64_t n_lines = 1;
                    int v = -INF;
                    if (edax_compliant && is_valid_score(book_elem.leaf.value)) {
                        v = book_elem.leaf.value;
                    }
                    Book_elem child_elem;
                    for (int i = 0; i < n_moved_boards; ++i) {
                        const Moved_board &moved_board = moved_boards[i];
                        int child_value = SCORE_UNDEFINED; // value for the player of this board
                        if (moved_board.is_end) { // game over
                            if (book.get_unlocked(moved_board.board, moved_board.hash, &child_elem)) {
                                if (child_elem.value != SCORE_UNDEFINED) {
                                    child_value = -child_elem.value;
                                }
                            } else if (book.get_unlocked(moved_board.end_board, moved_board.end_hash, &child_elem)) {
                                child_value = child_elem.value;
                            }
                        } else if (book.get_unlocked(moved_board.board, moved_board.hash, &child_elem)) {
                            if (-HW2 <= child_elem.value && child_elem.value <= HW2) {
                                child_value = moved_board.passed ? child_elem.value : -child_elem.value;
                            }
                        }
                        if (child_value != SCORE_UNDEFINED) {
                            if (v < child_value) { // update parent value
                                v = child_value;
                            }
                            n_lines += child_elem.n_lines;
                        }
                    }
                    // boards with this number of discs are not read by other threads now
                    if (v != -INF && v != book_elem.value) {
                        book_elem.value = v;
                        ++n_fix_local;
                    }
                    book_elem.n_lines = (uint32_t)std::min((uint64_t)MAX_N_LINES, n_lines);
                  });
                  n_seen.fetch_add(n_seen_local, std::memory_order_relaxed);
                  n_fix.fetch_add(n_fix_local, std::memory_order_relaxed);
                });
                int percent = n_boards ? (int)std::min<uint64_t>(100, (uint64_t)n_seen.load() * 100 / n_boards) : 100;
                std::cerr << "negamaxing book... " << percent << "%" << " fixed " << n_fix.load() << std::endl;
            }
            reset_seen();
            std::cerr << "negamaxed book fixed " << n_fix.load() << " boards seen " << n_seen.load() << " boards size " << book.size() << std::endl;
        }

        // flag keeped boards
        // void reduce_book_calculate_keeplist(Board board, int max_depth, int max_error_per_move, int remaining_error, uint64_t *n_flags, std::unordered_map<Board, int, Book_hash> &keep_list, bool *doing) {
        //     if (!*(doing)) {
        //         return;
        //     }
        //     // depth threshold
        //     if (board.n_discs() > 4 + max_depth) {
        //         return;
        //     }
        //     // pass
        //     if (board.get_legal() == 0) { // just pass
        //         board.pass();
        //         if (board.get_legal() == 0) { // game over
        //             if (keep_list.find(representative_board(board)) != keep_list.end()) {
        //                 return;
        //             }
        //             if (contain(&board)) {
        //                 Book_elem book_elem = get(board);
        //                 if (book_elem.seen) {
        //                     return;
        //                 }
        //             } else {
        //                 board.pass();
        //                 if (keep_list.find(representative_board(board)) != keep_list.end()) {
        //                     return;
        //                 }
        //                 Book_elem book_elem = get(board);
        //                 if (book_elem.seen) {
        //                     return;
        //                 }
        //             }
        //             keep_list[representative_board(board)] = 0;
        //             ++(*n_flags);
        //             return;
        //         }
        //     }
        //     Board unique_board = representative_board(board);
        //     // already seen
        //     int pre_searched_remaining_error = -1;
        //     bool keep_list_found = false;
        //     if (keep_list.find(unique_board) != keep_list.end()) {
        //         keep_list_found = true;
        //         pre_searched_remaining_error = keep_list[unique_board];
        //         if (remaining_error <= pre_searched_remaining_error) {
        //             return;
        //         }
        //     }
        //     // not in book
        //     if (!contain(&board)) {
        //         return;
        //     }
        //     Book_elem book_elem = get(board);
        //     keep_list[unique_board] = remaining_error; // update remaining error
        //     if (!keep_list_found) {
        //         ++(*n_flags);
        //         if ((*n_flags) % 1000 == 0) {
        //             std::cerr << "keep " << (*n_flags) << " boards of " << book.size() << std::endl;
        //         }
        //     }
        //     std::vector<Book_value> links = get_all_moves_with_value(&board);
        //     Flip flip;
        //     for (Book_value &link: links) {
        //         int link_error = book_elem.value - link.value;
        //         if (link_error <= max_error_per_move && link_error <= remaining_error) {
        //             calc_flip(&flip, &board, link.policy);
        //             board.move_board(&flip);
        //                 reduce_book_calculate_keeplist(board, max_depth, max_error_per_move, remaining_error - std::max(0, link_error), n_flags, keep_list, doing);
        //             board.undo_board(&flip);
        //         }
        //     }
        // }

        std::unordered_map<Board, int, Book_hash> reduce_book_calculate_keeplist_bfs(const int depth, std::unordered_map<Board, int, Book_hash> &root_list, int max_error_per_move, uint64_t *n_flags, std::unordered_map<Board, int, Book_hash> &keep_list, bool *doing) {
            std::unordered_map<Board, int, Book_hash> next_root_list;
            if (!(*doing)) {
                return next_root_list;
            }
            size_t root_list_size = root_list.size();
            size_t entry_count = 0;
            int last_percent = -1;
            for (const auto &entry: root_list) {
                if (!(*doing)) {
                    break;
                }
                int percent = entry_count * 100 / root_list_size;
                if (percent > last_percent) {
                    last_percent = percent;
                    std::cerr << "depth " << depth << " " << percent << "% " << "keep_list " << keep_list.size() + next_root_list.size() << std::endl;
                }
                ++entry_count;
                Board board = entry.first.copy();
                int remaining_error = entry.second;
                Book_elem book_elem = get(board);
                std::vector<Book_value> links = get_all_moves_with_value(&board);
                Flip flip;
                for (Book_value &link: links) {
                    if (!(*doing)) {
                        break;
                    }
                    int link_error = book_elem.value - link.value;
                    if (link_error <= max_error_per_move && link_error <= remaining_error) {
                        calc_flip(&flip, &board, link.policy);
                        board.move_board(&flip);
                            Board n_board = board.copy();
                            if (n_board.get_legal() == 0 && !n_board.is_end()) { // pass: the passed board is registered in book
                                n_board.pass();
                            }
                            Board n_unique_board = representative_board(n_board);
                            int new_remaining_error = remaining_error - link_error;
                            // auto keep_itr = keep_list.find(n_unique_board);
                            // bool is_new_board = (keep_itr == keep_list.end());
                            // if (!is_new_board && keep_itr->second >= new_remaining_error) {
                            //     // No improvement over existing keep_list entry
                            // } else {
                            auto next_itr = next_root_list.find(n_unique_board);
                            bool inserted = false;
                            if (next_itr == next_root_list.end()) {
                                next_root_list.emplace(n_unique_board, new_remaining_error);
                                inserted = true;
                            } else if (next_itr->second < new_remaining_error) {
                                next_itr->second = new_remaining_error;
                            }
                            // if (is_new_board && inserted) {
                            if (inserted) {
                                ++(*n_flags);
                                // if ((*n_flags) % 1000 == 0) {
                                //     std::cerr << "keep " << (*n_flags) << " boards of " << book.size() << std::endl;
                                // }
                            }
                            // }
                        board.undo_board(&flip);
                    }
                }
            }
            return next_root_list;
        }

        void reduce_book_update_leaves(Board board, std::unordered_map<Board, int, Book_hash> &keep_list, bool *doing) {
            if (!(*doing)) {
                return;
            }
            if (board.get_legal() == 0) { // the passed board is registered in book
                board.pass();
                if (board.get_legal() == 0) {
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
            book_elem.seen = true; // keep the flag when this element is registered again
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
                        if (is_end) { // kept if either of the board and the passed board is in the keep list
                            will_be_deleted = keep_list.find(representative_board(board)) == keep_list.end();
                            board.pass();
                                will_be_deleted &= keep_list.find(representative_board(board)) == keep_list.end();
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
                            reduce_book_update_leaves(board, keep_list, doing);
                        }
                    board.undo_board(&flip);
                }
                if (leaf_updated) {
                    reg(&board, book_elem);
                }
            }
        }

        void reduce_book_delete_unflagged_moves(Board board, uint64_t *n_delete, std::unordered_map<Board, int, Book_hash> &keep_list, bool *doing) {
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
                    reduce_book_delete_unflagged_moves(board, n_delete, keep_list, doing);
                board.undo_board(&flip);
            }
            if (keep_list.find(representative_board(board)) == keep_list.end()) {
                delete_elem(board);
                ++(*n_delete);
                if ((*n_delete) % 1000 == 0) {
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
            uint64_t strt = tim();
            uint64_t n_flags = 0, n_delete = 0;
            uint64_t book_size = book.size();
            std::unordered_map<Board, int, Book_hash> keep_list; // key-remaining_error
            // reduce_book_calculate_keeplist(root_board, max_depth, max_error_per_move, max_line_error, &n_flags, keep_list, doing);
            std::unordered_map<Board, int, Book_hash> root_list;
            Board root_unique = representative_board(root_board);
            root_list[root_unique] = max_line_error;
            keep_list[root_unique] = max_line_error;
            ++n_flags;
            for (int depth = 0; depth < max_depth && !root_list.empty(); ++depth) {
                std::unordered_map<Board, int, Book_hash> next_root_list = reduce_book_calculate_keeplist_bfs(depth, root_list, max_error_per_move, &n_flags, keep_list, doing);
                for (const auto &entry: next_root_list) {
                    auto keep_itr = keep_list.find(entry.first);
                    if (keep_itr == keep_list.end() || keep_itr->second < entry.second) {
                        keep_list[entry.first] = entry.second;
                    }
                }
                root_list = std::move(next_root_list);
                std::cerr << "done depth " << depth << " next task " << root_list.size() << " keep " << keep_list.size() << std::endl;
            }
            std::cerr << "keep " << keep_list.size() << " boards, updating leaves" << std::endl;
            reset_seen();
            reduce_book_update_leaves(root_board, keep_list, doing);
            reset_seen();
            reduce_book_delete_unflagged_moves(root_board, &n_delete, keep_list, doing);
            reset_seen();
            uint64_t elapsed = tim() - strt;
            std::cerr << "book reduced size before " << book_size << " n_keep " << n_flags << " n_delete " << n_delete << " elapsed " << elapsed << " ms" << std::endl;
            *doing = false;
        }

        void delete_terminal_midsearch_rec(Board board, bool *doing) {
            if (!(*doing)) {
                return;
            }
            if (board.get_legal() == 0) { // the passed board is registered in book
                board.pass();
                if (board.get_legal() == 0) {
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

        /*
            @brief recalculate n_lines of the boards which can be reached from the root board

            n_lines of a board is 1 + sum of n_lines of the boards after registered moves (game over: 1).
            Boards with more discs are processed first, boards with the same number of discs in parallel.
        */
        void recalculate_n_lines_reachable(Board root_board, bool *stop) {
            int max_n_discs = flag_reachable_boards(root_board, true, stop);
            for (int n_discs = max_n_discs; n_discs >= 4 && !(*stop); --n_discs) {
                parallel_for_each_seen_board(n_discs, [this, stop](const Board &board, Book_elem &book_elem) {
                    if (*stop) {
                        return;
                    }
                    Moved_board moved_boards[HW2];
                    int n_moved_boards = get_moved_boards(&board, moved_boards, true);
                    uint64_t n_lines = 1;
                    Book_elem child_elem;
                    for (int i = 0; i < n_moved_boards; ++i) {
                        const Moved_board &moved_board = moved_boards[i];
                        if (moved_board.is_end) { // game over
                            if (book.get_unlocked(moved_board.board, moved_board.hash, &child_elem) || book.get_unlocked(moved_board.end_board, moved_board.end_hash, &child_elem)) {
                                ++n_lines;
                            }
                        } else if (book.get_unlocked(moved_board.board, moved_board.hash, &child_elem)) {
                            n_lines += child_elem.n_lines;
                        }
                    }
                    // boards with this number of discs are not read by other threads now
                    book_elem.n_lines = (uint32_t)std::min((uint64_t)MAX_N_LINES, n_lines);
                });
            }
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
            Book_elem book_elem;
            bool already_seen = false;
            book.modify(board, [&book_elem, &already_seen](Book_elem &elem) {
                already_seen = elem.seen;
                elem.seen = true;
                book_elem = elem;
            });
            // already seen
            if (already_seen) {
                return 0;
            }
            std::vector<Book_value> links = get_all_moves_with_value(&board);
            Flip flip;
            bool leaf_valid = is_valid_policy(book_elem.leaf.move) && is_valid_score(book_elem.leaf.value) && (board.get_legal() & (1ULL << book_elem.leaf.move));
            if (links.size() && leaf_valid && !has_book_value_policy(links, book_elem.leaf.move)) {
                int link_max = -SCORE_INF;
                for (Book_value &link: links) {
                    link_max = std::max(link_max, link.value);
                }
                if (book_elem.leaf.value > link_max) { // upgrade
                    //board.print();
                    //std::cerr << (int)book_elem.value << " " << link_max << " " << idx_to_coord(book_elem.leaf.move) << " " << (int)book_elem.leaf.value << std::endl;
                    calc_flip(&flip, &board, book_elem.leaf.move);
                    board.move_board(&flip);
                        Board new_board = board.copy();
                        Book_elem new_elem;
                        new_elem.level = book_elem.leaf.level;
                        new_elem.seen = false;
                        new_elem.value = -book_elem.leaf.value;
                        if (new_board.get_legal() == 0 && !new_board.is_end()) { // pass: the passed board is registered in book
                            new_board.pass();
                            new_elem.value = book_elem.leaf.value;
                        }
                        reg(&new_board, new_elem);
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
            recalculate_n_lines_reachable(root_board, stop);
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
            parallel_for_segments([this](int segment_idx) {
                book.for_each_in_segment(segment_idx, [](const Board &board, Book_elem &book_elem) {
                    book_elem.seen = false;
                });
            });
        }

        void flag_book_elem(Board board) {
            book.modify(representative_board(board), [](Book_elem &book_elem) {
                book_elem.seen = true;
            });
        }

        /*
            @brief call func(Board) for all registered boards

            Boards are copied segment by segment, so that func can modify the book.
            If func returns false, iteration stops.
        */
        template<class F>
        inline void for_each_board_snapshot(F &&func) {
            std::vector<Board> boards;
            for (int segment_idx = 0; segment_idx < book.N_SEGMENTS; ++segment_idx) {
                book.get_keys_in_segment(segment_idx, boards);
                for (Board &board: boards) {
                    if (!func(board)) {
                        return;
                    }
                }
            }
        }

        void add_leaf(Board *board, int8_t value, int8_t policy, int8_t level) {
            std::lock_guard<std::mutex> lock(mtx);
            int rotate_idx;
            Board representive_board = representative_board(board, &rotate_idx);
            int8_t rotated_policy = policy;
            if (is_valid_policy(policy)) {
                rotated_policy = convert_coord_to_representative_board((int)policy, rotate_idx);
            }
            Leaf leaf;
            leaf.value = value;
            leaf.move = rotated_policy;
            leaf.level = level;
            book.modify(representive_board, [&leaf](Book_elem &book_elem) {
                book_elem.leaf = leaf;
            });
        }

        void search_leaf(Board board, int level, bool use_multi_thread) {
            Book_elem book_elem;
            book.get(board, &book_elem);
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
            // each board is checked independently: process segments in parallel
            parallel_for_segments([this](int segment_idx) {
                Flip flip;
                std::vector<Board> boards;
                book.get_keys_in_segment(segment_idx, boards);
                for (Board &board: boards) {
                    Book_elem book_elem;
                    if (!book.get(board, &book_elem)) {
                        continue;
                    }
                    int leaf_move = book_elem.leaf.move;
                    bool need_to_rewrite_leaf = leaf_move < 0 || MOVE_UNDEFINED <= leaf_move;
                    if (!need_to_rewrite_leaf && is_valid_policy(leaf_move)) {
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
            });
        }

        void check_add_leaf_all_search(int level, bool *stop) {
            Flip flip;
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1;
            uint64_t t = 0, n_boards = book.size();
            Book_elem book_elem;
            std::vector<std::future<void>> tasks;
            uint64_t n_doing = 0, n_done = 0;
            for_each_board_snapshot([&](Board &board) {
                if (!global_searching || *stop) {
                    return false;
                }
                int n_percent = n_boards ? (int)std::min(100.0, (double)t++ / n_boards * 100) : 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << " n_recalculated " << n_done << std::endl;
                }
                book_elem = get(board);
                int leaf_move = book_elem.leaf.move;
                bool need_to_rewrite_leaf = !is_valid_policy(leaf_move) || (board.get_legal() & (1ULL << leaf_move)) == 0;
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
                    bool use_multi_thread = (int64_t)(n_boards - n_doing) < (int64_t)thread_pool.get_n_idle();
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
                return true;
            });
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
            std::cerr << "add leaf to book" << std::endl;
            int percent = -1;
            uint64_t t = 0, n_boards = book.size();
            for_each_board_snapshot([&](Board &board) {
                if (!global_searching || *stop) {
                    return false;
                }
                int n_percent = n_boards ? (int)std::min(100.0, (double)t / n_boards * 100) : 100;
                if (n_percent > percent) {
                    percent = n_percent;
                    std::cerr << "adding leaf " << percent << "%" << std::endl;
                }
                ++t;
                search_leaf(board, level);
                return true;
            });
        }

        Book_info calculate_book_info(bool *calculating) {
            Book_info res;
            bool stopped = false;
            for (int segment_idx = 0; segment_idx < book.N_SEGMENTS && !stopped; ++segment_idx) {
                stopped = !book.for_each_in_segment_locked_while(segment_idx, [&res, calculating](const Board &board, const Book_elem &book_elem) {
                    if (!(*calculating)) {
                        return false;
                    }
                    int level = book_elem.level;
                    int ply = board.n_discs() - 4;
                    int leaf_level = book_elem.leaf.level;
                    int leaf_ply = ply + 1;
                    ++res.n_boards;
                    if (0 <= level && level <= LEVEL_HUMAN) {
                        ++res.n_boards_in_level[level];
                    }
                    if (0 <= ply && ply <= HW2 - 4) {
                        ++res.n_boards_in_ply[ply];
                    }
                    if (0 <= leaf_level && leaf_level <= LEVEL_HUMAN) {
                        ++res.n_leaves_in_level[leaf_level];
                        if (0 <= leaf_ply && leaf_ply <= HW2 - 4) {
                            ++res.n_leaves_in_ply[leaf_ply];
                        }
                    }
                    return true;
                });
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
            book.set(board, elem);
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
            return book.set(b, elem);
        }

        /*
            @brief delete a board

            @param b                    a board to delete
            @return board deleted?
        */
        inline bool delete_representative_board(Board b) {
            return book.erase(b);
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
            if (is_valid_policy(elem.leaf.move)) { // do not convert special values (MOVE_NOMOVE etc.)
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
            int idx;
            Board representive_board = representative_board(b, &idx);
            if (is_valid_policy(elem.leaf.move)) { // do not convert special values (MOVE_NOMOVE etc.)
                elem.leaf.move = convert_coord_to_representative_board(elem.leaf.move, idx);
            }
            return book.upsert(representive_board, elem, [&elem](Book_elem &registered) {
                merge_book_elem(registered, elem);
            });
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

void book_save_as_egaroucid(std::string file, int level, bool *stop_saving) {
    const std::string tmp_file = file + ".tmp";
    remove(tmp_file.c_str());
    if (!book.save_egbk3(tmp_file, level, stop_saving)) { // stopped or failed: keep the existing file
        remove(tmp_file.c_str());
        return;
    }
    remove(file.c_str());
    rename(tmp_file.c_str(), file.c_str());
}

void book_save_as_edax(std::string file, int level, int leaf_recalculation_level) {
    book.save_bin_edax(file, level, true, nullptr, leaf_recalculation_level);
}

void book_save_as_edax(std::string file, int level) {
    book_save_as_edax(file, level, ADD_LEAF_SPECIAL_LEVEL);
}

void book_save_as_edax(std::string file, int level, int leaf_recalculation_level, bool *stop_saving) {
    const std::string tmp_file = file + ".tmp";
    remove(tmp_file.c_str());
    book.save_bin_edax(tmp_file, level, true, stop_saving, leaf_recalculation_level);
    if (stop_saving && *stop_saving) {
        remove(tmp_file.c_str());
        return;
    }
    std::ifstream exported(tmp_file, std::ios::binary);
    if (!exported) {
        return;
    }
    exported.close();
    remove(file.c_str());
    rename(tmp_file.c_str(), file.c_str());
}

void book_save_as_edax(std::string file, int level, bool *stop_saving) {
    book_save_as_edax(file, level, ADD_LEAF_SPECIAL_LEVEL, stop_saving);
}

void book_save_as_edax_without_additional_calculation(std::string file, int level) {
    book.save_bin_edax_without_additional_calculation(file, level);
}

void book_save_as_edax_without_additional_calculation(std::string file, int level, bool *stop_saving) {
    const std::string tmp_file = file + ".tmp";
    remove(tmp_file.c_str());
    book.save_bin_edax_without_additional_calculation(tmp_file, level, stop_saving);
    if (stop_saving && *stop_saving) {
        remove(tmp_file.c_str());
        return;
    }
    std::ifstream exported(tmp_file, std::ios::binary);
    if (!exported) {
        return;
    }
    exported.close();
    remove(file.c_str());
    rename(tmp_file.c_str(), file.c_str());
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
            use_level = ADD_LEAF_SPECIAL_LEVEL_SEARCH_LEVEL;
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
    if (level == ADD_LEAF_SPECIAL_LEVEL) {
        book.add_leaf(&board, new_leaf_value, new_leaf_move, ADD_LEAF_SPECIAL_LEVEL_SEARCH_LEVEL);
    } else {
        book.add_leaf(&board, new_leaf_value, new_leaf_move, level);
    }
}

Book_info calculate_book_info(bool *calculating) {
    Book_info res = book.calculate_book_info(calculating);
    return res;
}
