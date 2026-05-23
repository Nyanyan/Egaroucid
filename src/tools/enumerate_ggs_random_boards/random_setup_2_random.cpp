#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <ios>
#include <filesystem>
#include "./../../engine/board.hpp"

constexpr int RANDOM_SETUP_2_START_DISCS = 5;
constexpr int GGS_RANDOM_SETUP_2_MIN_DISCS = 10;
constexpr int GGS_RANDOM_SETUP_MAX_DISCS = 48;
constexpr int N_BOARDS_PER_FILE = 10000;
constexpr uint64_t AVOID_CELLS = 0xC3C300000000C3C3ULL;

std::string fill0(int n, int d) {
    std::stringstream ss;
    ss << std::setfill('0') << std::right << std::setw(d) << n;
    return ss.str();
}

int randrange(std::mt19937_64 *engine, int n) {
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(*engine);
}

std::string get_str(Board board, int turn_color) {
    std::string res;
    for (int cell = HW2 - 1; cell >= 0; --cell) {
        uint64_t cell_bit = 1ULL << cell;
        if (turn_color == BLACK) {
            if (board.player & cell_bit) {
                res += "X";
            } else if (board.opponent & cell_bit) {
                res += "O";
            } else {
                res += "-";
            }
        } else {
            if (board.player & cell_bit) {
                res += "O";
            } else if (board.opponent & cell_bit) {
                res += "X";
            } else {
                res += "-";
            }
        }
    }
    res += " ";
    if (turn_color == BLACK) {
        res += "X";
    } else {
        res += "O";
    }
    return res;
}

inline int get_turn_color(int n_discs) {
    return (n_discs & 1) ? WHITE : BLACK;
}

void get_random_setup_cells(const int n_discs, std::mt19937_64 *engine, std::vector<int> *cells) {
    std::vector<std::vector<int>> cell_distance_list;
    for (int i = 0; i < HW; ++i) {
        cell_distance_list.emplace_back(std::vector<int>());
    }
    for (int cell = 0; cell < HW2; ++cell) {
        if ((AVOID_CELLS & (1ULL << cell)) == 0) {
            int x = cell % HW;
            int y = cell / HW;
            int d = (std::max(std::abs(2 * x - (HW - 1)), std::abs(2 * y - (HW - 1))) - 1) / 2;
            cell_distance_list[d].emplace_back(cell);
        }
    }

    std::vector<int> mm;
    for (int d = 0; d < HW; ++d) {
        if (!cell_distance_list[d].empty()) {
            std::shuffle(cell_distance_list[d].begin(), cell_distance_list[d].end(), *engine);
            mm.insert(mm.end(), cell_distance_list[d].begin(), cell_distance_list[d].end());
        }
    }

    int n_prefix = std::min(n_discs, (int)mm.size());
    std::shuffle(mm.begin(), mm.begin() + n_prefix, *engine);
    cells->assign(mm.begin(), mm.begin() + n_prefix);
}

int get_random_setup_n_white(const int n_discs, std::mt19937_64 *engine) {
    int rnd_white = n_discs / 2;
    if ((n_discs & 1) && randrange(engine, 2)) {
        ++rnd_white;
    }

    int imb = rnd_white / 3;
    rnd_white += randrange(engine, 2 * imb + 1) - imb;
    return rnd_white;
}

Board random_setup_5(std::mt19937_64 *engine) {
    std::vector<int> cells;
    get_random_setup_cells(RANDOM_SETUP_2_START_DISCS, engine, &cells);

    int n_white = get_random_setup_n_white(RANDOM_SETUP_2_START_DISCS, engine);
    int n_black = RANDOM_SETUP_2_START_DISCS - n_white;
    std::vector<int> colors;
    for (int i = 0; i < n_white; ++i) {
        colors.emplace_back(WHITE);
    }
    for (int i = 0; i < n_black; ++i) {
        colors.emplace_back(BLACK);
    }
    std::shuffle(colors.begin(), colors.end(), *engine);

    Board board;
    board.player = 0ULL;   // black
    board.opponent = 0ULL; // white
    for (int i = 0; i < RANDOM_SETUP_2_START_DISCS; ++i) {
        if (colors[i] == BLACK) {
            board.player |= 1ULL << cells[i];
        } else {
            board.opponent |= 1ULL << cells[i];
        }
    }
    board.pass(); // random_setup(5) leaves white to move
    return board;
}

int select_random_legal(uint64_t legal, std::mt19937_64 *engine) {
    int move_idx = randrange(engine, pop_count_ull(legal));
    for (int i = 0; i < move_idx; ++i) {
        next_bit(&legal);
    }
    return first_bit(&legal);
}

bool random_setup_2_once(const int n_discs, std::mt19937_64 *engine, Board *board) {
    *board = random_setup_5(engine);

    for (int move_idx = 0; move_idx < n_discs - RANDOM_SETUP_2_START_DISCS; ++move_idx) {
        uint64_t legal = board->get_legal() & ~AVOID_CELLS;
        if (legal == 0ULL) {
            return false;
        }

        int policy = select_random_legal(legal, engine);
        Flip flip;
        calc_flip(&flip, board, policy);
        board->move_board(&flip);
    }

    Board black_to_move_board = *board;
    if (get_turn_color(n_discs) == WHITE) {
        black_to_move_board.pass();
    }

    int n_black = black_to_move_board.count_player();
    int n_white = black_to_move_board.count_opponent();
    int min_count = std::max(n_discs / 4, 3);
    if (n_black <= min_count || n_white <= min_count) {
        return false;
    }

    return true;
}

Board generate_random_setup_2(const int n_discs, std::mt19937_64 *engine, uint64_t *n_failures) {
    Board board;
    while (!random_setup_2_once(n_discs, engine, &board)) {
        ++(*n_failures);
    }
    return board;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "input [n_discs] [n_boards] ([seed])" << std::endl;
        return 1;
    }

    int n_discs = atoi(argv[1]);
    if (n_discs < GGS_RANDOM_SETUP_2_MIN_DISCS || n_discs > GGS_RANDOM_SETUP_MAX_DISCS) {
        std::cerr << "random_setup_2 is used by GGS for n_discs from "
                  << GGS_RANDOM_SETUP_2_MIN_DISCS << " to " << GGS_RANDOM_SETUP_MAX_DISCS << std::endl;
        return 1;
    }

    int n_boards = atoi(argv[2]);
    if (n_boards < 0) {
        std::cerr << "n_boards must be non-negative" << std::endl;
        return 1;
    }

    uint64_t seed;
    if (argc >= 4) {
        seed = std::stoull(argv[3]);
    } else {
        std::random_device seed_gen;
        seed = ((uint64_t)seed_gen() << 32) ^ seed_gen();
    }
    std::mt19937_64 engine(seed);

    mobility_init();
    flip_init();

    uint64_t strt = tim();
    uint64_t n_failures = 0;
    int turn_color = get_turn_color(n_discs);

    std::filesystem::path output_root = std::filesystem::path(argv[0]).parent_path();
    if (output_root.empty()) {
        output_root = ".";
    }
    const std::string output_dir = (output_root / "output" / (std::to_string(n_discs) + "_random_setup_2_random")).string();
    std::filesystem::create_directories(output_dir);

    int n_files = (n_boards + N_BOARDS_PER_FILE - 1) / N_BOARDS_PER_FILE;
    int global_idx = 0;
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::string file = output_dir + "/" + fill0(file_idx, 7) + ".txt";
        std::cerr << "output file " << file << std::endl;
        std::ofstream ofs(file);
        if (!ofs) {
            std::cerr << "can't open " << file << std::endl;
            return 1;
        }
        for (int i = 0; i < N_BOARDS_PER_FILE && global_idx < n_boards; ++i) {
            Board board = generate_random_setup_2(n_discs, &engine, &n_failures);
            ofs << get_str(board, turn_color) << std::endl;
            ++global_idx;
        }
    }

    std::cerr << "generated " << n_boards << " boards" << std::endl;
    std::cerr << "failed attempts " << n_failures << std::endl;
    std::cerr << "seed " << seed << std::endl;
    std::cerr << "all done in " << tim() - strt << " ms" << std::endl;
    return 0;
}
