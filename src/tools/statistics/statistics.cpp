#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <algorithm>
#include "../../engine/board.hpp"

using namespace std;

#define N_DIRS 1

int N_EMPTIES_MIN = 5;
int N_EMPTIES_MAX = 7;

#define N_DIGIT 7
#define N_DOUBLE_DIGIT 3

#define INDENT "  "

void print_arr_64(uint64_t board_arr[], string head){
    for (int i = 0; i < HW; ++i){
        cout << head;
        for (int j = 0; j < HW; ++j)
            cout << right << setw(N_DIGIT) << board_arr[i * HW + j] << " ";
        cout << endl;
    }
    cout << endl;
}

void print_arr_10(uint64_t board_arr[], string head){
    uint64_t arr10[10];
    constexpr int cells[10][8] = {
        {0, 7, 56, 63, -1},
        {1, 6, 8, 15, 48, 55, 57, 62},
        {2, 5, 23, 16, 40, 47, 58, 61},
        {3, 4, 24, 31, 32, 39, 59, 60},
        {9, 14, 49, 54, -1},
        {10, 13, 17, 22, 41, 46, 50, 53},
        {11, 12, 25, 30, 33, 38, 51, 52},
        {18, 21, 42, 45, -1},
        {19, 20, 26, 29, 34, 37, 43, 44},
        {27, 28, 35, 36, -1}
    };
    for (int i = 0; i < 10; ++i){
        arr10[i] = 0ULL;
        for (int j = 0; j < 8; ++j){
            if (cells[i][j] == -1)
                break;
            arr10[i] += board_arr[cells[i][j]];
        }
    }
    int t = 0;
    for (int i = 0; i < 4; ++i){
        cout << head;
        for (int j = 0; j < i; ++j){
            for (int k = 0; k < N_DIGIT + 1; ++k)
                cout << " ";
        }
        for (int j = 4 - i; j > 0; --j)
            cout << right << setw(N_DIGIT) << arr10[t++] << " ";
        cout << endl;
    }
    cout << endl;
}

void print_arr(uint64_t board_arr[], string head){
    cout << head << "64 cells: " << endl;
    print_arr_64(board_arr, head + INDENT);
    cout << head << "10 cells: " << endl;
    print_arr_10(board_arr, head + INDENT);
    cout << endl;
}

void print_arr_64(double board_arr[], string head){
    for (int i = 0; i < HW; ++i){
        cout << head;
        for (int j = 0; j < HW; ++j)
            cout << setprecision(N_DOUBLE_DIGIT) << right << setw(N_DIGIT) << board_arr[i * HW + j] << " ";
        cout << endl;
    }
    cout << endl;
}

bool sort_priority(pair<int, double> a, pair<int, double> b){
    return a.second > b.second;
}

void print_arr_10(double board_arr[], string head){
    double arr10[10];
    constexpr int cells[10][8] = {
        {0, 7, 56, 63, -1},
        {1, 6, 8, 15, 48, 55, 57, 62},
        {2, 5, 23, 16, 40, 47, 58, 61},
        {3, 4, 24, 31, 32, 39, 59, 60},
        {9, 14, 49, 54, -1},
        {10, 13, 17, 22, 41, 46, 50, 53},
        {11, 12, 25, 30, 33, 38, 51, 52},
        {18, 21, 42, 45, -1},
        {19, 20, 26, 29, 34, 37, 43, 44},
        {27, 28, 35, 36, -1}
    };
    constexpr int n_cells[10] = {4, 8, 8, 8, 4, 8, 8, 4, 8, 4};
    vector<pair<int, double>> priorities;
    for (int i = 0; i < 10; ++i){
        arr10[i] = 0ULL;
        for (int j = 0; j < 8; ++j){
            if (cells[i][j] == -1)
                break;
            arr10[i] += board_arr[cells[i][j]];
        }
        arr10[i] /= n_cells[i];
        priorities.emplace_back(make_pair(i, arr10[i]));
    }
    sort(priorities.begin(), priorities.end(), sort_priority);
    int t = 0;
    for (int i = 0; i < 4; ++i){
        cout << head;
        for (int j = 0; j < i; ++j){
            for (int k = 0; k < N_DIGIT + 1; ++k)
                cout << " ";
        }
        for (int j = 4 - i; j > 0; --j)
            cout << setprecision(N_DOUBLE_DIGIT) << right << setw(N_DIGIT) << arr10[t++] << " ";
        cout << endl;
    }
    cout << head << "priority: ";
    for (int i = 0; i < 10; ++i)
        cout << priorities[i].first << " ";
    cout << endl;
    cout << endl;
}

void print_arr(double board_arr[], string head){
    cout << head << "64 cells: " << endl;
    print_arr_64(board_arr, head + INDENT);
    cout << head << "10 cells: " << endl;
    print_arr_10(board_arr, head + INDENT);
}

void print_arr(uint64_t arr[], int siz, string head){
    cout << head;
    for (int i = 0; i < siz; ++i)
        cout << right << setw(N_DIGIT) << arr[i] << " ";
    cout << endl;
}

void print_arr(double arr[], int siz, string head){
    cout << head;
    for (int i = 0; i < siz; ++i)
        cout << setprecision(N_DOUBLE_DIGIT) << right << setw(N_DIGIT) << arr[i] << " ";
    cout << endl;
}

struct Data{
    uint64_t n_data;
    uint64_t places[HW2];
    uint64_t parities[4];
    double places_weighted[HW2];
    double parities_weighted[2];
    uint64_t n_next_legals[35];
    double n_next_legals_weighted[35];


    Data(){
        n_data = 0;
        for (int i = 0; i < HW2; ++i)
            places[i] = 0;
        for (int i = 0; i < 4; ++i)
            parities[i] = 0;
        for (int i = 0; i < HW2; ++i)
            places_weighted[i] = 0.0;
        for (int i = 0; i < 2; ++i)
            parities_weighted[i] = 0.0;
        for (int i = 0; i < 35; ++i)
            n_next_legals[i] = 0;
        for (int i = 0; i < 35; ++i)
            n_next_legals_weighted[i] = 0.0;
    }

    void print(){
        cout << "n_data: " << n_data << endl;

        cout << "places: " << endl;
        print_arr(places, INDENT);
        cout << endl;

        cout << "parities: " << endl;
        cout << INDENT << "even:       " << parities[0] << endl;
        cout << INDENT << "odd:        " << parities[1] << endl;
        cout << INDENT << "even only:  " << parities[2] << endl;
        cout << INDENT << "odd only:   " << parities[3] << endl;
        cout << endl;

        cout << "places_weighted: " << endl;
        for (int i = 0; i < HW2; ++i)
            places_weighted[i] /= max(1ULL, places[i]);
        print_arr(places_weighted, INDENT);
        cout << endl;

        cout << "parities_weighted: " << endl;
        cout << INDENT << "even:       " << parities_weighted[0] / parities[0] << endl;
        cout << INDENT << "odd:        " << parities_weighted[1] / parities[1] << endl;
        cout << endl;

        cout << "n_next_legals: " << endl;
        print_arr(n_next_legals, 35, INDENT);
        cout << endl;

        cout << "n_next_legals_weighted: " << endl;
        for (int i = 0; i < 35; ++i)
            n_next_legals_weighted[i] /= max(1ULL, n_next_legals[i]);
        print_arr(n_next_legals_weighted, 35, INDENT);
        cout << endl;

        cout << endl << "done" << endl;
    }
};

void add_data(Data *data, string line){
    Board board;
    Flip flip;
    board.player = 0ULL;
    board.opponent = 0ULL;
    uint64_t p = 0ULL, o = 0ULL;
    for (int i = 0; i < HW2; ++i){
        if (line[i] == 'p')
            board.player |= 1ULL << i;
        else if (line[i] == 'o')
            board.opponent |= 1ULL << i;
    }
    int n_discs = pop_count_ull(board.player | board.opponent);
    if (HW2 - N_EMPTIES_MAX <= n_discs && n_discs <= HW2 - N_EMPTIES_MIN){
        int policy = line[HW2] - '!';
        int score = line[HW2 + 1] - '!';
        score *= 2;
        score -= HW2;

        uint64_t empty = ~(board.player | board.opponent);
        int parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
        parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
        parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
        parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;

        uint64_t legal = board.get_legal();
        int n_legal = pop_count_ull(legal);

        int n_even_legals = 0, n_odd_legals = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            if (parity & cell_div4[cell])
                ++n_odd_legals;
            else
                ++n_even_legals;
        }

        int best_next_n_moves = 1000;
        legal = board.get_legal();
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move(&flip);
                best_next_n_moves = min(best_next_n_moves, pop_count_ull(board.get_legal()));
            board.undo(&flip);
        }

        ++data->n_data;
        ++data->places[policy];
        if (parity == 0)
            ++data->parities[2]; // even only
        if (parity == 0b1111)
            ++data->parities[3]; // odd only
        else if (parity & cell_div4[policy])
            ++data->parities[1]; // odd
        else
            ++data->parities[0]; // even
        data->places_weighted[policy] += 1.0 - (1.0 / n_legal);
        if (parity != 0 && parity != 0b1111){
            if (parity & cell_div4[policy])
                data->parities_weighted[1] += 1.0 - ((double)n_odd_legals / n_legal); // odd
            else
                data->parities_weighted[0] += 1.0 - ((double)n_even_legals / n_legal); // even
        }
        calc_flip(&flip, &board, policy);
        board.move(&flip);
        int next_n_legals = pop_count_ull(board.get_legal());
        ++data->n_next_legals[next_n_legals];
        data->n_next_legals_weighted[next_n_legals] += (double)(1 + best_next_n_moves) / (1 + next_n_legals);
    }
}

int main(int argc, char *argv[]){
    bit_init();
    board_init();
    flip_init();

    cout << fixed;
    if (argc == 2){
        N_EMPTIES_MIN = atoi(argv[1]);
        N_EMPTIES_MAX = atoi(argv[1]);
    } else if (argc == 3){
        N_EMPTIES_MIN = atoi(argv[1]);
        N_EMPTIES_MAX = atoi(argv[2]);
    }
    cout << N_EMPTIES_MIN << " empties to " << N_EMPTIES_MAX << " empties" << endl;
    string data_dir = "data/";
    string sub_dirs[N_DIRS] = {
        "records15/"
    };
    int n_data[N_DIRS] = {
        1
    };
    Data data;
    for (int dir_idx = 0; dir_idx < N_DIRS; ++dir_idx){
        for (int file_idx = 0; file_idx < n_data[dir_idx]; ++file_idx){
            ostringstream sout;
            sout << setfill('0') << setw(7) << file_idx;
            string file_name = sout.str();
            ifstream ifs(data_dir + sub_dirs[dir_idx] + file_name + ".txt");
            string line;
            while (getline(ifs, line)){
                add_data(&data, line);
            }
        }
    }
    data.print();
    return 0;
}