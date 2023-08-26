#include "engine/evaluate.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

int main(int argc, char *argv[]){
    if (argc < 4){
        std::cerr << "input [in_file] [n_games] [eval_file]" << std::endl;
        return 1;
    }
    char *in_file = argv[1];
    int n_games = atoi(argv[2]);
    char *eval_file = argv[3];

    thread_pool.resize(15);
    bit_init();
    mobility_init();
    flip_init();
    if (!evaluate_init(eval_file, true))
        std::exit(0);

    std::ifstream ifs(in_file);
    if (ifs.fail()){
        std::cerr << "evaluation file not exist" << std::endl;
        return 1;
    }
    std::string line;
    std::vector<std::vector<std::vector<double>>> errors;
    for (int i = 0; i < 60; ++i){
        std::vector<std::vector<int>> arr2;
        for (int j = 0; j < 60; ++j){
            std::vector<int> arr;
            arr2.emplace_back(arr);
        }
        errors.emplace_back(arr2);
    }
    int vals[60];
    for (int i = 0; i < n_games; ++i){
        getline(ifs, line);
        Board board;
        board.reset();
        Flip flip;
        int sign = 1;
        for (int j = 0; j < (int)line.size(); j += 2){
            uint_fast8_t x = line[j] - 'a';
            uint_fast8_t y = line[j + 1] - '1';
            uint_fast8_t coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board, coord);
            board.move_board(&flip);
            sign *= -1;
            vals[j / 2] = mid_evaluate(&board) * sign;
            if (board.get_legal() == 0ULL){
                board.pass();
                sign *= -1;
            }
        }
        int max_mov = line.size() / 2;
        for (int m1 = 0; m1 < max_mov; ++m1){
            for (int m2 = m1 + 1; m2 < max_mov; ++m2){
                errors[m1][m2] = vals[m1] - vals[m2];
            }
        }
    }

    double sigmas[60];
    for (int mov = 0; mov < 60; ++mov){
        double avg = 0.0;
        for (int &err: errors[mov])
            avg += (double)err / errors[mov].size();
        double sigma2 = 0.0;
        for (int &err: errors[mov])
            sigma2 += (double)(avg - err) / errors[mov].size() * (avg - err);
        double sigma = sqrt(sigma2);
        std::cerr << "mov " << mov << " avg " << avg << " sigma2 " << sigma2 << " sigma " << sigma << std::endl;
        sigmas[mov] = sigma;
    }

    for (int d1 = 0; d1 < 50; ++d1){
        for (int d2 = d1 + 1; d2 < 60; ++d2)
            std::cout << d1 << " " << d2 << " " << sqrt(sigmas[d1] * sigmas[d1] + sigmas[d2] * sigmas[d2]) << std::endl;
    }

    return 0;
}