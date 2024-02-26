#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ios>
#include <iomanip>

#define EVAL_MAX 4091

int main(int argc, char* argv[]){
    // if (argc < 3){
    //     std::cerr << "input [model_dir] [n_phases]" << std::endl;
    //     return 1;
    // }
    if (argc < 2){
        std::cerr << "input [n_phases]" << std::endl;
        return 1;
    }
    // std::string model_dir = std::string(argv[1]);
    // int n_phases = atoi(argv[2]);
    std::string model_dir = "./trained";
    int n_phases = atoi(argv[1]);
    std::ofstream fout;
    fout.open("trained/eval.egev", std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open eval.egev" << std::endl;
        return 1;
    }
    short elem;
    int max_elem = -10000000, min_elem = 10000000;
    int n_over = 0, n_under = 0;
    for (int phase = 0; phase < n_phases; ++phase){
        std::ifstream ifs(model_dir + "/" + std::to_string(phase) + ".txt");
        if (ifs.fail()){
            std::cerr << (model_dir + "/" + std::to_string(phase) + ".txt") << " not exist" << std::endl;
            return 0;
        }
        std::string line;
        int t = 0;
        while (std::getline(ifs, line)){
            int elem_int = stoi(line);
            max_elem = std::max((int)max_elem, elem_int);
            min_elem = std::min((int)min_elem, elem_int);
            if (elem_int > EVAL_MAX){
                elem_int = EVAL_MAX;
                ++n_over;
            } else if (elem_int < -EVAL_MAX){
                elem_int = -EVAL_MAX;
                ++n_under;
            }
            elem = (short)elem_int;
            //max_elem = std::max(max_elem, elem);
            //min_elem = std::min(min_elem, elem);
            fout.write((char*)&elem, 2);
            ++t;
        }
        std::cerr << phase << " " << t << std::endl;
    }
    std::cerr << "EVAL_MAX " << EVAL_MAX << std::endl;
    std::cerr << "min " << min_elem << " max " << max_elem << std::endl;
    std::cerr << "n_over " << n_over << " n_under " << n_under << std::endl;
    std::cerr << "done" << std::endl;

    return 0;
}
