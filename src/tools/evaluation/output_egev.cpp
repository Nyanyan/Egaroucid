#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ios>
#include <iomanip>

#define EVAL_MAX 4091
//#define SIMD_EVAL_OFFSET 4092

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "input [model_dir] [n_phases]" << std::endl;
        return 1;
    }
    std::string model_dir = std::string(argv[1]);
    int n_phases = atoi(argv[2]);
    std::ofstream fout;
    fout.open("eval.egev", std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open eval.egev" << std::endl;
        return 1;
    }
    short elem;
    short max_elem = -10000, min_elem = 10000;
    for (int phase = 0; phase < n_phases; ++phase){
        std::ifstream ifs(model_dir + "/" + std::to_string(phase) + ".txt");
        if (ifs.fail()){
            std::cerr << (model_dir + "/" + std::to_string(phase) + ".txt") << " not exist" << std::endl;
            return 0;
        }
        std::string line;
        int t = 0;
        while (std::getline(ifs, line)){
            elem = stoi(line);
            if (elem > EVAL_MAX)
                elem = EVAL_MAX;
            else if (elem < -EVAL_MAX)
                elem = -EVAL_MAX;
            max_elem = std::max(max_elem, elem);
            min_elem = std::min(min_elem, elem);
            //elem += SIMD_EVAL_OFFSET;
            fout.write((char*)&elem, 2);
            ++t;
        }
        std::cerr << phase << " " << t << std::endl;
    }
    std::cerr << "EVAL_MAX " << EVAL_MAX << std::endl;
    std::cerr << "min " << min_elem << std::endl;
    std::cerr << "max " << max_elem << std::endl;
    std::cerr << "done" << std::endl;

    return 0;
}