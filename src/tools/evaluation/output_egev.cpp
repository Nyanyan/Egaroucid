#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ios>
#include <iomanip>

//#define EVAL_MAX 4091 // for 16 patterns
#define EVAL_MAX 3270 // for 20 patterns

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cerr << "input [n_phases] [eval_max=" << EVAL_MAX << "] [out_file=trained/eval.egev]" << std::endl;
        return 1;
    }
    std::string model_dir = "./trained";
    int n_phases = atoi(argv[1]);
    int eval_max = EVAL_MAX;
    if (argc >= 3){
        eval_max = atoi(argv[2]);
    }
    std::string out_file = "trained/eval.egev";
    if (argc >= 4){
        out_file = argv[3];
    }
    std::cerr << "n_phases " << n_phases << " eval_max " << eval_max << " out_file " << out_file << std::endl;
    std::ofstream fout;
    fout.open(out_file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open eval.egev" << std::endl;
        return 1;
    }
    short elem;
    int max_elem = -10000000, min_elem = 10000000;
    int n_over = 0, n_under = 0;
    int n_params = -1;
    for (int phase = 0; phase < n_phases; ++phase){
        std::ifstream ifs(model_dir + "/" + std::to_string(phase) + ".txt");
        if (ifs.fail()){
            std::cerr << (model_dir + "/" + std::to_string(phase) + ".txt") << " not exist" << std::endl;
            if (n_params == -1){
                std::cerr << "max " << max_elem << " min " << min_elem << std::endl;
                std::cerr << "n_over " << n_over << " n_under " << n_under << std::endl;
                return 1;
            } else{
                elem = 0;
                for (int i = 0; i < n_params; ++i){
                    fout.write((char*)&elem, 2);
                }
            }
        } else{
            int t = 0;
            std::string line;
            while (std::getline(ifs, line)){
                int elem_int = stoi(line);
                max_elem = std::max((int)max_elem, elem_int);
                min_elem = std::min((int)min_elem, elem_int);
                if (elem_int > eval_max){
                    elem_int = eval_max;
                    ++n_over;
                } else if (elem_int < -eval_max){
                    elem_int = -eval_max;
                    ++n_under;
                }
                elem = (short)elem_int;
                //max_elem = std::max(max_elem, elem);
                //min_elem = std::min(min_elem, elem);
                fout.write((char*)&elem, 2);
                ++t;
            }
            std::cerr << phase << " " << t << std::endl;
            n_params = t;
        }
    }
    std::cerr << "eval_max " << eval_max << std::endl;
    std::cerr << "max " << max_elem << " min " << min_elem << std::endl;
    std::cerr << "n_over " << n_over << " n_under " << n_under << std::endl;
    std::cerr << "done" << std::endl;

    return 0;
}
