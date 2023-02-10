#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ios>
#include <iomanip>

#define N_PARAM 126860

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "input [model_dir] [n_phases]" << std::endl;
        return 1;
    }
    std::string model_dir = std::string(argv[1]);
    int n_phases = atoi(argv[2]);
    std::ofstream fout;
    fout.open(model_dir + "/move_ordering.egmo", std::ios::out|std::ios::binary|std::ios::trunc);
    if (!fout){
        std::cerr << "can't open move_ordering.egmo" << std::endl;
        return 1;
    }
    short elem;
    short max_elem = -10000, min_elem = 10000;
    std::ifstream ifs(model_dir + "/mo_eval.txt");
    if (ifs.fail()){
        std::cerr << (model_dir + "/mo_eval.txt") << " not exist" << std::endl;
        return 0;
    }
    std::string line;
    int t = 0;
    while (std::getline(ifs, line) && t < N_PARAM){
        elem = stoi(line);
        max_elem = std::max(max_elem, elem);
        min_elem = std::min(min_elem, elem);
        fout.write((char*)&elem, 2);
        ++t;
    }
    std::cerr << t << std::endl;
    std::cerr << "min " << min_elem << std::endl;
    std::cerr << "max " << max_elem << std::endl;
    std::cerr << "done" << std::endl;

    return 0;
}