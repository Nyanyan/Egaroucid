#include <iostream>
#include <chrono>
#include <string>
#include <vector>

using namespace std;

#define n_phases 15
#define phase_n_stones 4
#define n_patterns 11
#define n_dense2 2
#define n_add_input 3
#define n_add_dense1 8
#define n_all_input 30
#define max_canput 50
#define max_surround 80
#define max_evaluate_idx 59049

const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10};
int pattern_pow_sizes[n_patterns];
double pattern_arr[n_phases][2][n_patterns][max_evaluate_idx][n_dense2];
double add_arr[n_phases][2][n_add_input][max_surround + 1][n_add_dense1];
vector<vector<vector<int>>> test_data;


inline unsigned long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline double loss(double x){
    return x * x;
}

void input_param(){
    ifstream ifs("nn_proc_param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int phase_idx, player_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            cerr << "=";
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                for (pattern_elem = 0; pattern_elem < pattern_pow_sizes[pattern_idx]; ++pattern_elem){
                    for (dense_idx = 0; dense_idx < n_dense2; ++dense_idx){
                        getline(ifs, line);
                        pattern_arr[phase_idx][player_idx][pattern_idx][pattern_elem][dense_idx] = stof(line);
                    }
                }
            }
            for (canput = 0; canput <= max_canput; ++canput){
                for (sur0 = 0; sur0 <= max_surround; ++sur0){
                    for (sur1 = 0; sur1 <= max_surround; ++sur1){
                        for (dense_idx = 0; dense_idx < n_add_dense1; ++dense_idx){
                            getline(ifs, line);
                            add_arr[phase_idx][player_idx][canput][sur0][sur1][dense_idx] = stof(line);
                        }
                    }
                }
            }
        }
    }
    cerr << endl;
}

void input_test_data(){

}

void output_param(){
    int phase_idx, player_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            cerr << "=";
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                for (pattern_elem = 0; pattern_elem < pattern_pow_sizes[pattern_idx]; ++pattern_elem){
                    for (dense_idx = 0; dense_idx < n_dense2; ++dense_idx){
                        cout << pattern_arr[phase_idx][player_idx][pattern_idx][pattern_elem][dense_idx] << endl;
                    }
                }
            }
            for (canput = 0; canput <= max_canput; ++canput){
                for (sur0 = 0; sur0 <= max_surround; ++sur0){
                    for (sur1 = 0; sur1 <= max_surround; ++sur1){
                        for (dense_idx = 0; dense_idx < n_add_dense1; ++dense_idx){
                            cout << add_arr[phase_idx][player_idx][canput][sur0][sur1][dense_idx] << endl;
                        }
                    }
                }
            }
        }
    }
    cerr << endl;
}

void sa(unsigned long long tl){
    unsigned long long strt = tim(), now = tim();
    double score = scoring();
    int t;
    for (;;){
        ++t;

        

        if ((t & 0b11111111) == 0){
            now = tim();
            if (now - strt > tl)
                break;
        }
    }
}

int main(){
    for (int i = 0; i < n_patterns; ++i)
        pattern_pow_sizes[i] = pow(3, pattern_sizes[i]);

    input_param();
    sa();
    output_param();

    return 0;
}