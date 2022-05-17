#include <iostream>
#include <fstream>

using namespace std;

#define N_PARAM 804572

int arr1[N_PARAM];
int arr2[N_PARAM];

int main(int argc, char *argv[]){
    string line;

    ifstream ifs1(argv[1]);
    if (ifs1.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    for (int i = 0; i < N_PARAM; ++i){
        getline(ifs1, line);
        arr1[i] = stoi(line);
    }

    ifstream ifs2(argv[2]);
    if (ifs2.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    for (int i = 0; i < N_PARAM; ++i){
        getline(ifs2, line);
        arr2[i] = stoi(line);
    }

    uint64_t error_sum = 0;
    int div = 0;
    for (int i = 0; i < N_PARAM; ++i){
        if (arr1[i] == 0)
            continue;
        error_sum += abs(arr1[i] - arr2[i]);
        ++div;
    }
    cout << (double)error_sum / div << " " << (double)error_sum / div / 256 << endl;
}