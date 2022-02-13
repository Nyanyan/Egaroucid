#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "common.hpp"

#define p40 1
#define p41 4
#define p42 16
#define p43 64
#define p44 256
#define p45 1024
#define p46 4096
#define p47 16384
#define p48 65536

using namespace std;

inline int pop_digit(unsigned long long x, int place){
    return 1 & (x >> place);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2, const int p3){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44);
}


inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45) + 
        ((pop_digit(f, p6) * 3 + pop_digit(o, p6) * 2 + pop_digit(p, p6)) * p46);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45) + 
        ((pop_digit(f, p6) * 3 + pop_digit(o, p6) * 2 + pop_digit(p, p6)) * p46) + 
        ((pop_digit(f, p7) * 3 + pop_digit(o, p7) * 2 + pop_digit(p, p7)) * p47);
}

inline void calc_idx(unsigned long long p, unsigned long long o, unsigned long long v, unsigned long long f, char res, int n_moves){
    cout << n_moves << " ";

    cout << pick_pattern(p, o, v, f, 0, 1, 2, 3, 4, 5, 6, 7) << " ";
    cout << pick_pattern(p, o, v, f, 0, 8, 16, 24, 32, 40, 48, 56) << " ";
    cout << pick_pattern(p, o, v, f, 7, 15, 23, 31, 39, 47, 55, 63) << " ";
    cout << pick_pattern(p, o, v, f, 56, 57, 58, 59, 60, 61, 62, 63) << " ";

    cout << pick_pattern(p, o, v, f, 8, 9, 10, 11, 12, 13, 14, 15) << " ";
    cout << pick_pattern(p, o, v, f, 1, 9, 17, 25, 33, 41, 49, 57) << " ";
    cout << pick_pattern(p, o, v, f, 6, 14, 22, 30, 38, 46, 54, 62) << " ";
    cout << pick_pattern(p, o, v, f, 48, 49, 50, 51, 52, 53, 54, 55) << " ";

    cout << pick_pattern(p, o, v, f, 16, 17, 18, 19, 20, 21, 22, 23) << " ";
    cout << pick_pattern(p, o, v, f, 2, 10, 18, 26, 34, 42, 50, 58) << " ";
    cout << pick_pattern(p, o, v, f, 5, 13, 21, 29, 37, 45, 53, 61) << " ";
    cout << pick_pattern(p, o, v, f, 40, 41, 42, 43, 44, 45, 46, 47) << " ";

    cout << pick_pattern(p, o, v, f, 24, 25, 26, 27, 28, 29, 30, 31) << " ";
    cout << pick_pattern(p, o, v, f, 3, 11, 19, 27, 35, 43, 51, 59) << " ";
    cout << pick_pattern(p, o, v, f, 4, 12, 20, 28, 36, 44, 52, 60) << " ";
    cout << pick_pattern(p, o, v, f, 32, 33, 34, 35, 36, 37, 38, 39) << " ";
    
    cout << pick_pattern(p, o, v, f, 5, 14, 23) << " ";
    cout << pick_pattern(p, o, v, f, 2, 9, 16) << " ";
    cout << pick_pattern(p, o, v, f, 40, 49, 58) << " ";
    cout << pick_pattern(p, o, v, f, 61, 54, 47) << " ";

    cout << pick_pattern(p, o, v, f, 4, 13, 22, 31) << " ";
    cout << pick_pattern(p, o, v, f, 3, 10, 17, 24) << " ";
    cout << pick_pattern(p, o, v, f, 32, 41, 50, 59) << " ";
    cout << pick_pattern(p, o, v, f, 60, 53, 46, 39) << " ";

    cout << pick_pattern(p, o, v, f, 3, 12, 21, 30, 39) << " ";
    cout << pick_pattern(p, o, v, f, 4, 11, 18, 25, 32) << " ";
    cout << pick_pattern(p, o, v, f, 24, 33, 42, 51, 60) << " ";
    cout << pick_pattern(p, o, v, f, 59, 52, 45, 38, 31) << " ";

    cout << pick_pattern(p, o, v, f, 2, 11, 20, 29, 38, 47) << " ";
    cout << pick_pattern(p, o, v, f, 5, 12, 19, 26, 33, 40) << " ";
    cout << pick_pattern(p, o, v, f, 16, 25, 34, 43, 52, 61) << " ";
    cout << pick_pattern(p, o, v, f, 58, 51, 44, 37, 30, 23) << " ";

    cout << pick_pattern(p, o, v, f, 1, 10, 19, 28, 37, 46, 55) << " ";
    cout << pick_pattern(p, o, v, f, 6, 13, 20, 27, 34, 41, 48) << " ";
    cout << pick_pattern(p, o, v, f, 8, 17, 26, 35, 44, 53, 62) << " ";
    cout << pick_pattern(p, o, v, f, 57, 50, 43, 36, 29, 22, 15) << " ";

    cout << pick_pattern(p, o, v, f, 0, 9, 18, 27, 36, 45, 54, 63) << " ";
    cout << pick_pattern(p, o, v, f, 7, 14, 21, 28, 35, 42, 49, 56) << " ";
    cout << res << endl;
}

void solve(string line){
    unsigned long long p = 0, o = 0, v = 0, f = 0;
    int n_moves = (line[0] - '0') * 10 + (line[1] - '0');
    for (int i = 0; i < 64; ++i){
        if (line[3 + i] == '0')
            p |= 1ULL << i;
        else if (line[3 + i] == '1')
            o |= 1ULL << i;
        else if (line[3 + i] == '2')
            v |= 1ULL << i;
        else if (line[3 + i] == '3')
            f |= 1ULL << i;
    }
    char ans = line[68];
    calc_idx(p, o, v, f, ans, n_moves);
}

#define start_file 0
#define n_files 60

int main(){
    int t = 0;

    for (int i = start_file; i < n_files; ++i){
        cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/" + file_name + ".txt");
        if (ifs.fail()){
            cerr << "file not exist" << endl;
            return 0;
        }
        string line;
        while (getline(ifs, line)){
            ++t;
            solve(line);
        }
        if (i % 25 == 24)
            cerr << endl;
    }
    cerr << t << endl;

    return 0;

}