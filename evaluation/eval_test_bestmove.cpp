#include <iostream>
#include <fstream>

#define HW2 64

using namespace std;

inline void output_board(unsigned long long p, unsigned long long o, int best_move){
    for (int i = 0; i < HW2; ++i){
        if (1 & (p >> i)){
            cout << '0';
        } else if (1 & (o >> i)){
            cout << '1';
        } else{
            cout << '.';
        }
    }
    cout << best_move / 10 << best_move % 10 << endl;
}

inline bool import_edax_book(string file) {
    FILE* fp;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        cerr << "can't open " << file << endl;
        return false;
    }
    char elem_char;
    int elem_int;
    short elem_short;
    int i, j;
    for (i = 0; i < 38; ++i){
        if (fread(&elem_char, 1, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
    }
    if (fread(&elem_int, 4, 1, fp) < 1) {
        cerr << "file broken" << endl;
        fclose(fp);
        return false;
    }
    int n_boards = elem_int;
    unsigned long long player, opponent;
    short value;
    char link = 0, link_value, link_move;
    int best_score, best_move;
    for (i = 0; i < n_boards; ++i){
        if (i % 32768 == 0)
            cerr << "loading edax book " << (i * 100 / n_boards) << "%" << endl;
        if (fread(&player, 8, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(&opponent, 8, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
        for (j = 0; j < 4; ++j) {
            if (fread(&elem_int, 4, 1, fp) < 1) {
                cerr << "file broken" << endl;
                fclose(fp);
                return false;
            }
        }
        if (fread(&value, 2, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
        for (j = 0; j < 2; ++j) {
            if (fread(&elem_short, 2, 1, fp) < 1) {
                cerr << "file broken" << endl;
                fclose(fp);
                return false;
            }
        }
        if (fread(&link, 1, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(&elem_char, 1, 1, fp) < 1) {
            cerr << "file broken" << endl;
            fclose(fp);
            return false;
        }
        best_score = -100000;
        best_move = -1;
        for (j = 0; j < (int)link + 1; ++j) {
            if (fread(&link_value, 1, 1, fp) < 1) {
                cerr << "file broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(&link_move, 1, 1, fp) < 1) {
                cerr << "file broken" << endl;
                fclose(fp);
                return false;
            }
            if (link_value > best_score){
                best_score = link_value;
                best_move = link_move;
            }
        }
        if (best_score != value)
            cerr << best_score << " " << value << endl;
        if (0 <= best_move && best_move < 64)
            output_board(player, opponent, best_move);
    }
    return true;
}

int main(){
    import_edax_book("third_party/okojo_book.dat");
    return 0;
}