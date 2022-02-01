#include <iostream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "mobility.hpp"
#include "board.hpp"


inline void input_board(board *b, int ai_player){
    int i, j;
    char elem;
    int arr[hw2];
    for (i = 0; i < hw2; ++i){
        cin >> elem;
        if (elem == '.'){
            arr[i] = vacant;
        } else
            arr[i] = (int)elem - (int)'0';
    }
    b->translate_from_arr(arr, ai_player);
}

int main(){
    mobility_init();
    board b;
    mobility m;
    unsigned long long mob;
    int ai_player;
    while (true){
        cin >> ai_player;
        input_board(&b, ai_player);
        b.print();
        /*
        for (int i = 23; i < 24; ++i){
            cerr << i << endl;
            calc_flip(&m, &b, i);
            cerr << endl << endl;
            b.move(&m);
            b.print();
        }
        return 0;
        */
        mob = b.mobility_ull();
        for (int i = 0; i < hw2; ++i){
            if (1 & (mob >> i)){
                calc_flip(&m, &b, i);
                for (int i = hw2_m1; i >= 0; --i){
                    cerr << (1 & (m.flip >> i));
                    if (i % hw == 0)
                        cerr << endl;
                }
                b.move(&m);
                cerr << i << endl;
                b.print();
                cerr << endl;
                b.undo(&m);
                //b.print();
                cerr << endl;
            }
        }
    }
    return 0;
}