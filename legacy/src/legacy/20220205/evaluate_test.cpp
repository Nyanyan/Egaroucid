#include <iostream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "mobility.hpp"
#include "board.hpp"
#include "evaluate.hpp"


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
    evaluate_init();
    board b;
    mobility m;
    unsigned long long mob;
    int ai_player;
    while (true){
        cin >> ai_player;
        input_board(&b, ai_player);
        b.print();
        cerr << mid_evaluate(&b) << endl;
    }
    return 0;
}