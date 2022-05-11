#include <iostream>
#include "./../setting.hpp"
#include "./../bit.hpp"
#include "./../board.hpp"
#include "./../util.hpp"

using namespace std;

int main(){
    Board board;
    board.p = 0;
    board.player = 9264090048841921536ULL;
    board.opponent = 4480051588954062904ULL;
    board.print();

    return 0;
}