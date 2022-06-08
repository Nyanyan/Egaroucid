#include <iostream>

using namespace std;

int main(){
    uint64_t place, neighbor;
    cout << showbase;
    for (int cell = 0; cell < 64; ++cell){
        place = 1ULL << cell;
        neighbor = ((place & 0x7F7F7F7F7F7F7F7FULL) << 1) | ((place & 0xFEFEFEFEFEFEFEFEULL) >> 1);
        neighbor |= ((place & 0x00FFFFFFFFFFFFFFULL) << 8) | ((place & 0xFFFFFFFFFFFFFF00ULL) >> 8);
        neighbor |= ((place & 0xFEFEFEFEFEFEFEFEULL) << 7) | ((place & 0x7F7F7F7F7F7F7F7FULL) >> 7);
        neighbor |= ((place & 0x7F7F7F7F7F7F7F7FULL) << 9) | ((place & 0xFEFEFEFEFEFEFEFEULL) >> 9);
        cout << hex << neighbor << "ULL, ";
        if (cell % 8 == 7)
            cout << endl;
    }
    return 0;
}