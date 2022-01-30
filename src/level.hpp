#pragma once
#include <iostream>

using namespace std;

void get_level(int level, int n_moves, int *depth1, int *depth2, bool *use_mpc, double *mpct){
    constexpr double mpct_73 = 0.61;
    constexpr double mpct_87 = 1.13;
    constexpr double mpct_95 = 1.64;
    constexpr double mpct_98 = 2.05;
    constexpr double mpct_99 = 2.33;
    switch (level)
    {
    case 0:
        *depth1 = 0;
        *depth2 = 0;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 1:
        *depth1 = 1;
        *depth2 = 2;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 2:
        *depth1 = 2;
        *depth2 = 4;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 3:
        *depth1 = 3;
        *depth2 = 6;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 4:
        *depth1 = 4;
        *depth2 = 8;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 5:
        *depth1 = 5;
        *depth2 = 10;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 6:
        *depth1 = 6;
        *depth2 = 12;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 7:
        *depth1 = 7;
        *depth2 = 14;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 8:
        *depth1 = 8;
        *depth2 = 16;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 9:
        *depth1 = 9;
        *depth2 = 18;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 10:
        *depth1 = 10;
        *depth2 = 20;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    case 11:
        *depth1 = 11;
        *depth2 = 24;
        if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 12:
        *depth1 = 12;
        *depth2 = 24;
        if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 13:
        *depth1 = 13;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 14:
        *depth1 = 14;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 15:
        *depth1 = 15;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 16:
        *depth1 = 16;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 17:
        *depth1 = 17;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 18:
        *depth1 = 18;
        *depth2 = 27;
        if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 36){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 19:
        *depth1 = 19;
        *depth2 = 30;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 20:
        *depth1 = 20;
        *depth2 = 30;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 21:
        *depth1 = 21;
        *depth2 = 30;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 22:
        *depth1 = 22;
        *depth2 = 33;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 23:
        *depth1 = 23;
        *depth2 = 33;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 24:
        *depth1 = 24;
        *depth2 = 33;
        if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 39){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 25:
        *depth1 = 25;
        *depth2 = 33;
        if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 26:
        *depth1 = 26;
        *depth2 = 33;
        if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 27:
        *depth1 = 27;
        *depth2 = 33;
        if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 28:
        *depth1 = 28;
        *depth2 = 36;
        if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 29:
        *depth1 = 29;
        *depth2 = 36;
        if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 33){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 30:
        *depth1 = 30;
        *depth2 = 36;
        if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 31:
        *depth1 = 31;
        *depth2 = 36;
        if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_98;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 32:
        *depth1 = 32;
        *depth2 = 39;
        if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 33:
        *depth1 = 33;
        *depth2 = 39;
        if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 34:
        *depth1 = 34;
        *depth2 = 39;
        if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 35:
        *depth1 = 35;
        *depth2 = 39;
        if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 36:
        *depth1 = 36;
        *depth2 = 45;
        if (n_moves < 18){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 30){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 37:
        *depth1 = 37;
        *depth2 = 46;
        if (n_moves < 17){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 20){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 23){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 26){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 29){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 38:
        *depth1 = 38;
        *depth2 = 47;
        if (n_moves < 16){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 19){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 22){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 25){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 28){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 39:
        *depth1 = 39;
        *depth2 = 48;
        if (n_moves < 15){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 18){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 27){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 40:
        *depth1 = 40;
        *depth2 = 49;
        if (n_moves < 14){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 17){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 20){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 23){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 26){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 41:
        *depth1 = 41;
        *depth2 = 50;
        if (n_moves < 13){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 16){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 19){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 22){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 25){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 42:
        *depth1 = 42;
        *depth2 = 51;
        if (n_moves < 12){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 15){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 18){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 24){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 43:
        *depth1 = 43;
        *depth2 = 52;
        if (n_moves < 11){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 14){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 17){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 20){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 23){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 44:
        *depth1 = 44;
        *depth2 = 53;
        if (n_moves < 10){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 13){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 16){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 19){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 22){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 45:
        *depth1 = 45;
        *depth2 = 54;
        if (n_moves < 9){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 12){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 15){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 18){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 21){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 46:
        *depth1 = 46;
        *depth2 = 55;
        if (n_moves < 8){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 11){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 14){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 17){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 20){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 47:
        *depth1 = 47;
        *depth2 = 56;
        if (n_moves < 7){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 10){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 13){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 16){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 19){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 48:
        *depth1 = 48;
        *depth2 = 57;
        if (n_moves < 6){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 9){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 12){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 15){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 18){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 49:
        *depth1 = 49;
        *depth2 = 58;
        if (n_moves < 5){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 8){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 11){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 14){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 17){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 50:
        *depth1 = 50;
        *depth2 = 59;
        if (n_moves < 4){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 7){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 10){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 13){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 16){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 51:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 3){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 6){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 9){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 12){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 15){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 52:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 2){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 5){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 8){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 11){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 14){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 53:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 1){
            *use_mpc = true;
            *mpct = mpct_73;
        } else if (n_moves < 4){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 7){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 10){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 13){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 54:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 3){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 6){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 9){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 12){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 55:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 2){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 5){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 8){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 11){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 56:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 1){
            *use_mpc = true;
            *mpct = mpct_87;
        } else if (n_moves < 4){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 7){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 10){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 57:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 3){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 6){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 9){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 58:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 2){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 5){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 8){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 59:
        *depth1 = -1;
        *depth2 = 60;
        if (n_moves < 1){
            *use_mpc = true;
            *mpct = mpct_95;
        } else if (n_moves < 4){
            *use_mpc = true;
            *mpct = mpct_98;
        } else if (n_moves < 7){
            *use_mpc = true;
            *mpct = mpct_99;
        } else{
            *use_mpc = false;
            *mpct = 0.0;
        }
        break;
    case 60:
        *depth1 = -1;
        *depth2 = 60;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    default:
        *depth1 = 0;
        *depth2 = 0;
        *use_mpc = false;
        *mpct = 0.0;
        break;
    }
}

void get_level(int level, int n_moves, int *depth1, int *depth2){
    bool dammy1;
    double dammy2;
    get_level(level, n_moves, depth1, depth2, &dammy1, &dammy2);
}

void get_level(int level, int n_moves, bool *use_mpc, double *mpct){
    int dammy1, dammy2;
    get_level(level, n_moves, &dammy1, &dammy2, use_mpc, mpct);
}