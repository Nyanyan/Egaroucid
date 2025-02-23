/*
    Egaroucid Project

    @file ggs.hpp
        Telnet client for Generic Game Server https://skatgame.net/mburo/ggs/
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/
#pragma once
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define GGS_URL "skatgame.net"
#define GGS_PORT 5000

void ggs_client(Options *options) {

}