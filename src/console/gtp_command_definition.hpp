/*
    Egaroucid Project

    @file gtp_command_definition.hpp
        Definition of commands for GTP (Gp Text Protocol)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <string>
#include <vector>
#include "command_definition.hpp"

#define N_GTP_COMMANDS 2

#define GTP_CMD_ID_QUIT 0
#define GTP_CMD_ID_VERSION 1

const Command_info gtp_command_data[N_GTP_COMMANDS] = {
    {GTP_CMD_ID_QUIT,           {"quit"},                                                   "",                 "Quit"},
    {GTP_CMD_ID_VERSION,        {"protocol_version"},                                       "",                 "See version"}
};