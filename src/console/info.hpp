/*
    Egaroucid Project

    @file info.hpp
        Egaroucid's software information
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "./../engine/setting.hpp"

#define EGAROUCID_NAME "Egaroucid for Console"
#define EGAROUCID_CONSOLE_VERSION "0"
#if USE_BETA_VERSION
    const std::string EGAROUCID_VERSION = EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_CONSOLE_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION + (std::string)" beta";
#else
    const std::string EGAROUCID_VERSION = EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_CONSOLE_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION;
#endif

#define EGAROUCID_DATE "2021-2022"
#define EGAROUCID_URL "https://www.egaroucid.nyanyan.dev/"
#define EGAROUCID_AUTHOR "Takuto Yamana (a.k.a. Nyanyan)"
#define EGAROUCID_LICENSE "GPL-3.0 license"