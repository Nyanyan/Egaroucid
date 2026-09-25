/*
    Egaroucid Project

    @file info.hpp
        Egaroucid for Console's software information
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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

#define EGAROUCID_DATE "2021-2026"
#define EGAROUCID_URL "https://www.egaroucid.nyanyan.dev/"
#define EGAROUCID_AUTHOR "Takuto Yamana"
#define EGAROUCID_LICENSE "GPL-3.0 license"