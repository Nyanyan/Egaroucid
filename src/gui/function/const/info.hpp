/*
    Egaroucid Project

    @file info.hpp
        Egaroucid's software information
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include "./../../../engine/setting.hpp"
#include "settings.hpp"

#define EGAROUCID_GUI_VERSION "1"
#if USE_BETA_VERSION
    #if GUI_PORTABLE_MODE
        const String EGAROUCID_VERSION = Unicode::Widen(EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_GUI_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION + (std::string)" Portable" + (std::string)" beta");
    #else
        const String EGAROUCID_VERSION = Unicode::Widen(EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_GUI_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION + (std::string)" beta");
    #endif
#else
    #if GUI_PORTABLE_MODE
        const String EGAROUCID_VERSION = Unicode::Widen(EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_GUI_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION + (std::string)" Portable");
    #else
        const String EGAROUCID_VERSION = Unicode::Widen(EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_GUI_VERSION + (std::string)" " + EGAROUCID_ENGINE_ENV_VERSION);
    #endif
#endif

const String EGAROUCID_NUM_VERSION = Unicode::Widen(EGAROUCID_ENGINE_VERSION + (std::string)"." + EGAROUCID_GUI_VERSION);

#define EGAROUCID_DATE "2021-2026"
#define EGAROUCID_AUTHOR "Takuto Yamana"