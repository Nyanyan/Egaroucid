/*
    Egaroucid Project

    @file info.hpp
        Egaroucid's software information
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "./../engine/setting.hpp"

#define EGAROUCID_NAME "Egaroucid for Console"
#if USE_AVX2
    #define EGAROUCID_VERSION "6.1.0 AVX2 beta"
#else
    #define EGAROUCID_VERSION "6.1.0 Generic beta"
#endif
#define EGAROUCID_DATE "2021-2022"
#define EGAROUCID_URL "https://www.egaroucid.nyanyan.dev/"
#define EGAROUCID_AUTHOR "Takuto Yamana (a.k.a. Nyanyan)"
#define EGAROUCID_LICENSE "GPL-3.0 license"
