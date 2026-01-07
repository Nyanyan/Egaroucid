/*
    Egaroucid Project

    @file version.hpp
        version information
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <string>

/*
    @brief version settings
*/
#define EGAROUCID_ENGINE_VERSION "7.8"
#define USE_BETA_VERSION false


// OS
#ifdef _WIN64 
    #define EGAROUCID_OS (std::string)"Windows"
#elif defined _WIN32
    #define EGAROUCID_OS (std::string)"Windows"
#elif defined __APPLE__
    #define EGAROUCID_OS (std::string)"macOS"
#else
    #define EGAROUCID_OS (std::string)"Linux"
#endif

// CPU type
#if USE_ARM
    #if USE_64_BIT
        #define EGAROUCID_CPU (std::string)"ARM64"
    #else
        #define EGAROUCID_CPU (std::string)"ARM"
    #endif
#else
    #if USE_64_BIT
        #if USE_AMD
            #define EGAROUCID_CPU (std::string)"x64 (AMD)"
        #else
            #define EGAROUCID_CPU (std::string)"x64"
        #endif
    #else
        #if USE_AMD
            #define EGAROUCID_CPU (std::string)"x86 (AMD)"
        #else
            #define EGAROUCID_CPU (std::string)"x86"
        #endif
    #endif
#endif

// revision
#if USE_SIMD
    #if USE_AVX512
        #define EGAROUCID_REVISION (std::string)"AVX512"
    #else
        #define EGAROUCID_REVISION (std::string)"SIMD"
    #endif
#else
    #define EGAROUCID_REVISION (std::string)"Generic"
#endif

// compiler
#ifdef __clang_version__
    #define EGAROUCID_COMPILER (std::string)"Clang"
#elif defined __GNUC__
    #define EGAROUCID_COMPILER (std::string)"GCC"
#elif defined _MSC_VER
    #define EGAROUCID_COMPILER (std::string)"MSVC"
#else
    #define EGAROUCID_COMPILER (std::string)"Unknown Compiler"
#endif

#define EGAROUCID_ENGINE_ENV_VERSION (EGAROUCID_OS + " " + EGAROUCID_CPU + " " + EGAROUCID_REVISION + " (" + EGAROUCID_COMPILER + ")")

#ifdef BUILD_TZ_OFFSET
    #define EGAROUCID_BUILD_DATETIME ((std::string)__DATE__ + " " + (std::string)__TIME__ + " UTC" + BUILD_TZ_OFFSET)
#else
    #define EGAROUCID_BUILD_DATETIME ((std::string)__DATE__ + " " + (std::string)__TIME__)
#endif
