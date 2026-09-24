/*
    Egaroucid Project

    @file console_common.hpp
        Common things
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
//#include <stdlib.h>
#include <string>
#include <filesystem>
#ifdef _WIN64
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #if INCLUDE_GGS
        #include <winsock2.h>
        #include <ws2tcpip.h>
    #endif
    #include <windows.h>
#elif _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #if INCLUDE_GGS
        #include <winsock2.h>
        #include <ws2tcpip.h>
        // #include <windows.h>
    #endif
    #include <windows.h>
#elif __APPLE__ // TBD
#else
    #include <linux/limits.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

#define MODE_HUMAN_AI 0
#define MODE_AI_HUMAN 1
#define MODE_AI_AI 2
#define MODE_HUMAN_HUMAN 3

// Windows 11 may run a process whose window is in the background (or that
// has none) at EcoQoS, which prefers efficiency cores and lower clocks. The
// search always wants full speed, so opt out. The API is resolved at run time
// because it needs Windows 8 or later.
inline void disable_power_throttling() {
#ifdef _WIN32
    struct Power_throttling_state {
        ULONG Version;
        ULONG ControlMask;
        ULONG StateMask;
    };
    using Set_process_information = BOOL (WINAPI *)(HANDLE, int, LPVOID, DWORD);
    constexpr int PROCESS_POWER_THROTTLING_CLASS = 4; // ProcessPowerThrottling
    const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    if (kernel32 == nullptr) {
        return;
    }
    const auto set_process_information = reinterpret_cast<Set_process_information>(
        reinterpret_cast<void *>(GetProcAddress(kernel32, "SetProcessInformation")));
    if (set_process_information == nullptr) {
        return;
    }
    // Version 1; control EXECUTION_SPEED (1) and turn it off.
    Power_throttling_state state{1, 1, 0};
    set_process_information(GetCurrentProcess(), PROCESS_POWER_THROTTLING_CLASS, &state, sizeof(state));
#endif
}

std::string get_parent_path(char raw_path[]) {
    std::filesystem::path p = raw_path;
    //p = std::filesystem::canonical(p);
    std::string res = p.parent_path().string() + "/";
    return res;
}

std::string get_parent_path(wchar_t raw_path[]) {
    std::filesystem::path p = raw_path;
    //p = std::filesystem::canonical(p);
    std::string res = p.parent_path().string() + "/";
    return res;
}

#ifdef _WIN64 // Windows
    std::string get_binary_path() {
        std::string res;
        #ifdef UNICODE
            wchar_t raw_path[MAX_PATH + 1];
        #else
            char raw_path[MAX_PATH + 1];
        #endif
        if (GetModuleFileName(NULL, raw_path, MAX_PATH))
            res = get_parent_path(raw_path);
        return res;
    }
#elif _WIN32
    std::string get_binary_path() {
        std::string res;
        #ifdef UNICODE
            wchar_t raw_path[MAX_PATH + 1];
        #else
            char raw_path[MAX_PATH + 1];
        #endif
        if (GetModuleFileName(NULL, raw_path, MAX_PATH))
            res = get_parent_path(raw_path);
        return res;
    }
#elif __APPLE__ // Mac TBD
    std::string get_binary_path() {
        std::string res;
        return res;
    }
#else // Linux
    std::string get_binary_path() {
        char raw_path[PATH_MAX + 1];
        const size_t LINKSIZE = 100;
        char link[LINKSIZE];
        snprintf(link, LINKSIZE, "/proc/%d/exe", getpid());
        ssize_t e = readlink(link, raw_path, PATH_MAX);
        if (e == -1) {
            std::cerr << "[ERROR] can't get binary path. You can ignore this error." << std::endl;
            return "";
        }
        std::string res = get_parent_path(raw_path);
        return res;
    }
#endif
