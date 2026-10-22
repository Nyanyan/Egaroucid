/*
    Egaroucid Project

    @file large_pages.hpp
        Large page (2 MiB) memory for the big tables
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    -DEGAROUCID_LARGE_PAGES puts the transposition table (EGAROUCID_LARGE_PAGES_TT) and the evaluation
    tables (EGAROUCID_LARGE_PAGES_EVAL: pattern tables, the unflipped-disc table of the feature update, move
    ordering models, NNUE weights) in large pages; either part can also be built alone. A 4 KiB page covers
    128 transposition table entries, so nearly every table lookup misses the TLB (the CPU's cache of address
    translations); a 2 MiB page covers 512 times more.

    Windows maps large pages only for a user who holds "Lock pages in memory" (SeLockMemoryPrivilege),
    and the pages must be found as free 2 MiB blocks of physical memory, so a large request can fail
    when memory is fragmented even if enough is free. There is no fallback to normal pages: a failure
    stops the engine, so a build never runs in a mode other than the one it was built for.
*/

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "setting.hpp"

#if defined(EGAROUCID_LARGE_PAGES_TT) || defined(EGAROUCID_LARGE_PAGES_EVAL)
#ifndef _WIN32
#error "large pages (EGAROUCID_LARGE_PAGES) are implemented only for Windows"
#endif
#include <cstdlib>
#include <iostream>
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#if INCLUDE_GGS
    #include <winsock2.h> // before windows.h, as in console_common.hpp
#endif
#include <windows.h>

/*
    @brief size of a large page; enables "Lock pages in memory" in this process on the first call
*/
inline size_t large_page_size() {
    static const size_t size = [] {
        HANDLE token;
        TOKEN_PRIVILEGES privileges{};
        privileges.PrivilegeCount = 1;
        privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
            std::cerr << "[ERROR] [FATAL] large pages: cannot open the process token (Windows error " << GetLastError() << ")" << std::endl;
            std::exit(1);
        }
        const bool found = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &privileges.Privileges[0].Luid);
        // AdjustTokenPrivileges succeeds without the privilege and reports it only in GetLastError
        const bool enabled = found && AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;
        CloseHandle(token);
        if (!enabled) {
            std::cerr << "[ERROR] [FATAL] large pages: this user does not hold \"Lock pages in memory\" (SeLockMemoryPrivilege). "
                         "Grant it in secpol.msc (Local Policies > User Rights Assignment) and sign in again, or build without -DEGAROUCID_LARGE_PAGES" << std::endl;
            std::exit(1);
        }
        const size_t page = GetLargePageMinimum();
        if (page == 0) {
            std::cerr << "[ERROR] [FATAL] large pages: this processor or Windows has no large pages" << std::endl;
            std::exit(1);
        }
        return page;
    }();
    return size;
}

/*
    @brief allocate zero-filled memory in large pages (stops the engine on failure)

    @param bytes                size (rounded up to whole large pages)
    @param what                 name of the table for the error message
    @return memory aligned to the large page size
*/
inline void *large_page_alloc(size_t bytes, const char *what) {
    const size_t page = large_page_size();
    const size_t rounded = (bytes + page - 1) / page * page;
    void *memory = VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (memory == nullptr) {
        std::cerr << "[ERROR] [FATAL] large pages: cannot get " << rounded / 1024 / 1024 << " MiB for the " << what
                  << " (Windows error " << GetLastError() << "). Free physical memory is too small or too fragmented: "
                     "use a smaller -hash, close other programs or restart Windows, or build without -DEGAROUCID_LARGE_PAGES" << std::endl;
        std::exit(1);
    }
    return memory;
}

inline void large_page_free(void *memory) {
    if (memory != nullptr && !VirtualFree(memory, 0, MEM_RELEASE)) {
        std::cerr << "[ERROR] [FATAL] large pages: cannot free a table (Windows error " << GetLastError() << ")" << std::endl;
        std::exit(1);
    }
}
#endif

#ifdef EGAROUCID_LARGE_PAGES_EVAL
template <typename T>
struct Large_page_allocator {
    using value_type = T;
    Large_page_allocator() = default;
    template <typename U>
    Large_page_allocator(const Large_page_allocator<U> &) noexcept {}
    T *allocate(size_t n) { return static_cast<T*>(large_page_alloc(n * sizeof(T), "evaluation table")); }
    void deallocate(T *p, size_t) noexcept { large_page_free(p); }
    template <typename U>
    bool operator==(const Large_page_allocator<U> &) const noexcept { return true; }
    template <typename U>
    bool operator!=(const Large_page_allocator<U> &) const noexcept { return false; }
};
// pattern tables that are allocated as vectors (move ordering models)
using Eval_table_vector = std::vector<int16_t, Large_page_allocator<int16_t>>;
#else
using Eval_table_vector = std::vector<int16_t>;
#endif
