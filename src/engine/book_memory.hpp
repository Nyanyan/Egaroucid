/*
    Egaroucid Project

    @file book_memory.hpp
        Memory budget for rebuilding large book-table segments
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    // The SDK's core headers need the target macros normally set by Windows.h.
    // Include only system information declarations: importing GDI here would
    // collide with GUI types such as RoundRect, or disable later GDI includes.
    #if defined(_MSC_VER)
        #if (defined(_M_AMD64) || defined(_M_ARM64EC)) && !defined(_AMD64_)
            #define _AMD64_
        #endif
        #if defined(_M_IX86) && !defined(_X86_)
            #define _X86_
        #endif
        #if defined(_M_ARM) && !defined(_ARM_)
            #define _ARM_
        #endif
        #if defined(_M_ARM64) && !defined(_ARM64_)
            #define _ARM64_
        #endif
        #if defined(_M_ARM64EC) && !defined(_ARM64EC_)
            #define _ARM64EC_
        #endif
    #endif
    #include <sysinfoapi.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#endif

namespace book_table_memory {
    struct Memory_status {
        bool known = false;
        uint64_t available_bytes = 0;
        uint64_t total_physical_bytes = 0;
    };

    // Best-effort snapshot. If unavailable, try the usual allocation first.
    inline Memory_status query_status() {
        Memory_status result;
#if defined(_WIN32)
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) {
            result.known = memory.ullTotalPhys != 0;
            result.total_physical_bytes = memory.ullTotalPhys;
            result.available_bytes = std::min<uint64_t>(memory.ullAvailPhys,
                std::min<uint64_t>(memory.ullAvailPageFile, memory.ullAvailVirtual));
        }
#elif defined(__linux__)
        FILE *file = std::fopen("/proc/meminfo", "r");
        if (file) {
            bool have_total = false, have_available = false;
            char line[256];
            unsigned long long kib;
            while (std::fgets(line, sizeof(line), file)) {
                if (std::sscanf(line, "MemTotal: %llu kB", &kib) == 1 && kib <= UINT64_MAX / 1024) {
                    result.total_physical_bytes = uint64_t(kib) * 1024;
                    have_total = kib != 0;
                } else if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1 && kib <= UINT64_MAX / 1024) {
                    result.available_bytes = uint64_t(kib) * 1024;
                    have_available = true;
                }
            }
            std::fclose(file);
            result.known = have_total && have_available;
        }
#elif defined(__APPLE__)
        uint64_t total = 0;
        size_t length = sizeof(total);
        if (sysctlbyname("hw.memsize", &total, &length, nullptr, 0) == 0 && total) {
            const mach_port_t host = mach_host_self();
            vm_size_t page_size = 0;
            vm_statistics64_data_t memory{};
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            if (host_page_size(host, &page_size) == KERN_SUCCESS &&
                host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&memory), &count) == KERN_SUCCESS) {
                result.known = true;
                result.total_physical_bytes = total;
                result.available_bytes = std::min(total,
                    (uint64_t(memory.free_count) + memory.inactive_count) * page_size);
            }
            mach_port_deallocate(mach_task_self(), host);
        }
#endif
        return result;
    }

    // Count whole destination arrays while old and new arrays coexist. Keeping
    // these reservations until copying finishes is conservative when the OS
    // already includes some destination pages in its memory usage snapshot.
    inline std::atomic<uint64_t> pending_rebuild_bytes{0};

    class Reservation {
        uint64_t bytes;

    public:
        explicit Reservation(uint64_t bytes_) : bytes(bytes_) {
            pending_rebuild_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
        ~Reservation() {
            pending_rebuild_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        bool under_pressure(const Memory_status &status) const {
            if (!status.known) return false;
            // Leave 10% of physical RAM (at most 1 GiB) for the OS and other work.
            const uint64_t headroom = std::min<uint64_t>(status.total_physical_bytes / 10, 1ULL << 30);
            return status.available_bytes < headroom ||
                pending_rebuild_bytes.load(std::memory_order_relaxed) > status.available_bytes - headroom;
        }

        void reduce_to(uint64_t smaller_bytes) {
            if (smaller_bytes < bytes) {
                pending_rebuild_bytes.fetch_sub(bytes - smaller_bytes, std::memory_order_relaxed);
                bytes = smaller_bytes;
            }
        }
    };
}
