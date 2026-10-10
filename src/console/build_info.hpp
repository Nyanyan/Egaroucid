/*
    Egaroucid Project

    @file build_info.hpp
        Build identification written to the log at startup
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include "console_common.hpp"
#include "../engine/ai.hpp"

/*
    Every log should name the executable that produced it, because test
    builds are often written over the same file name.  Pass the source
    revision when building, for example in PowerShell or bash:
        clang++ ... -DEGAROUCID_BUILD_COMMIT=$(git describe --always --dirty --abbrev=9)
    The value is written without quotes.  Without it the commit is logged as
    "unknown", and the SHA-256 of the running executable still identifies the
    binary.  The command line is not logged: it contains the GGS password.
*/
#define BUILD_INFO_STRINGIFY_RAW(x) #x
#define BUILD_INFO_STRINGIFY(x) BUILD_INFO_STRINGIFY_RAW(x)

#ifdef EGAROUCID_BUILD_COMMIT
    #define BUILD_INFO_COMMIT BUILD_INFO_STRINGIFY(EGAROUCID_BUILD_COMMIT)
#else
    #define BUILD_INFO_COMMIT "unknown"
#endif

class Build_info_sha256 {
    public:
        Build_info_sha256() {
            state = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                     0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        }

        void update(const unsigned char *data, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                block[block_size++] = data[i];
                if (block_size == 64) {
                    transform();
                    block_size = 0;
                }
            }
            total_bytes += size;
        }

        std::string hex_digest() {
            const uint64_t total_bits = total_bytes * 8;
            const unsigned char pad_start = 0x80;
            const unsigned char zero = 0x00;
            update(&pad_start, 1);
            while (block_size != 56) {
                update(&zero, 1);
            }
            unsigned char length[8];
            for (int i = 0; i < 8; ++i) {
                length[i] = (unsigned char)(total_bits >> (56 - 8 * i));
            }
            update(length, 8);
            static const char *digits = "0123456789abcdef";
            std::string res;
            for (uint32_t word : state) {
                for (int shift = 28; shift >= 0; shift -= 4) {
                    res += digits[(word >> shift) & 0xF];
                }
            }
            return res;
        }

    private:
        std::array<uint32_t, 8> state;
        std::array<unsigned char, 64> block{};
        size_t block_size = 0;
        uint64_t total_bytes = 0;

        static uint32_t rotr(uint32_t x, int n) {
            return (x >> n) | (x << (32 - n));
        }

        void transform() {
            static const uint32_t k[64] = {
                0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
                0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
                0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
                0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
                0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
                0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
                0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
            };
            uint32_t w[64];
            for (int i = 0; i < 16; ++i) {
                w[i] = ((uint32_t)block[4 * i] << 24) | ((uint32_t)block[4 * i + 1] << 16) |
                       ((uint32_t)block[4 * i + 2] << 8) | (uint32_t)block[4 * i + 3];
            }
            for (int i = 16; i < 64; ++i) {
                const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
            uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
            for (int i = 0; i < 64; ++i) {
                const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                const uint32_t ch = (e & f) ^ (~e & g);
                const uint32_t t1 = h + s1 + ch + k[i] + w[i];
                const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const uint32_t t2 = s0 + maj;
                h = g; g = f; f = e; e = d + t1;
                d = c; c = b; b = a; a = t1 + t2;
            }
            state[0] += a; state[1] += b; state[2] += c; state[3] += d;
            state[4] += e; state[5] += f; state[6] += g; state[7] += h;
        }
};

inline std::string build_info_sha256_hex(const std::string &data) {
    Build_info_sha256 sha;
    sha.update((const unsigned char*)data.data(), data.size());
    return sha.hex_digest();
}

// SHA-256 of a file, or an empty string when it cannot be read.
inline std::string build_info_sha256_file(const std::filesystem::path &path, uint64_t *n_bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return "";
    }
    Build_info_sha256 sha;
    std::vector<char> buf(1 << 16);
    uint64_t total = 0;
    while (ifs) {
        ifs.read(buf.data(), (std::streamsize)buf.size());
        const std::streamsize got = ifs.gcount();
        if (got > 0) {
            sha.update((const unsigned char*)buf.data(), (size_t)got);
            total += (uint64_t)got;
        }
    }
    if (ifs.bad()) {
        return "";
    }
    if (n_bytes != nullptr) {
        *n_bytes = total;
    }
    return sha.hex_digest();
}

inline std::filesystem::path build_info_executable_path() {
#if defined(_WIN64) || defined(_WIN32)
    std::vector<wchar_t> buf(MAX_PATH + 1);
    for (;;) {
        const DWORD n = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
        if (n == 0) {
            return {};
        }
        if (n < buf.size()) {
            return std::filesystem::path(std::wstring(buf.data(), n));
        }
        buf.resize(buf.size() * 2);
    }
#elif defined(__APPLE__)
    return {};
#else
    std::error_code ec;
    std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path() : p;
#endif
}

#define BUILD_INFO_PRINT_MACRO(os, name) build_info_print_macro(os, #name, BUILD_INFO_STRINGIFY(name), #name)

// A macro that is not defined stringifies to its own name; a macro defined
// without a value stringifies to an empty string.
inline void build_info_print_macro(std::ostream &os, const char *name, const char *value, const char *self) {
    os << " " << name << "=";
    if (std::string(value) == self) {
        os << "undef";
    } else if (value[0] == '\0') {
        os << "defined";
    } else {
        os << value;
    }
}

inline std::string build_info_compiler() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

// Root endgame MPC cap of the 3-minute profile as empties ranges, e.g.
// "1-35:100 36-40:99.9 ... 49-60:88".
inline std::string build_info_endgame_mpc_cap() {
    std::string res;
    int start = 1;
    for (int e = 1; e <= HW2 - 4; ++e) {
        const bool last = e == HW2 - 4;
        if (last || ai_endgame_max_mpc_level(e + 1) != ai_endgame_max_mpc_level(e)) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s%d-%d:%g", res.empty() ? "" : " ", start, e,
                          SELECTIVITY_PERCENTAGE[ai_endgame_max_mpc_level(e)]);
            res += buf;
            start = e + 1;
        }
    }
    return res;
}

/*
    Lines written at startup (all begin with "build "):
        build commit <revision or unknown>
        build compiled <date> <time> compiler <name and version>
        build flags NAME=value ...        (profile and A/B switches)
        build ggs plan NAME=value ...     (tournament builds only)
        build endgame mpc cap <ranges>
        build executable <path> bytes <n> sha256 <hex>
*/
inline void print_build_info(std::ostream &os) {
    os << "build commit " << BUILD_INFO_COMMIT << std::endl;
    os << "build compiled " << __DATE__ << " " << __TIME__ << " compiler " << build_info_compiler() << std::endl;

    os << "build flags";
    BUILD_INFO_PRINT_MACRO(os, INCLUDE_GGS);
    BUILD_INFO_PRINT_MACRO(os, IS_GGS_TOURNAMENT);
    BUILD_INFO_PRINT_MACRO(os, IS_GGS_3MIN);
    BUILD_INFO_PRINT_MACRO(os, HAS_NO_AVX2);
    BUILD_INFO_PRINT_MACRO(os, HAS_AVX512);
    BUILD_INFO_PRINT_MACRO(os, HAS_AMD_PROCESSOR);
    BUILD_INFO_PRINT_MACRO(os, HAS_ARM_PROCESSOR);
    BUILD_INFO_PRINT_MACRO(os, HAS_32_BIT_OS);
    BUILD_INFO_PRINT_MACRO(os, USE_CLUSTERED_TT);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_EARLY_ENDGAME_SCHEDULE);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_PONDER_MAIN_SCHEDULE);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_PONDER_FOCUS);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_PONDER_FOCUS_MARGIN);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_PONDER_VISIT_LOG);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_GGS_SELFPLAY_RESOLVE);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_GGS_EARLY_PAIR_EXACT_NWS);
    BUILD_INFO_PRINT_MACRO(os, EGAROUCID_GGS_MATCH_REVALIDATE_SWITCH_MARGIN);
    os << std::endl;

#if IS_GGS_TOURNAMENT
    os << "build ggs plan";
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_ENABLED);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_FIRST_MOVE_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_OPENING_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_MIDGAME_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_PRE_COMPLETE_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_32_TO_29_WEIGHT);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_LATE_RESERVE_MSEC);
    BUILD_INFO_PRINT_MACRO(os, TIME_MANAGEMENT_GGS_PLAN_MOVE_OVERHEAD_MSEC);
    os << std::endl;
#endif

    os << "build endgame mpc cap " << build_info_endgame_mpc_cap() << std::endl;

    const std::filesystem::path exe = build_info_executable_path();
    uint64_t n_bytes = 0;
    const std::string sha = exe.empty() ? "" : build_info_sha256_file(exe, &n_bytes);
    os << "build executable " << (exe.empty() ? std::string("unknown") : exe.string())
       << " bytes " << n_bytes << " sha256 " << (sha.empty() ? std::string("unknown") : sha) << std::endl;
}
