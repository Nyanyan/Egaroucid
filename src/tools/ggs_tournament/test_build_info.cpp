/*
    Egaroucid Project

    Regression tests for the "build ..." lines that Console writes to the
    search log at startup (src/console/build_info.hpp).

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DINCLUDE_GGS -DIS_GGS_TOURNAMENT -DIS_GGS_3MIN \
            test_build_info.cpp -o test_build_info.exe -lws2_32
*/

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "../../engine/engine_all.hpp"
#include "../../console/console_all.hpp"

#define TEST_BUILD_INFO_EMPTY
#define TEST_BUILD_INFO_VALUE 2500.0
#define TEST_BUILD_INFO_REVISION 2e9e898ba-dirty

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(const std::string &actual, const std::string &expected, const std::string &what) {
    require(actual == expected, what + ": expected \"" + expected + "\", got \"" + actual + "\"");
}

// FIPS 180-2 test vectors.
void test_sha256_vectors() {
    require_equal(build_info_sha256_hex(""),
                  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    require_equal(build_info_sha256_hex("abc"),
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    require_equal(build_info_sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "two blocks");
    require_equal(build_info_sha256_hex(std::string(1000000, 'a')),
                  "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "million a");
}

void test_sha256_file() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "egaroucid_test_build_info.bin";
    std::string data;
    for (int i = 0; i < 200000; ++i) {
        data += (char)(i * 7 + 3);
    }
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(data.data(), (std::streamsize)data.size());
    }
    uint64_t n_bytes = 0;
    const std::string file_hash = build_info_sha256_file(path, &n_bytes);
    std::filesystem::remove(path);
    require_equal(file_hash, build_info_sha256_hex(data), "file hash");
    require(n_bytes == data.size(), "file size");
    require(build_info_sha256_file(path, nullptr).empty(), "missing file gives an empty hash");
}

void test_macro_values() {
    std::ostringstream os;
    BUILD_INFO_PRINT_MACRO(os, TEST_BUILD_INFO_VALUE);
    BUILD_INFO_PRINT_MACRO(os, TEST_BUILD_INFO_EMPTY);
    BUILD_INFO_PRINT_MACRO(os, TEST_BUILD_INFO_NOT_DEFINED);
    BUILD_INFO_PRINT_MACRO(os, IS_GGS_3MIN);
    require_equal(os.str(),
                  " TEST_BUILD_INFO_VALUE=2500.0 TEST_BUILD_INFO_EMPTY=defined"
                  " TEST_BUILD_INFO_NOT_DEFINED=undef IS_GGS_3MIN=1", "macro values");
    // An unquoted revision passed with -D keeps its spelling.
    require_equal(BUILD_INFO_STRINGIFY(TEST_BUILD_INFO_REVISION), "2e9e898ba-dirty", "revision");
}

void test_endgame_mpc_cap() {
    require_equal(build_info_endgame_mpc_cap(),
                  "1-35:100 36-40:99.9 41-42:99 43-44:98 45-48:93 49-60:88", "3-minute endgame cap");
}

void test_startup_lines() {
    std::ostringstream os;
    print_build_info(os);
    std::istringstream is(os.str());
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(is, line)) {
        lines.push_back(line);
    }
    require(lines.size() == 6, "six build lines, got " + std::to_string(lines.size()));
    require_equal(lines[0], std::string("build commit ") + BUILD_INFO_COMMIT, "commit line");
    require(lines[1].rfind("build compiled ", 0) == 0 && lines[1].find(" compiler ") != std::string::npos, "compiled line");
    require(lines[2].rfind("build flags INCLUDE_GGS=1 IS_GGS_TOURNAMENT=1 IS_GGS_3MIN=1 ", 0) == 0, "flags line: " + lines[2]);
    require(lines[2].find(" EGAROUCID_GGS_SELFPLAY_RESOLVE=0") != std::string::npos, "selfplay flag: " + lines[2]);
    require(lines[3].find(" TIME_MANAGEMENT_GGS_PLAN_EARLY_ENDGAME_WEIGHT=7000.0") != std::string::npos, "plan line: " + lines[3]);
    require(lines[3].find(" TIME_MANAGEMENT_GGS_PLAN_32_TO_31_WEIGHT=2500.0") != std::string::npos, "plan line: " + lines[3]);
    require(lines[3].find(" TIME_MANAGEMENT_GGS_PLAN_30_TO_29_WEIGHT=1000.0") != std::string::npos, "plan line: " + lines[3]);
    require_equal(lines[4], "build endgame mpc cap 1-35:100 36-40:99.9 41-42:99 43-44:98 45-48:93 49-60:88", "cap line");
    require(lines[5].find("[ERROR]") == std::string::npos, "no error marker");

    // The executable line names this test program and its own hash.
    const std::filesystem::path exe = build_info_executable_path();
    require(!exe.empty(), "executable path");
    uint64_t n_bytes = 0;
    const std::string sha = build_info_sha256_file(exe, &n_bytes);
    require(sha.size() == 64 && n_bytes > 0, "executable hash");
    require_equal(lines[5], "build executable " + exe.string() + " bytes " + std::to_string(n_bytes) + " sha256 " + sha,
                  "executable line");
}

}  // namespace

int main() {
    try {
        test_sha256_vectors();
        test_sha256_file();
        test_macro_values();
        test_endgame_mpc_cap();
        test_startup_lines();
    } catch (const std::exception &e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "all build info tests passed" << std::endl;
    return 0;
}
