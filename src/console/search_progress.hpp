/*
    Egaroucid Project

    @file search_progress.hpp
        Search summaries on standard output
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include "../engine/search_progress.hpp"

inline std::string search_progress_summary(const Search_progress &progress) {
    std::ostringstream summary;
    summary << "search depth " << progress.depth << " mpc " << progress.probability
            << "% score " << progress.score << " time " << progress.elapsed_msec << " ms";
    return summary.str();
}

inline Search_progress_callback console_search_progress_callback(bool enabled) {
    if (!enabled) {
        return {};
    }
    return [](const Search_progress &progress) {
        static std::mutex output_mutex;
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << search_progress_summary(progress) << std::endl;
    };
}
