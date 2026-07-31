/*
    Egaroucid Project

    @file graph_error.hpp
        Current-position graph error calculation
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <vector>

struct Graph_error_sample {
    int n_discs;
    int value;
    bool value_available;
};

struct Graph_error_result {
    int black{ 0 };
    int white{ 0 };
    bool complete{ true };
};

inline std::vector<Graph_error_sample> get_current_error_samples(
    const std::vector<Graph_error_sample>& main_line,
    const std::vector<Graph_error_sample>& inspect_line,
    bool use_inspect_line,
    int current_n_discs,
    int start_n_discs = -1
) {
    std::vector<Graph_error_sample> result;
    int inspect_start_n_discs = current_n_discs + 1;
    if (use_inspect_line && !inspect_line.empty()) {
        inspect_start_n_discs = inspect_line.front().n_discs;
    }

    for (const Graph_error_sample& sample : main_line) {
        if (sample.n_discs < start_n_discs) {
            continue;
        }
        if (sample.n_discs > current_n_discs || sample.n_discs >= inspect_start_n_discs) {
            break;
        }
        result.emplace_back(sample);
    }
    if (use_inspect_line) {
        for (const Graph_error_sample& sample : inspect_line) {
            if (sample.n_discs < start_n_discs) {
                continue;
            }
            if (sample.n_discs > current_n_discs) {
                break;
            }
            result.emplace_back(sample);
        }
    }
    return result;
}

inline Graph_error_result calc_current_error(
    const std::vector<Graph_error_sample>& samples,
    int current_n_discs
) {
    Graph_error_result result;
    if (samples.empty()) {
        return result;
    }

    int previous_value = samples.front().value_available ? samples.front().value : 0;
    int previous_n_discs = samples.front().n_discs;
    for (int i = 1; i < static_cast<int>(samples.size()); ++i) {
        const Graph_error_sample& sample = samples[i];
        if (sample.n_discs != previous_n_discs + 1) {
            result.complete = false;
        }
        previous_n_discs = sample.n_discs;
        if (!sample.value_available) {
            result.complete = false;
            continue;
        }

        const int error = sample.value - previous_value;
        result.black += std::max(0, -error);
        result.white += std::max(0, error);
        previous_value = sample.value;
    }
    if (samples.back().n_discs < current_n_discs) {
        result.complete = false;
    }
    return result;
}
