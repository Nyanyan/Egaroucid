/*
    Egaroucid Project

    @file util.hpp
        GUI Utility
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <string>
#include <vector>
#include "const/gui_common.hpp"

std::string get_extension(std::string file) {
    std::string res;
    bool dot_found = false;
    for (int i = (int)file.size() - 1; i >= 0; --i) {
        if (file[i] == '.') {
            dot_found = true;
            break;
        }
        res.insert(0, {file[i]});
    }
    if (dot_found) {
        return res;
    }
    return "";
}

std::string get_transcript(Graph_resources graph_resources, History_elem current_history_elem) {
    std::string transcript;
    int inspect_switch_n_discs = INF;
    if (graph_resources.branch == GRAPH_MODE_INSPECT) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT].size()) {
            inspect_switch_n_discs = graph_resources.nodes[GRAPH_MODE_INSPECT][0].board.n_discs();
        } else {
            std::cerr << "no node found in inspect mode" << std::endl;
        }
    }
    std::cerr << inspect_switch_n_discs << std::endl;
    for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_NORMAL]) {
        if (history_elem.board.n_discs() + 1 >= inspect_switch_n_discs || history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
            break;
        }
        if (history_elem.next_policy != -1) {
            transcript += idx_to_coord(history_elem.next_policy);
        }
    }
    if (inspect_switch_n_discs != INF) {
        if (graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy != -1) {
            transcript += idx_to_coord(graph_resources.nodes[GRAPH_MODE_INSPECT][0].policy);
        }
        for (History_elem& history_elem : graph_resources.nodes[GRAPH_MODE_INSPECT]) {
            if (history_elem.board.n_discs() >= current_history_elem.board.n_discs()) {
                break;
            }
            if (history_elem.next_policy != -1) {
                transcript += idx_to_coord(history_elem.next_policy);
            }
        }
    }
    return transcript;
}