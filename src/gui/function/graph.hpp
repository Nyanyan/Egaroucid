/*
    Egaroucid Project

    @file graph.hpp
        Graph drawing
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <vector>
#include "./../../engine/board.hpp"
#include "language.hpp"
#include "./const/gui_common.hpp"

constexpr Color graph_color = Color(51, 51, 51);
constexpr Color graph_history_color = Palette::White;
constexpr Color graph_fork_color = Palette::Black;
constexpr Color graph_place_color = Palette::White;
constexpr Color graph_history_not_calculated_color = Color(200, 200, 200);
constexpr Color graph_fork_not_calculated_color = Color(65, 65, 65);
constexpr double graph_transparency = 1.0;

constexpr Color color_selectivity[N_GRPAPH_COLOR_TYPES][N_SELECTIVITY_LEVEL] = {
    {
        Color(190, 46, 221),  // purple
        Color(227, 78, 62),   // red
        Color(240, 135, 20),  // oragne
        Color(247, 143, 179), // pink
        Color(82, 62, 212),   // blue
        Color(255, 255, 255), // empty
        Color(51, 161, 255)   // light blue
    },
    {
        Color(245, 58, 58),  // red
        Color(178, 28, 108), // dark purple
        Color(214, 56, 248), // purple
        Color(18, 4, 171),   // blue
        Color(66, 109, 255), // light blue
        Color(255, 255, 255), // empty
        Color(44, 166, 167)  // cyan
    }
};

constexpr bool selectivity_used_display[N_SELECTIVITY_LEVEL] = {1, 1, 1, 1, 1, 0, 1};

constexpr Color midsearch_color = Color(51, 51, 51);
constexpr Color endsearch_color = Palette::White;
constexpr Color level_info_color = Palette::White;
constexpr Color level_prob_color = Palette::White;

constexpr Color graph_rect_color = Palette::White;

struct Graph_loss_elem {
    int ply;
    int v;
};

class Graph {
public:
    int sx;
    int sy;
    int size_x;
    int size_y;
    int resolution;

private:
    int font_size{ 12 };
    int y_max;
    int y_min;
    double dy;
    double dx;

public:
    void draw(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, int n_discs, bool show_graph, int level, Font font, int color_type, bool show_graph_sum_of_loss, bool show_endgame_error) {
        std::vector<std::vector<Graph_loss_elem>> sum_of_loss_nodes1(2);
        std::vector<std::vector<Graph_loss_elem>> sum_of_loss_nodes2(2);
        resolution = GRAPH_RESOLUTION;
        if (show_graph) {
            if (show_graph_sum_of_loss) {
                calc_sum_of_loss_nodes(nodes1, nodes2, sum_of_loss_nodes1, sum_of_loss_nodes2);
                calc_range_sum_of_loss(sum_of_loss_nodes1, sum_of_loss_nodes2);
            } else {
                calc_range(nodes1, nodes2);
            }
            while ((y_max - y_min) / resolution > 8) { // when range is too wide
                resolution *= 2;
            }
            if ((-y_min) % resolution) { // fit y_min/max to resolution
                y_min -= resolution - (-y_min) % resolution;
            }
            if (y_max % resolution) {
                y_max += resolution - y_max % resolution;
            }
        } else {
            if (show_graph_sum_of_loss) {
                y_min = -GRAPH_RESOLUTION;
                y_max = 0;
            } else {
                y_min = -GRAPH_RESOLUTION;
                y_max = GRAPH_RESOLUTION;
            }
        }
        dy = (double)size_y / (y_max - y_min);
        dx = (double)size_x / 60;
        RoundRect round_rect{ sx + GRAPH_RECT_DX, sy + GRAPH_RECT_DY, GRAPH_RECT_WIDTH, GRAPH_RECT_HEIGHT, GRAPH_RECT_RADIUS };
        round_rect.drawFrame(GRAPH_RECT_THICKNESS, graph_rect_color);
        if (!show_endgame_error) { // search probability
            int n_selectivity_level_displayed = 0;
            for (int i = 0; i < N_SELECTIVITY_LEVEL; ++i) {
                if (selectivity_used_display[i]) {
                    ++n_selectivity_level_displayed;
                }
            }
            int info_x = sx + GRAPH_RECT_DX + GRAPH_RECT_WIDTH / 2 - (LEVEL_PROB_WIDTH + LEVEL_INFO_WIDTH * n_selectivity_level_displayed) / 2;
            int info_y = sy + LEVEL_INFO_DY;
            Rect rect_prob{info_x, info_y, LEVEL_PROB_WIDTH, LEVEL_INFO_HEIGHT};
            rect_prob.draw(graph_color);
            font(language.get("info", "probability")).draw(font_size, Arg::center(info_x + LEVEL_PROB_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_prob_color);
            info_x += LEVEL_PROB_WIDTH;
            for (int i = 0; i < N_SELECTIVITY_LEVEL; ++i) {
                if (selectivity_used_display[i]) {
                    Rect rect_selectivity{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
                    rect_selectivity.draw(color_selectivity[color_type][i]);
                    font(Format(SELECTIVITY_PERCENTAGE[i]) + U"%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
                    info_x += LEVEL_INFO_WIDTH;
                }
            }
        } else { // endgame error
            int endgame_error_black = 0, endgame_error_white = 0;
            bool endgame_error_calculated = calc_endgame_error(nodes1, nodes2, &endgame_error_black, &endgame_error_white);
            int endgame_error_cy = sy - 48;
            int endgame_error_cx = sx + GRAPH_RECT_DX + GRAPH_RECT_WIDTH / 2;
            constexpr int ENDGAME_ERROR_DISC_RADIUS = 7;
            font(language.get("display", "graph", "endgame_error")).draw(11, Arg::rightCenter(endgame_error_cx - 73, endgame_error_cy), Palette::White);
            Line(endgame_error_cx, endgame_error_cy - 7, endgame_error_cx, endgame_error_cy + 7).draw(2, graph_color);
            Circle(endgame_error_cx - 60, endgame_error_cy, ENDGAME_ERROR_DISC_RADIUS).draw(Palette::Black);
            Circle(endgame_error_cx + 60, endgame_error_cy, ENDGAME_ERROR_DISC_RADIUS).draw(Palette::White);
            if (endgame_error_calculated) {
                font(-endgame_error_black).draw(16, Arg::rightCenter(endgame_error_cx - 10, endgame_error_cy), Palette::White);
                font(-endgame_error_white).draw(16, Arg::leftCenter(endgame_error_cx + 10, endgame_error_cy), Palette::White);
            } else {
                font(U"-").draw(16, Arg::rightCenter(endgame_error_cx - 10, endgame_error_cy), Palette::White);
                font(U"-").draw(16, Arg::leftCenter(endgame_error_cx + 10, endgame_error_cy), Palette::White);
            }
        }
        // search depth
        bool is_mid_search;
        int depth;
        uint_fast8_t mpc_level;
        double mpct;
        int first_endsearch_n_moves = -1;
        for (int x = 0; x < 60; ++x) {
            int x_coord1 = sx + dx * x;
            int x_coord2 = sx + dx * (x + 1);
            Rect rect{ x_coord1, sy, x_coord2 - x_coord1, size_y };
            get_level(level, x, &is_mid_search, &depth, &mpc_level);
            Color color = color_selectivity[color_type][mpc_level];
            rect.draw(color);
            if (!is_mid_search) {
                if (first_endsearch_n_moves == -1) {
                    first_endsearch_n_moves = x;
                }
                Line{ x_coord1, sy + LEVEL_DEPTH_DY, x_coord2, sy + LEVEL_DEPTH_DY }.draw(3, endsearch_color);
            } else {
                Line{ x_coord1, sy + LEVEL_DEPTH_DY, x_coord2, sy + LEVEL_DEPTH_DY }.draw(3, midsearch_color);
            }
        }
        if (first_endsearch_n_moves == -1) {
            font(Format(level) + language.get("info", "lookahead")).draw(font_size, Arg::topCenter(sx + size_x / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
        } else if (first_endsearch_n_moves == 0) {
            font(language.get("info", "to_last_move")).draw(font_size, Arg::topCenter(sx + size_x / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
        } else {
            int endsearch_bound_coord = sx + dx * first_endsearch_n_moves;
            font(Format(level) + language.get("info", "lookahead")).draw(font_size, Arg::topCenter((sx + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
            font(language.get("info", "to_last_move")).draw(font_size, Arg::topCenter((sx + size_x + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
        }
        // y scale
        if (show_graph) {
            for (int y = 0; y <= y_max - y_min; y += resolution) {
                int yy = sy + dy * y;
                if (show_graph_sum_of_loss) {
                    font(y_max - y).draw(font_size, sx - font(y_max - y).region(font_size, Point{ 0, 0 }).w - 10, yy - font(y_max - y).region(font_size, Point{ 0, 0 }).h / 2, graph_color);
                } else {
                    if (y_max - y >= 0) {
                        font(y_max - y).draw(font_size, sx - font(y_max - y).region(font_size, Point{ 0, 0 }).w - 10, yy - font(y_max - y).region(font_size, Point{ 0, 0 }).h / 2, graph_color);
                    } else if (y_max - y < 0) {
                        font(std::abs(y_max - y)).draw(font_size, sx - font(std::abs(y_max - y)).region(font_size, Point{ 0, 0 }).w - 10, yy - font(y_max - y).region(font_size, Point{ 0, 0 }).h / 2, Palette::White);
                    }
                }
                if (y_max - y == 0) {
                    Line{ sx, yy, sx + size_x, yy }.draw(2, graph_color);
                } else {
                    Line{ sx, yy, sx + size_x, yy }.draw(1, graph_color);
                }
            }
        } else {
            // graph frame
            Line{ sx, sy, sx + size_x, sy }.draw(1, graph_color);
            Line{ sx, sy + size_y, sx + size_x, sy + size_y }.draw(1, graph_color);
        }
        // x scale
        for (int x = 0; x <= 60; x += 10) {
            font(x).draw(font_size, sx + dx * x - font(x).region(font_size, Point{0, 0}).w / 2, sy + size_y + 5, graph_color);
            Line{ sx + dx * x, sy, sx + dx * x, sy + size_y }.draw(1, graph_color);
        }
        if (show_graph) {
            if (show_graph_sum_of_loss) {
                int max_ply1 = nodes1.back().board.n_discs() - 4;
                int max_ply2 = 0;
                if (nodes2.size()) {
                    max_ply2 = nodes2.back().board.n_discs() - 4;
                }
                draw_graph_sum_of_loss(sum_of_loss_nodes1[0], Palette::Black, graph_history_not_calculated_color, max_ply1);
                draw_graph_sum_of_loss(sum_of_loss_nodes1[1], Palette::White, graph_history_not_calculated_color, max_ply1);
                draw_graph_sum_of_loss(sum_of_loss_nodes2[0], Palette::Dimgray, graph_fork_not_calculated_color, max_ply2);
                draw_graph_sum_of_loss(sum_of_loss_nodes2[1], Palette::Silver, graph_fork_not_calculated_color, max_ply2);
            } else {
                draw_graph(nodes1, graph_history_color, graph_history_not_calculated_color);
                draw_graph(nodes2, graph_fork_color, graph_fork_not_calculated_color);
            }
        } else {
            draw_graph_not_calculated(nodes1, graph_history_not_calculated_color);
            draw_graph_not_calculated(nodes2, graph_fork_not_calculated_color);
        }
        if (show_graph && !show_graph_sum_of_loss) {
            Circle(sx, sy, 6).draw(Palette::Black);
            Circle(sx, sy + size_y, 6).draw(Palette::White);
        }
        if (n_discs >= 4) {
            int place_x = sx + dx * (n_discs - 4);
            Line(place_x, sy + 1, place_x, sy + size_y - 1).draw(3, graph_place_color);
            if (sx <= Cursor::Pos().x && Cursor::Pos().x <= sx + size_x && sy <= Cursor::Pos().y && Cursor::Pos().y <= sy + size_y && abs(Cursor::Pos().x - place_x) <= 4) {
                Cursor::RequestStyle(CursorStyle::ResizeLeftRight);
            }
        }
    }

    int update_n_discs(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, int n_discs) {
        if (Rect(sx - 30, sy, size_x + 40, size_y + 10).leftPressed()) {
            int cursor_x = Cursor::Pos().x;
            int min_err = INF;
            for (int i = 0; i < (int)nodes1.size(); ++i) {
                int x = sx + dx * (nodes1[i].board.n_discs() - 4);
                if (abs(x - cursor_x) < min_err) {
                    min_err = abs(x - cursor_x);
                    n_discs = nodes1[i].board.n_discs();
                }
            }
            for (int i = 0; i < (int)nodes2.size(); ++i) {
                int x = sx + dx * (nodes2[i].board.n_discs() - 4);
                if (abs(x - cursor_x) < min_err) {
                    min_err = abs(x - cursor_x);
                    n_discs = nodes2[i].board.n_discs();
                }
            }
        }
        return n_discs;
    }

    bool clicked() {
        return Rect(sx - 30, sy, size_x + 40, size_y).leftClicked();
    }

    bool pressed() {
        return Rect(sx - 30, sy, size_x + 40, size_y).leftPressed();
    }

private:
    void calc_sum_of_loss_nodes(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, std::vector<std::vector<Graph_loss_elem>> &sum_of_loss_nodes1, std::vector<std::vector<Graph_loss_elem>> &sum_of_loss_nodes2) {
        Graph_loss_elem elem;
        elem.ply = nodes1[0].board.n_discs() - 4;
        elem.v = 0;
        sum_of_loss_nodes1[0].emplace_back(elem);
        sum_of_loss_nodes1[1].emplace_back(elem);
        int last_val_black = nodes1[0].v;
        int last_val_white = -nodes1[0].v;
        for (int i = 1; i < (int)nodes1.size(); ++i) {
            int val_black = nodes1[i].v;
            int val_white = -nodes1[i].v;
            int loss_black = sum_of_loss_nodes1[0].back().v;
            int loss_white = sum_of_loss_nodes1[1].back().v;
            if (-HW2 <= val_black && val_black <= HW2) {
                if (-HW2 <= last_val_black && last_val_black <= HW2) {
                    loss_black += std::max(0, last_val_black - val_black);
                    loss_white += std::max(0, last_val_white - val_white);
                }
                last_val_black = val_black;
                last_val_white = val_white;
                elem.ply = nodes1[i].board.n_discs() - 4;
                elem.v = loss_black;
                sum_of_loss_nodes1[0].emplace_back(elem);
                elem.v = loss_white;
                sum_of_loss_nodes1[1].emplace_back(elem);
            }
        }
        if (nodes2.size()) {
            elem.ply = nodes2[0].board.n_discs() - 4;
            elem.v = 0;
            for (Graph_loss_elem &elem1: sum_of_loss_nodes1[0]) {
                if (elem1.ply <= elem.ply) {
                    elem.v = elem1.v;
                }
            }
            for (History_elem &elem2: nodes1) {
                if (elem2.board.n_discs() - 4 == elem.ply - 1 && -HW2 <= elem2.v && elem2.v <= HW2 && -HW2 <= nodes2[0].v && nodes2[0].v <= HW2) {
                    elem.v += std::max(0, elem2.v - nodes2[0].v);
                }
            }
            sum_of_loss_nodes2[0].emplace_back(elem);
            elem.v = 0;
            for (Graph_loss_elem &elem1: sum_of_loss_nodes1[1]) {
                if (elem1.ply <= elem.ply) {
                    elem.v = elem1.v;
                }
            }
            for (History_elem &elem2: nodes1) {
                if (elem2.board.n_discs() - 4 == elem.ply - 1 && -HW2 <= elem2.v && elem2.v <= HW2 && -HW2 <= nodes2[0].v && nodes2[0].v <= HW2) {
                    elem.v -= std::max(0, (-elem2.v) - (-nodes2[0].v));
                }
            }
            sum_of_loss_nodes2[1].emplace_back(elem);
            int last_val_black = nodes2[0].v;
            int last_val_white = -nodes2[0].v;
            for (int i = 1; i < (int)nodes2.size(); ++i) {
                int val_black = nodes2[i].v;
                int val_white = -nodes2[i].v;
                int loss_black = sum_of_loss_nodes2[0].back().v;
                int loss_white = sum_of_loss_nodes2[1].back().v;
                if (-HW2 <= val_black && val_black <= HW2) {
                    if (-HW2 <= last_val_black && last_val_black <= HW2) {
                        loss_black += std::max(0, last_val_black - val_black);
                        loss_white += std::max(0, last_val_white - val_white);
                    }
                    last_val_black = val_black;
                    last_val_white = val_white;
                    elem.ply = nodes2[i].board.n_discs() - 4;
                    elem.v = loss_black;
                    sum_of_loss_nodes2[0].emplace_back(elem);
                    elem.v = loss_white;
                    sum_of_loss_nodes2[1].emplace_back(elem);
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            for (Graph_loss_elem &el: sum_of_loss_nodes1[i]) {
                el.v *= -1;
            }
            for (Graph_loss_elem &el: sum_of_loss_nodes2[i]) {
                el.v *= -1;
            }
        }
    }

    bool calc_endgame_error(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, int *endgame_error_black, int *endgame_error_white) {
        constexpr int N_EMPTIES_ENDGAME_ERROR = 20; // last 20 empties
        *endgame_error_black = 0;
        *endgame_error_white = 0;
        bool endgame_error_calculated = false;
        for (int i = 1; i < (int)nodes1.size(); ++i) {
            if (nodes2.size()) {
                if (nodes2[0].board.n_discs() <= nodes1[i].board.n_discs()) {
                    break;
                }
            }
            if (nodes1[i].board.n_discs() >= HW2 - N_EMPTIES_ENDGAME_ERROR) {
                if (nodes1[i].board.n_discs() - nodes1[i - 1].board.n_discs() != 1) {
                    return false;
                }
                if (-SCORE_MAX <= nodes1[i].v && nodes1[i].v <= SCORE_MAX && -SCORE_MAX <= nodes1[i - 1].v && nodes1[i - 1].v <= SCORE_MAX) {
                    int error = nodes1[i].v - nodes1[i - 1].v;
                    *endgame_error_black += std::max(0, -error);
                    *endgame_error_white += std::max(0, error);
                    endgame_error_calculated = true;
                } else {
                    return false;
                }
            }
        }
        for (int i = 0; i < (int)nodes2.size(); ++i) {
            if (nodes2[i].board.n_discs() >= HW2 - N_EMPTIES_ENDGAME_ERROR) {
                if (i == 0) {
                    for (int j = (int)nodes1.size() - 1; j >= 0; --j) {
                        if (nodes1[j].board.n_discs() < nodes2[i].board.n_discs()) {
                            if (nodes2[i].board.n_discs() - nodes1[j].board.n_discs() != 1) {
                                return false;
                            }
                            if (-SCORE_MAX <= nodes1[j].v && nodes1[j].v <= SCORE_MAX && -SCORE_MAX <= nodes2[i].v && nodes2[i].v <= SCORE_MAX) {
                                int error = nodes2[i].v - nodes1[j].v;
                                *endgame_error_black += std::max(0, -error);
                                *endgame_error_white += std::max(0, error);
                                endgame_error_calculated = true;
                                break;
                            } else {
                                return false;
                            }
                        }
                    }
                } else {
                    if (nodes2[i].board.n_discs() - nodes2[i - 1].board.n_discs() != 1) {
                        return false;
                    }
                    if (-SCORE_MAX <= nodes2[i].v && nodes2[i].v <= SCORE_MAX && -SCORE_MAX <= nodes2[i - 1].v && nodes2[i - 1].v <= SCORE_MAX) {
                        int error = nodes2[i].v - nodes2[i - 1].v;
                        *endgame_error_black += std::max(0, -error);
                        *endgame_error_white += std::max(0, error);
                        endgame_error_calculated = true;
                    } else {
                        return false;
                    }
                }
            }
        }
        return endgame_error_calculated;
    }

    void calc_range(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2) {
        y_min = -GRAPH_RESOLUTION;
        y_max = GRAPH_RESOLUTION;
        for (const History_elem& b : nodes1) {
            if (-HW2 <= b.v && b.v <= HW2) {
                y_min = std::min(y_min, b.v);
                y_max = std::max(y_max, b.v);
            }
        }
        for (const History_elem& b : nodes2) {
            if (-HW2 <= b.v && b.v <= HW2) {
                y_min = std::min(y_min, b.v);
                y_max = std::max(y_max, b.v);
            }
        }
    }

    void calc_range_sum_of_loss(std::vector<std::vector<Graph_loss_elem>> sum_of_loss_nodes1, std::vector<std::vector<Graph_loss_elem>> sum_of_loss_nodes2) {
        y_min = -GRAPH_RESOLUTION;
        y_max = 0;
        for (int i = 0; i < 2; ++i) {
            for (const Graph_loss_elem& b : sum_of_loss_nodes1[i]) {
                y_min = std::min(y_min, b.v);
            }
            for (const Graph_loss_elem& b : sum_of_loss_nodes2[i]) {
                y_min = std::min(y_min, b.v);
            }
        }
    }

    void draw_graph(std::vector<History_elem> nodes, Color color, Color color2) {
        std::vector<std::pair<int, int>> values;
        for (const History_elem& b : nodes) {
            if (b.board.n_discs() >= 4) {
                if (abs(b.v) <= HW2) {
                    int xx = sx + dx * (b.board.n_discs() - 4);
                    int yy = sy + dy * (y_max - b.v);
                    values.emplace_back(std::make_pair(xx, yy));
                    Circle{ xx, yy, 3 }.draw(color);
                } else {
                    int xx = sx + dx * (b.board.n_discs() - 4);
                    int yy = sy + dy * y_max;
                    Circle{ xx, yy, 2.5 }.draw(color2);
                }
            }
        }
        for (int i = 0; i < (int)values.size() - 1; ++i) {
            Line(values[i].first, values[i].second, values[i + 1].first, values[i + 1].second).draw(2, ColorF(color, graph_transparency));
        }
    }

    void draw_graph_sum_of_loss(std::vector<Graph_loss_elem> nodes, Color color, Color color2, int max_ply) {
        if (nodes.size()) {
            std::vector<std::pair<int, int>> values;
            int last_ply = nodes[0].ply;
            for (const Graph_loss_elem& b : nodes) {
                if (b.ply >= 0) {
                    for (int ply = last_ply + 1; ply < b.ply; ++ply) {
                        int xx = sx + dx * ply;
                        int yy = sy + dy * y_max;
                        Circle{ xx, yy, 2.5 }.draw(color2);
                    }
                    int xx = sx + dx * b.ply;
                    int yy = sy + dy * (y_max - b.v);
                    values.emplace_back(std::make_pair(xx, yy));
                    Circle{ xx, yy, 3 }.draw(color);
                    last_ply = b.ply;
                }
            }
            for (int ply = last_ply + 1; ply < max_ply; ++ply) {
                int xx = sx + dx * ply;
                int yy = sy + dy * y_max;
                Circle{ xx, yy, 2.5 }.draw(color2);
            }
            for (int i = 0; i < (int)values.size() - 1; ++i) {
                Line(values[i].first, values[i].second, values[i + 1].first, values[i + 1].second).draw(2, ColorF(color, graph_transparency));
            }
        }
    }

    void draw_graph_not_calculated(std::vector<History_elem> nodes, Color color) {
        std::vector<std::pair<int, int>> values;
        for (const History_elem& b : nodes) {
            if (b.board.n_discs() >= 4) {
                int yy = sy + dy * y_max;
                Circle{ sx + dx * (b.board.n_discs() - 4), yy, 2.5 }.draw(color);
            }
        }
    }
};