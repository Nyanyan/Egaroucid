/*
    Egaroucid Project

	@file graph.hpp
		Graph drawing
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp>
#include <vector>
#include "./../../engine/board.hpp"
#include "language.hpp"
#include "gui_common.hpp"

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
		Color(51, 161, 255)   // light blue
	},
	{
		Color(245, 58, 58),  // red
		Color(178, 28, 108), // dark purple
		Color(214, 56, 248), // purple
		Color(18, 4, 171),   // blue
		Color(66, 109, 255), // light blue
		Color(44, 166, 167)  // cyan
	}
};
constexpr Color midsearch_color = Color(51, 51, 51);
constexpr Color endsearch_color = Palette::White;
constexpr Color level_info_color = Palette::White;
constexpr Color level_prob_color = Palette::White;

constexpr Color graph_rect_color = Palette::White;

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
	void draw(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, int n_discs, bool show_graph, int level, Font font, int color_type) {
		bool fix_resolution_flag = false;
		if (show_graph) {
			calc_range(nodes1, nodes2);
			if (y_max - y_min > 80) { // range is too wide
				fix_resolution_flag = true;
				resolution *= 2;
				y_min -= (y_min + HW2) % resolution;
				y_max += (resolution - (y_max + HW2) % resolution) % resolution;
				dy = (double)size_y / (y_max - y_min);
				dx = (double)size_x / 60;
			}
		}
		else {
			y_min = -resolution;
			y_max = resolution;
			dy = (double)size_y / (y_max - y_min);
			dx = (double)size_x / 60;
		}
		RoundRect round_rect{ sx + GRAPH_RECT_DX, sy + GRAPH_RECT_DY, GRAPH_RECT_WIDTH, GRAPH_RECT_HEIGHT, GRAPH_RECT_RADIUS };
		round_rect.drawFrame(GRAPH_RECT_THICKNESS, graph_rect_color);
		int info_x = sx + LEVEL_INFO_DX + GRAPH_RECT_DX + GRAPH_RECT_WIDTH / 2 - (LEVEL_INFO_WIDTH * 5 + LEVEL_PROB_WIDTH) / 2;
		int info_y = sy + LEVEL_INFO_DY;
		Rect rect_prob{info_x, info_y, LEVEL_PROB_WIDTH, LEVEL_INFO_HEIGHT};
		rect_prob.draw(graph_color);
		font(language.get("info", "probability")).draw(font_size, Arg::center(info_x + LEVEL_PROB_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_prob_color);
		info_x += LEVEL_PROB_WIDTH;
		for (int i = 0; i < N_SELECTIVITY_LEVEL; ++i){
			Rect rect_selectivity{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
			rect_selectivity.draw(color_selectivity[color_type][i]);
			font(Format(SELECTIVITY_PERCENTAGE[i]) + U"%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
			info_x += LEVEL_INFO_WIDTH;
		}
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
			}
			else {
				Line{ x_coord1, sy + LEVEL_DEPTH_DY, x_coord2, sy + LEVEL_DEPTH_DY }.draw(3, midsearch_color);
			}
		}
		if (first_endsearch_n_moves == -1) {
			font(Format(level) + language.get("info", "lookahead")).draw(font_size, Arg::topCenter(sx + size_x / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
		}
		else if (first_endsearch_n_moves == 0) {
			font(language.get("info", "to_last_move")).draw(font_size, Arg::topCenter(sx + size_x / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
		}
		else {
			int endsearch_bound_coord = sx + dx * first_endsearch_n_moves;
			font(Format(level) + language.get("info", "lookahead")).draw(font_size, Arg::topCenter((sx + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
			font(language.get("info", "to_last_move")).draw(font_size, Arg::topCenter((sx + size_x + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
		}
		for (int y = 0; y <= y_max - y_min; y += resolution) {
			int yy = sy + dy * y;
			font(y_max - y).draw(font_size, sx - font(y_max - y).region(font_size, Point{ 0, 0 }).w - 12, yy - font(y_max - y).region(font_size, Point{ 0, 0 }).h / 2, graph_color);
			if (y_max - y == 0)
				Line{ sx, yy, sx + size_x, yy }.draw(2, graph_color);
			else
				Line{ sx, yy, sx + size_x, yy }.draw(1, graph_color);
		}
		for (int x = 0; x <= 60; x += 10) {
			font(x).draw(font_size, sx + dx * x - font(x).region(font_size, Point{0, 0}).w / 2, sy + size_y + 5, graph_color);
			Line{ sx + dx * x, sy, sx + dx * x, sy + size_y }.draw(1, graph_color);
		}
		if (show_graph) {
			draw_graph(nodes1, graph_history_color, graph_history_not_calculated_color);
			draw_graph(nodes2, graph_fork_color, graph_fork_not_calculated_color);
		}
		else {
			draw_graph_not_calculated(nodes1, graph_history_not_calculated_color);
			draw_graph_not_calculated(nodes2, graph_fork_not_calculated_color);
		}
		int place_x = sx + dx * (n_discs - 4);
		Circle(sx, sy, 7).draw(Palette::Black);
		Circle(sx, sy + size_y, 7).draw(Palette::White);
		Line(place_x, sy, place_x, sy + size_y).draw(3, graph_place_color);
		if (fix_resolution_flag) {
			resolution /= 2;
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
	void calc_range(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2) {
		y_min = -resolution;
		y_max = resolution;
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
		y_min -= (y_min + HW2) % resolution;
		y_max += (resolution - (y_max + HW2) % resolution) % resolution;
		y_min = std::max(-HW2, y_min);
		y_max = std::min(HW2, y_max);
		dy = (double)size_y / (y_max - y_min);
		dx = (double)size_x / 60;
	}

	void draw_graph(std::vector<History_elem> nodes, Color color, Color color2) {
		std::vector<std::pair<int, int>> values;
		for (const History_elem& b : nodes) {
			if (abs(b.v) <= HW2) {
				int xx = sx + dx * (b.board.n_discs() - 4);
				int yy = sy + dy * (y_max - b.v);
				values.emplace_back(std::make_pair(xx, yy));
				Circle{ xx, yy, 3 }.draw(color);
			}
			else {
				int yy = sy + dy * y_max;
				Circle{ sx + dx * (b.board.n_discs() - 4), yy, 2.5 }.draw(color2);
			}
		}
		for (int i = 0; i < (int)values.size() - 1; ++i) {
			Line(values[i].first, values[i].second, values[i + 1].first, values[i + 1].second).draw(2, ColorF(color, graph_transparency));
		}
	}

	void draw_graph_not_calculated(std::vector<History_elem> nodes, Color color) {
		std::vector<std::pair<int, int>> values;
		for (const History_elem& b : nodes) {
			int yy = sy + dy * y_max;
			Circle{ sx + dx * (b.board.n_discs() - 4), yy, 2.5 }.draw(color);
		}
	}
};