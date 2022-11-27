/*
    Egaroucid Project

	@file graph.hpp
		Graph drawing
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
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

constexpr Color color_81 = Color(227, 88, 72);
constexpr Color color_95 = Color(240, 135, 20);
constexpr Color color_98 = Color(205, 170, 36);
constexpr Color color_99 = Color(176, 210, 88);
constexpr Color color_100 = Color(51, 161, 255);
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
	Font font{ FontMethod::MSDF, FONT_DEFAULT_SIZE };
	int y_max;
	int y_min;
	int dy;
	int dx;
	int adj_y;
	int adj_x;

public:
	void draw(std::vector<History_elem> nodes1, std::vector<History_elem> nodes2, int n_discs, bool show_graph, int level) {
		bool fix_resolution_flag = false;
		if (show_graph) {
			calc_range(nodes1, nodes2);
			if (y_max - y_min > 80) {
				fix_resolution_flag = true;
				resolution *= 2;
				y_min -= (y_min + HW2) % resolution;
				y_max += (resolution - (y_max + HW2) % resolution) % resolution;
			}
		}
		else {
			y_min = -resolution;
			y_max = resolution;
			dy = size_y / (y_max - y_min);
			dx = size_x / 60;
			adj_y = size_y - dy * (y_max - y_min);
			adj_x = size_x - dx * 60;
		}
		RoundRect round_rect{ sx + GRAPH_RECT_DX, sy + GRAPH_RECT_DY, GRAPH_RECT_WIDTH, GRAPH_RECT_HEIGHT, GRAPH_RECT_RADIUS };
		round_rect.drawFrame(GRAPH_RECT_THICKNESS, graph_rect_color);
		int info_x = sx + GRAPH_RECT_DX + GRAPH_RECT_WIDTH / 2 - (LEVEL_INFO_WIDTH * 5 + LEVEL_PROB_WIDTH) / 2;
		int info_y = sy + LEVEL_INFO_DY;
		Rect rect_prob{info_x, info_y, LEVEL_PROB_WIDTH, LEVEL_INFO_HEIGHT};
		rect_prob.draw(graph_color);
		font(language.get("info", "probability")).draw(font_size, Arg::center(info_x + LEVEL_PROB_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_prob_color);
		info_x += LEVEL_PROB_WIDTH;
		Rect rect_81{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_81.draw(color_81);
		font(U"81%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_95{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_95.draw(color_95);
		font(U"95%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_98{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_98.draw(color_98);
		font(U"98%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_99{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_99.draw(color_99);
		font(U"99%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_100{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_100.draw(color_100);
		font(U"100%").draw(font_size, Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		bool is_mid_search;
		int depth;
		uint_fast8_t mpc_level;
		double mpct;
		int first_endsearch_n_moves = -1;
		for (int x = 0; x < 60; ++x) {
			int x_coord1 = sx + x * dx + adj_x * x / 60;
			int x_coord2 = sx + (x + 1) * dx + adj_x * (x + 1) / 60;
			Rect rect{ x_coord1, sy, x_coord2 - x_coord1, size_y };
			get_level(level, x, &is_mid_search, &depth, &mpc_level);
			Color color = color_100;
			int probability = SELECTIVITY_PERCENTAGE[mpc_level];
			if (probability == 81) {
				color = color_81;
			}
			else if (probability == 95) {
				color = color_95;
			}
			else if (probability == 98) {
				color = color_98;
			}
			else if (probability == 99) {
				color = color_99;
			}
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
			int endsearch_bound_coord = sx + first_endsearch_n_moves * dx + adj_x * first_endsearch_n_moves / 60;
			font(Format(level) + language.get("info", "lookahead")).draw(font_size, Arg::topCenter((sx + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
			font(language.get("info", "to_last_move")).draw(font_size, Arg::topCenter((sx + size_x + endsearch_bound_coord) / 2, sy + LEVEL_DEPTH_DY - 18), graph_color);
		}
		for (int y = 0; y <= y_max - y_min; y += resolution) {
			int yy = sy + y * dy + adj_y * y / (y_max - y_min);
			font(y_max - y).draw(font_size, sx - font(y_max - y).region(font_size, Point{ 0, 0 }).w - 12, yy - font(y_max - y).region(font_size, Point{ 0, 0 }).h / 2, graph_color);
			if (y_max - y == 0)
				Line{ sx, yy, sx + size_x, yy }.draw(2, graph_color);
			else
				Line{ sx, yy, sx + size_x, yy }.draw(1, graph_color);
		}
		for (int x = 0; x <= 60; x += 10) {
			font(x).draw(font_size, sx + x * dx + adj_x * x / 60 - font(x).region(font_size, Point{0, 0}).w / 2, sy + size_y + 5, graph_color);
			Line{ sx + x * dx + adj_x * x / 60, sy, sx + x * dx + adj_x * x / 60, sy + size_y }.draw(1, graph_color);
		}
		if (show_graph) {
			draw_graph(nodes1, graph_history_color, graph_history_not_calculated_color);
			draw_graph(nodes2, graph_fork_color, graph_fork_not_calculated_color);
		}
		else {
			draw_graph_not_calculated(nodes1, graph_history_not_calculated_color);
			draw_graph_not_calculated(nodes2, graph_fork_not_calculated_color);
		}
		int place_x = sx + (n_discs - 4) * dx + (n_discs - 4) * adj_x / 60;
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
				int x = sx + (nodes1[i].board.n_discs() - 4) * dx + (nodes1[i].board.n_discs() - 4) * adj_x / 60;
				if (abs(x - cursor_x) < min_err) {
					min_err = abs(x - cursor_x);
					n_discs = nodes1[i].board.n_discs();
				}
			}
			for (int i = 0; i < (int)nodes2.size(); ++i) {
				int x = sx + (nodes2[i].board.n_discs() - 4) * dx + (nodes2[i].board.n_discs() - 4) * adj_x / 60;
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
			if (b.v != GRAPH_IGNORE_VALUE) {
				y_min = std::min(y_min, b.v);
				y_max = std::max(y_max, b.v);
			}
		}
		for (const History_elem& b : nodes2) {
			if (b.v != GRAPH_IGNORE_VALUE) {
				y_min = std::min(y_min, b.v);
				y_max = std::max(y_max, b.v);
			}
		}
		y_min -= (y_min + HW2) % resolution;
		y_max += (resolution - (y_max + HW2) % resolution) % resolution;
		y_min = std::max(-HW2, y_min);
		y_max = std::min(HW2, y_max);
		dy = size_y / (y_max - y_min);
		dx = size_x / 60;
		adj_y = size_y - dy * (y_max - y_min);
		adj_x = size_x - dx * 60;
	}

	void draw_graph(std::vector<History_elem> nodes, Color color, Color color2) {
		std::vector<std::pair<int, int>> values;
		for (const History_elem& b : nodes) {
			if (abs(b.v) <= HW2) {
				int xx = sx + (b.board.n_discs() - 4) * dx + (b.board.n_discs() - 4) * adj_x / 60;
				int yy = sy + (y_max - b.v) * dy + adj_y * (y_max - b.v) / (y_max - y_min);
				values.emplace_back(std::make_pair(xx, yy));
				Circle{ xx, yy, 3 }.draw(color);
			}
			else {
				int yy = sy + y_max * dy + adj_y * y_max / (y_max - y_min);
				Circle{ sx + (b.board.n_discs() - 4) * dx + (b.board.n_discs() - 4) * adj_x / 60, yy, 2.5 }.draw(color2);
			}
		}
		for (int i = 0; i < (int)values.size() - 1; ++i) {
			Line(values[i].first, values[i].second, values[i + 1].first, values[i + 1].second).draw(2, ColorF(color, graph_transparency));
		}
	}

	void draw_graph_not_calculated(std::vector<History_elem> nodes, Color color) {
		std::vector<std::pair<int, int>> values;
		for (const History_elem& b : nodes) {
			int yy = sy + y_max * dy + adj_y * y_max / (y_max - y_min);
			Circle{ sx + (b.board.n_discs() - 4) * dx + (b.board.n_discs() - 4) * adj_x / 60, yy, 2.5 }.draw(color);
		}
	}
};

