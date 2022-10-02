/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include "../level.hpp"
#include "gui_common.hpp"

using namespace std;

constexpr Color level_color = Color(51, 51, 51);
constexpr Color level_place_color = Palette::White;
constexpr Color level_endsearch_color = Palette::White;
constexpr Color level_info_color = Palette::White;

struct Level_colors {
	Color color_81{ Color(153, 50, 204) };
	Color color_95{ Color(178, 34, 34) };
	Color color_98{ Color(205, 133, 63) };
	Color color_99{ Color(20, 100, 0) };
	Color color_100{ Color(100, 149, 237) };
};

class Level_display {
public:
	int sx;
	int sy;
	int size_x;
	int size_y;

private:
	int font_size{ 12 };
	Font font{ 12 };
	Level_colors level_colors;
	Font font_big{ 15 };

public:
	void draw(int level, int n_discs) {
		font_big(language.get("common", "level") + U" " + Format(level)).draw(INFO_SX, LEVEL_INFO_SY);
		String level_type_info;
		if (level <= LIGHT_LEVEL) {
			level_type_info = language.get("info", "light");
		}
		else if (level <= STANDARD_MAX_LEVEL) {
			level_type_info = language.get("info", "standard");
		}
		else if (level <= PRAGMATIC_MAX_LEVEL) {
			level_type_info = language.get("info", "pragmatic");
		}
		else if (level <= ACCURATE_MAX_LEVEL) {
			level_type_info = language.get("info", "accurate");
		}
		else {
			level_type_info = language.get("info", "danger");
		}
		font(level_type_info).draw(INFO_SX, LEVEL_INFO_SY + 23);
		int n_moves = n_discs - 4;
		int info_x = LEVEL_INFO_SX;
		int info_y = LEVEL_INFO_RECT_SY;
		Rect rect_81{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_81.draw(level_colors.color_81);
		font(U"81%").draw(Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_95{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_95.draw(level_colors.color_95);
		font(U"95%").draw(Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_98{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_98.draw(level_colors.color_98);
		font(U"98%").draw(Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_99{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_99.draw(level_colors.color_99);
		font(U"99%").draw(Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		info_x += LEVEL_INFO_WIDTH;
		Rect rect_100{ info_x, info_y, LEVEL_INFO_WIDTH, LEVEL_INFO_HEIGHT };
		rect_100.draw(level_colors.color_100);
		font(U"100%").draw(Arg::center(info_x + LEVEL_INFO_WIDTH / 2, info_y + LEVEL_INFO_HEIGHT / 2), level_info_color);
		int dx = size_x / 60;
		int adj_x = size_x - dx * 60;
		bool is_mid_search, use_mpc;
		int depth;
		double mpct;
		int first_endsearch_n_moves = -1;
		for (int x = 0; x < 60; ++x) {
			int x_coord1 = sx + x * dx + adj_x * x / 60;
			int x_coord2 = sx + (x + 1) * dx + adj_x * (x + 1) / 60;
			Rect rect{ x_coord1, sy, x_coord2 - x_coord1, size_y };
			get_level(level, x, &is_mid_search, &depth, &use_mpc, &mpct);
			Color color = level_colors.color_100;
			int probability = calc_probability(mpct);
			if (probability == 81) {
				color = level_colors.color_81;
			}
			else if (probability == 95) {
				color = level_colors.color_95;
			}
			else if (probability == 98) {
				color = level_colors.color_98;
			}
			else if (probability == 99) {
				color = level_colors.color_99;
			}
			rect.draw(color);
			if (!is_mid_search) {
				if (first_endsearch_n_moves == -1) {
					first_endsearch_n_moves = x;
				}
				Line{ x_coord1, LEVEL_DEPTH_SY, x_coord2, LEVEL_DEPTH_SY }.draw(3, level_endsearch_color);
			}
			else {
				Line{ x_coord1, LEVEL_DEPTH_SY, x_coord2, LEVEL_DEPTH_SY }.draw(3, level_color);
			}
		}
		if (first_endsearch_n_moves == -1) {
			font(Format(level) + language.get("info", "lookahead")).draw(Arg::topCenter(sx + LEVEL_WIDTH / 2, LEVEL_DEPTH_SY + 4), level_color);
		}
		else if (first_endsearch_n_moves == 0) {
			font(language.get("info", "to_last_move")).draw(Arg::topCenter(sx + LEVEL_WIDTH / 2, LEVEL_DEPTH_SY + 4), level_color);
		}
		else  {
			int endsearch_bound_coord = sx + first_endsearch_n_moves * dx + adj_x * first_endsearch_n_moves / 60;
			font(Format(level) + language.get("info", "lookahead")).draw(Arg::topCenter((sx + endsearch_bound_coord) / 2, LEVEL_DEPTH_SY + 4), level_color);
			font(language.get("info", "to_last_move")).draw(Arg::topCenter((sx + LEVEL_WIDTH + endsearch_bound_coord) / 2, LEVEL_DEPTH_SY + 4), level_color);
		}
		for (int x = 0; x <= 60; x += 10) {
			font(x).draw(sx + x * dx + adj_x * x / 60 - font(x).region(Point{ 0, 0 }).w / 2, sy - 1.5 * font_size, level_color);
			Line{ sx + x * dx + adj_x * x / 60, sy, sx + x * dx + adj_x * x / 60, sy + size_y }.draw(1, level_color);
		}
		Line{ sx, sy, sx + size_x, sy }.draw(1, level_color);
		Line{ sx, sy + size_y, sx + size_x, sy + size_y }.draw(1, level_color);
		int move_place = sx + n_moves * dx + adj_x * n_moves / 60;
		Line{ move_place, sy, move_place, sy + size_y }.draw(3, level_place_color);
	}
};
