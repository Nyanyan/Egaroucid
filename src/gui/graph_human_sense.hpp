#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include <math.h>
#include "../board.hpp"
#include "../human_value.hpp"
#include "gui_common.hpp"

using namespace std;

constexpr Color human_sense_graph_color = Color(51, 51, 51);
Color human_sense_line_color_black = Palette::Black;
Color human_sense_line_color_white = Palette::White;
constexpr double human_sense_graph_transparency = 1.0;

class Human_sense_graph {
public:
	int sx_black;
	int sx_white;
	int sy;
	int size_x;
	int size_y;
	int stone_resolution;
	int stability_resolution;
	int font_size;
	Font font;

private:
	int stone_min;
	int stone_max;
	int stability_max;

public:
	void draw(vector<Human_value> nodes, Board bd) {
		calc_range(nodes, bd);
		for (int y = 0; y <= stability_max; y += stability_resolution) {
			int xx = size_x * (-stone_min) / (stone_max - stone_min);
			int yy = sy + size_y * (stability_max - y) / stability_max;
			font(y).draw(sx_black + xx - font(y).region(Point{0, 0}).w * 1.5, yy - font(y).region(Point{0, 0}).h, graph_color);
			Line{ sx_black, yy, sx_black + size_x, yy }.draw(1, graph_color);
			font(y).draw(sx_white + xx - font(y).region(Point{ 0, 0 }).w * 1.5, yy - font(y).region(Point{ 0, 0 }).h, graph_color);
			Line{ sx_white, yy, sx_white + size_x, yy }.draw(1, graph_color);
		}
		for (int x = stone_min; x <= stone_max; x += stone_resolution) {
			int xx = size_x * (x - stone_min) / (stone_max - stone_min);
			font(x).draw(sx_black + xx - font(x).region(Point{0, 0}).w / 2, sy + size_y + 7, human_sense_graph_color);
			Line{ sx_black + xx, sy, sx_black + xx, sy + size_y }.draw(1, graph_color);
			font(x).draw(sx_white + xx - font(x).region(Point{ 0, 0 }).w / 2, sy + size_y + 7, human_sense_graph_color);
			Line{ sx_white + xx, sy, sx_white + xx, sy + size_y }.draw(1, graph_color);
		}
		Circle(sx_black + size_x, sy + size_y, 10).draw(Palette::Black);
		Circle(sx_black, sy + size_y, 10).draw(Palette::White);
		Circle(sx_white + size_x, sy + size_y, 10).draw(Palette::Black);
		Circle(sx_white, sy + size_y, 10).draw(Palette::White);
		draw_graph(nodes, bd);
	}

private:
	void calc_range(vector<Human_value> nodes, Board bd) {
		double res = 0.0;
		stone_min = 0;
		stone_max = 0;
		for (const Human_value& elem : nodes) {
			if (elem.moves < bd.n - 4) {
				res = max(res, elem.stability_black);
				res = max(res, elem.stability_white);
				stone_min = min(stone_min, elem.prospect);
				stone_max = max(stone_max, elem.prospect);
			}
		}
		stability_max = max((int)ceil(res), stability_resolution - 1);
		stability_max += stability_resolution - stability_max % stability_resolution;
		stone_min = min(-stone_resolution + 1, stone_min);
		stone_max = max(stone_resolution - 1, stone_max);
		stone_min -= (stone_min + HW2) % stone_resolution;
		stone_max += stone_resolution - stone_max % stone_resolution;
	}

	void draw_graph(vector<Human_value> nodes, Board bd) {
		vector<pair<int, int>> values_black, values_white;
		int xx_black, xx_white, yy_black, yy_white;
		for (const Human_value& elem : nodes) {
			if (elem.moves < bd.n - 4) {
				xx_black = sx_black + size_x * (elem.prospect - stone_min) / (stone_max - stone_min);
				xx_white = sx_white + size_x * (elem.prospect - stone_min) / (stone_max - stone_min);
				yy_black = sy + size_y - size_y * elem.stability_black / stability_max;
				yy_white = sy + size_y - size_y * elem.stability_white / stability_max;
				Circle{ xx_black, yy_black, 2 }.draw(human_sense_line_color_black);
				Circle{ xx_white, yy_white, 2 }.draw(human_sense_line_color_white);
				values_black.emplace_back(make_pair(xx_black, yy_black));
				values_white.emplace_back(make_pair(xx_white, yy_white));
			}
		}
		for (int i = 0; i < (int)values_black.size() - 1; ++i) {
			Line(values_black[i].first, values_black[i].second, values_black[i + 1].first, values_black[i + 1].second).draw(1, human_sense_line_color_black);
		}
		for (int i = 0; i < (int)values_white.size() - 1; ++i) {
			Line(values_white[i].first, values_white[i].second, values_white[i + 1].first, values_white[i + 1].second).draw(1, human_sense_line_color_white);
		}
	}
};

