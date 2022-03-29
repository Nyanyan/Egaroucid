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

int calc_human_sense_stability_range(vector<Human_value> nodes) {
	double res = 0.0;
	for (const Human_value& elem : nodes) {
		res = max(res, elem.stability_black);
		res = max(res, elem.stability_white);
	}
	return (int)ceil(res);
}

class Human_sense_graph {
public:
	int sx_black;
	int sx_white;
	int sy;
	int size_x;
	int size_y;
	int stone_resolution;
	int stability_resolution;
	int stability_max;
	int font_size;
	Font font;

public:
	void set_range(int stab_max) {
		stability_max = max(stab_max, stability_resolution);
	}

	void draw(vector<Human_value> nodes) {
		for (int y = 0; y <= stability_max; y += stability_resolution) {
			int yy = sy + size_y * (stability_max - y) / stability_max;
			font(y).draw(sx_black + size_x / 2 - font(y).region(Point{0, 0}).w * 2, yy - font(y).region(Point{0, 0}).h / 2, graph_color);
			Line{ sx_black, yy, sx_black + size_x, yy }.draw(1, graph_color);
			font(y).draw(sx_white + size_x / 2 - font(y).region(Point{ 0, 0 }).w * 2, yy - font(y).region(Point{ 0, 0 }).h / 2, graph_color);
			Line{ sx_white, yy, sx_white + size_x, yy }.draw(1, graph_color);
		}
		for (int x = -HW2; x <= HW2; x += stone_resolution) {
			font(x).draw(sx_black + size_x * (x + HW2) / (HW2 * 2) - font(x).region(Point{0, 0}).w / 2, sy + size_y + font_size * 2, human_sense_graph_graph_color);
			Line{ sx_black + size_x * (x + HW2) / (HW2 * 2), sy, sx + size_x * (x + HW2) / (HW2 * 2), sy + size_y }.draw(1, graph_color);
			font(x).draw(sx_white + size_x * (x + HW2) / (HW2 * 2) - font(x).region(Point{ 0, 0 }).w / 2, sy + size_y + font_size * 2, human_sense_graph_graph_color);
			Line{ sx_white + size_x * (x + HW2) / (HW2 * 2), sy, sx + size_x * (x + HW2) / (HW2 * 2), sy + size_y }.draw(1, graph_color);
		}
		draw_graph(nodes);
	}

private:
	void draw_graph(vector<Human_value> nodes) {
		vector<pair<int, int>> values_black, values_white;
		int xx_black, xx_white, yy_black, yy_white;
		for (const Human_value& elem : nodes) {
			xx_black = sx_black + size_x * (elem.prospects + HW2) / (HW2 * 2);
			xx_white = sx_white + size_x * (elem.prospects + HW2) / (HW2 * 2);
			yy_black = sy + size_y * elem.stability_black / stability_max;
			yy_white = sy + size_y * elem.stability_white / stability_max;
			Circle{ xx_black, yy_black }.draw(human_sense_line_color_black);
			Circle{ xx_white, yy_white }.draw(human_sense_line_color_white);
			values_black.emplace_back(make_pair(xx_black, yy_black));
			values_white.emplace_back(make_pair(xx_white, yy_white));
		}
		for (int i = 0; i < (int)values_black.size() - 1; ++i) {
			Line(values_black[i].first, values_black[i].second, values_black[i + 1].first, values_black[i + 1].second).draw(2, human_sense_line_color_black);
		}
		for (int i = 0; i < (int)values_white.size() - 1; ++i) {
			Line(values_white[i].first, values_white[i].second, values_white[i + 1].first, values_white[i + 1].second).draw(2, human_sense_line_color_white);
		}
	}
};

