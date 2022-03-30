#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include <math.h>
#include "../board.hpp"
#include "../human_value.hpp"
#include "gui_common.hpp"

using namespace std;

constexpr Color human_sense_graph_color = Color(51, 51, 51);
constexpr double human_sense_graph_transparency = 1.0;

class Human_sense_graph {
public:
	int sx;
	int sy;
	int size_x;
	int size_y;
	int resolution;
	int font_size;
	Font font;

private:
	int stability_black_min;
	int stability_black_max;
	int stability_white_min;
	int stability_white_max;

public:
	void draw(vector<Human_value> nodes, Board bd) {
		calc_range(nodes, bd);
		int xx, yy;
		yy = size_y - size_y * (50 - stability_white_min) / (stability_white_max - stability_white_min);
		for (int x = stability_black_min; x <= stability_black_max; x += resolution) {
			xx = size_x * (x - stability_black_min) / (stability_black_max - stability_black_min);
			font(x).draw(sx + xx - font(x).region(Point{ 0, 0 }).w * 1.5, sy + yy, human_sense_graph_color);
			if (x == 50) {
				Line{ sx + xx, sy, sx + xx, sy + size_y }.draw(2, graph_color);
			}
			else {
				Line{ sx + xx, sy, sx + xx, sy + size_y }.draw(1, graph_color);
			}
		}
		xx = size_x * (50 - stability_black_min) / (stability_black_max - stability_black_min);
		for (int y = stability_white_min; y <= stability_white_max; y += resolution) {
			yy = size_y - size_y * (y - stability_white_min) / (stability_white_max - stability_white_min);
			font(y).draw(sx + xx - font(y).region(Point{0, 0}).w * 1.5, sy + yy - font(y).region(Point{0, 0}).h, graph_color);
			if (y == 50) {
				Line{ sx, sy + yy, sx + size_x, sy + yy }.draw(2, graph_color);
			}
			else {
				Line{ sx, sy + yy, sx + size_x, sy + yy }.draw(1, graph_color);
			}
		}
		yy = size_y - size_y * (50 - stability_white_min) / (stability_white_max - stability_white_min);
		xx = size_x * (50 - stability_black_min) / (stability_black_max - stability_black_min);
		Circle(sx, sy + yy, 7).draw(Palette::Black);
		Circle(sx + size_x, sy + yy, 7).draw(Palette::Black);
		Circle(sx + xx, sy, 7).draw(Palette::White);
		Circle(sx + xx, sy + size_y, 7).draw(Palette::White);
		draw_graph(nodes, bd);
	}

private:
	void calc_range(vector<Human_value> nodes, Board bd) {
		stability_black_min = 50 - resolution;
		stability_black_max = 50 + resolution;
		stability_white_min = 50 - resolution;
		stability_white_max = 50 + resolution;
		for (const Human_value& elem : nodes) {
			if (elem.moves <= bd.n - 4) {
				stability_black_min = min(stability_black_min, (int)round(elem.stability_black));
				stability_black_max = max(stability_black_max, (int)round(elem.stability_black));
				stability_white_min = min(stability_white_min, (int)round(elem.stability_white));
				stability_white_max = max(stability_white_max, (int)round(elem.stability_white));
			}
		}
		stability_black_min -= stability_black_min % resolution;
		stability_black_max += (resolution - stability_black_max % resolution) % resolution;
		stability_white_min -= stability_white_min % resolution;
		stability_white_max += (resolution - stability_white_max % resolution) % resolution;
	}

	void draw_graph(vector<Human_value> nodes, Board bd) {
		vector<pair<int, int>> values;
		int xx, yy;
		for (const Human_value& elem : nodes) {
			if (elem.moves <= bd.n - 4) {
				xx = sx + size_x * ((int)round(elem.stability_black) - stability_black_min) / (stability_black_max - stability_black_min);
				yy = sy + size_y - size_y * ((int)round(elem.stability_white) - stability_white_min) / (stability_white_max - stability_white_min);
				Circle{ xx, yy, 3 }.draw(Palette::White);
				values.emplace_back(make_pair(xx, yy));
			}
		}
		for (int i = 0; i < (int)values.size() - 1; ++i) {
			Line(values[i].first, values[i].second, values[i + 1].first, values[i + 1].second).draw(2, Palette::White);
		}
	}
};

