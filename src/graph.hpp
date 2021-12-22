#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>

class Graph {
	public:
		int sx;
		int sy;
		int size_x;
		int size_y;
		int resolution;
		int font_size;
		Font font;
	private:
		vector<pair<int, int>> nodes;
		int dx;
		int dy;
		int y_max;
		int y_min;
	public:
		void draw() {
			calc_range();
			dy = size_y / (y_max - y_min);
			dx = size_x / 60;
			for (int y = 0; y <= y_max - y_min; y += resolution) {
				font(y_max - y).draw(sx - font_size * 3, sy + y * dy - font_size);
				if (y_max - y == 0)
					Line{sx, sy + y * dy, sx + size_x, sy + y * dy}.draw(2, Palette::White);
				else
					Line{sx, sy + y * dy, sx + size_x, sy + y * dy}.draw(1, Palette::White);
			}
			for (int x = 0; x <= 60; x += 10){
				font(x).draw(sx + x * dx, sy - 2 * font_size);
				Line{sx + x * dx, sy, sx + x * dx, sy + size_y}.draw(1, Palette::White);
			}
			for (pair<int, int> yx : nodes)
				Circle{sx + yx.first * dx, sy + y_max * dy - yx.second * dy, 3}.draw(Palette::White);
			if (nodes.size() >= 2) {
				for (int i = 0; i < (int)nodes.size() - 1; ++i) {
					Line(sx + nodes[i].first * dx, sy + y_max * dy - nodes[i].second * dy, sx + nodes[i + 1].first * dx, sy + y_max * dy - nodes[i + 1].second * dy).draw(3, Palette::White);
				}
			}
		}

		void push(int x, double y) {
			nodes.push_back(make_pair(x, (int)round(y)));
		}

		void clear() {
			nodes.clear();
		}

	private:
		void calc_range() {
			y_min = 1000;
			y_max = -1000;
			for (pair<int, int> yx : nodes) {
				y_min = min(y_min, yx.second);
				y_max = max(y_max, yx.second);
			}
			y_min = min(y_min, 0);
			y_max = max(y_max, 0);
			y_min = y_min - resolution + abs(y_min) % resolution;
			y_max = y_max + resolution - abs(y_max) % resolution;
		}
};
