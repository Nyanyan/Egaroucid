#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include <algorithm>

using namespace std;

Color graph_color = Palette::Black;

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
		vector<int> show_nodes;
		int dx;
		int dy;
		int y_max;
		int y_min;
	public:
		void draw() {
			sort(nodes.begin(), nodes.end());
			show_nodes.clear();
			for (int i = 0; i < (int)nodes.size() - 1; ++i)
				show_nodes.emplace_back(round((double)(nodes[i].second + nodes[i + 1].second) / 2.0));
			if (nodes.size())
				show_nodes.emplace_back(nodes[nodes.size() - 1].second);
			calc_range();
			dy = size_y / (y_max - y_min);
			dx = size_x / 60;
			font(U"黒").draw(sx + 5, sy, graph_color);
			font(U"白").draw(sx + 5, sy + size_y - font_size * 1.5, graph_color);
			for (int y = 0; y <= y_max - y_min; y += resolution) {
				font(y_max - y).draw(sx - font_size * 3, sy + y * dy - font_size, graph_color);
				if (y_max - y == 0)
					Line{sx, sy + y * dy, sx + size_x, sy + y * dy}.draw(2, graph_color);
				else
					Line{sx, sy + y * dy, sx + size_x, sy + y * dy}.draw(1, graph_color);
			}
			for (int x = 0; x <= 60; x += 10){
				font(x).draw(sx + x * dx, sy - 2 * font_size);
				Line{sx + x * dx, sy, sx + x * dx, sy + size_y}.draw(1, graph_color);
			}
			for (int i = 0; i < (int)nodes.size(); ++i)
				Circle{sx + nodes[i].first * dx, sy + y_max * dy - show_nodes[i] * dy, 3}.draw(graph_color);
			if (nodes.size() >= 2) {
				for (int i = 0; i < (int)nodes.size() - 1; ++i) {
					Line(sx + nodes[i].first * dx, sy + y_max * dy - show_nodes[i] * dy, sx + nodes[i + 1].first * dx, sy + y_max * dy - show_nodes[i + 1] * dy).draw(4, graph_color);
				}
			}
		}

		void push(int x, double y) {
			nodes.push_back(make_pair(x, (int)round(y)));
		}

		void push(int x, int y) {
			nodes.push_back(make_pair(x, y));
		}

		void clear() {
			nodes.clear();
		}

	private:
		void calc_range() {
			y_min = 1000;
			y_max = -1000;
			for (const int &val: show_nodes) {
				y_min = min(y_min, val);
				y_max = max(y_max, val);
			}
			y_min = min(y_min, 0);
			y_max = max(y_max, 0);
			y_min = y_min - resolution + abs(y_min) % resolution;
			y_max = y_max + resolution - abs(y_max) % resolution;
		}
};
