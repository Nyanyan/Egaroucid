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
		int adj_y;
		int y_max;
		int y_min;
	public:
		void draw() {
			sort(nodes.begin(), nodes.end());
			show_nodes.clear();
			for (int i = 0; i < (int)nodes.size() - 1; ++i) {
				//if (nodes[i].first + 1 == nodes[i + 1].first)
				//	show_nodes.emplace_back(round((double)(nodes[i].second + nodes[i + 1].second) / 2.0));
				//else
					show_nodes.emplace_back(nodes[i].second);
			}
			if (nodes.size())
				show_nodes.emplace_back(nodes[nodes.size() - 1].second);
			calc_range();
			dy = size_y / (y_max - y_min);
			dx = size_x / 60;
			adj_y = size_y - dy * (y_max - y_min);
			font(U"黒").draw(sx + 5, sy, graph_color);
			font(U"白").draw(sx + 5, sy + size_y - font_size * 1.5, graph_color);
			for (int y = 0; y <= y_max - y_min; y += resolution) {
				int yy = sy + y * dy + adj_y * y / (y_max - y_min);
				font(y_max - y).draw(sx - font_size * 3, yy - font_size, graph_color);
				if (y_max - y == 0)
					Line{sx, yy, sx + size_x, yy}.draw(2, graph_color);
				else
					Line{sx, yy, sx + size_x, yy}.draw(1, graph_color);
			}
			for (int x = 0; x <= 60; x += 10){
				font(x).draw(sx + x * dx, sy - 2 * font_size, graph_color);
				Line{sx + x * dx, sy, sx + x * dx, sy + size_y}.draw(1, graph_color);
			}
			for (int i = 0; i < (int)nodes.size(); ++i) {
				int yy = sy + (y_max - show_nodes[i]) * dy + adj_y * (y_max - show_nodes[i]) / (y_max - y_min);
				Circle{ sx + nodes[i].first * dx, yy, 3 }.draw(graph_color);
			}
			if (nodes.size() >= 2) {
				for (int i = 0; i < (int)nodes.size() - 1; ++i) {
					int yy1 = sy + (y_max - show_nodes[i]) * dy + adj_y * (y_max - show_nodes[i]) / (y_max - y_min);
					int yy2 = sy + (y_max - show_nodes[i + 1]) * dy + adj_y * (y_max - show_nodes[i + 1]) / (y_max - y_min);
					Line(sx + nodes[i].first * dx, yy1, sx + nodes[i + 1].first * dx, yy2).draw(4, graph_color);
				}
			}
		}

		void push(int x, double y) {
			nodes.push_back(make_pair(x, (int)round(y)));
		}

		void push(int x, int y) {
			nodes.push_back(make_pair(x, y));
		}

		void pop() {
			nodes.pop_back();
		}

		int last_x() {
			if (nodes.size())
				return nodes[nodes.size() - 1].first;
			return -1;
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
