#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include "../board.hpp"

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
	int y_max;
	int y_min;

public:
	void draw(vector<board> nodes) {
		calc_range(nodes);
		int dy = size_y / (y_max - y_min);
		int dx = size_x / 60;
		int adj_y = size_y - dy * (y_max - y_min);
		int adj_x = size_x - dx * 60;
		font(U"黒").draw(sx + 5, sy, graph_color);
		font(U"白").draw(sx + 5, sy + size_y - font_size * 1.5, graph_color);
		for (int y = 0; y <= y_max - y_min; y += resolution) {
			int yy = sy + y * dy + adj_y * y / (y_max - y_min);
			font(y_max - y).draw(sx - font(y_max - y).region(Point{ 0, 0 }).w - 12, yy - font(y_max - y).region(Point{ 0, 0 }).h / 2, graph_color);
			if (y_max - y == 0)
				Line{ sx, yy, sx + size_x, yy }.draw(2, graph_color);
			else
				Line{ sx, yy, sx + size_x, yy }.draw(1, graph_color);
		}
		for (int x = 0; x <= 60; x += 10) {
			font(x).draw(sx + x * dx + adj_x * x / 60 - font(x).region(Point{0, 0}).w, sy - 2 * font_size, graph_color);
			Line{ sx + x * dx + adj_x * x / 60, sy, sx + x * dx + adj_x * x / 60, sy + size_y }.draw(1, graph_color);
		}
		for (const board& b : nodes) {
			if (b.v != -inf) {
				int yy = sy + (y_max - (b.p ? 1 : -1) * b.v) * dy + adj_y * (y_max - (b.p ? 1 : -1) * b.v) / (y_max - y_min);
				Circle{ sx + (b.n - 4) * dx + (b.n - 4) * adj_x / 60, yy, 4}.draw(graph_color);
			}
		}
		if (nodes.size() >= 2) {
			int idx1 = 0, idx2 = 1;
			while (idx2 < (int)nodes.size()) {
				while (nodes[idx1].v == -inf) {
					++idx1;
				}
				int yy1 = sy + (y_max - (nodes[idx1].p ? 1 : -1) * nodes[idx1].v) * dy + adj_y * (y_max - (nodes[idx1].p ? 1 : -1) * nodes[idx1].v) / (y_max - y_min);
				idx2 = idx1 + 1;
				while (idx2 < (int)nodes.size()) {
					if (nodes[idx2].v != -inf)
						break;
					++idx2;
				}
				if (idx2 >= (int)nodes.size())
					break;
				int yy2 = sy + (y_max - (nodes[idx2].p ? 1 : -1) * nodes[idx2].v) * dy + adj_y * (y_max - (nodes[idx2].p ? 1 : -1) * nodes[idx2].v) / (y_max - y_min);
				Line(sx + (nodes[idx1].n - 4) * dx + (nodes[idx1].n - 4) * adj_x / 60, yy1, sx + (nodes[idx2].n - 4) * dx + (nodes[idx2].n - 4) * adj_x / 60, yy2).draw(4, graph_color);
				idx1 = idx2;
			}
		}
	}

private:
	void calc_range(vector<board> nodes) {
		y_min = 1000;
		y_max = -1000;
		for (const board& b : nodes) {
			if (b.v != -inf) {
				y_min = min(y_min, (b.p ? 1 : -1) * b.v);
				y_max = max(y_max, (b.p ? 1 : -1) * b.v);
			}
		}
		y_min = min(y_min, 0);
		y_max = max(y_max, 0);
		y_min = y_min - resolution + abs(y_min) % resolution;
		y_max = y_max + resolution - abs(y_max) % resolution;
	}
};
