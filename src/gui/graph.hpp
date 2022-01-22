#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include "../board.hpp"

using namespace std;

constexpr Color graph_color = Palette::Black;
constexpr Color graph_fork_color = Palette::Purple;
constexpr Color graph_place_color = Palette::Darkblue;

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
	int dy;
	int dx;
	int adj_y;
	int adj_x;

public:
	void draw(vector<board> nodes1, vector<board> nodes2, int place) {
		calc_range(nodes1, nodes2);
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
		draw_graph(nodes1, graph_color);
		draw_graph(nodes2, graph_fork_color);
		int place_x = sx + place * dx + place * adj_x / 60;
		Line(place_x, sy, place_x, sy + size_y).draw(3, graph_place_color);
	}

	int update_place(vector<board> nodes1, vector<board> nodes2, int place) {
		if (Rect(sx - 40, sy, size_x + 20, size_y).leftPressed()) {
			int cursor_x = Cursor::Pos().x;
			int min_err = inf;
			for (int i = 0; i < (int)nodes1.size(); ++i) {
				int x = sx + (nodes1[i].n - 4) * dx + (nodes1[i].n - 4) * adj_x / 60;
				if (abs(x - cursor_x) < min_err) {
					min_err = abs(x - cursor_x);
					place = nodes1[i].n - 4;
				}
			}
			for (int i = 0; i < (int)nodes2.size(); ++i) {
				int x = sx + (nodes2[i].n - 4) * dx + (nodes2[i].n - 4) * adj_x / 60;
				if (abs(x - cursor_x) < min_err) {
					min_err = abs(x - cursor_x);
					place = nodes2[i].n - 4;
				}
			}
		}
		return place;
	}

private:
	void calc_range(vector<board> nodes1, vector<board> nodes2) {
		y_min = 1000;
		y_max = -1000;
		for (const board& b : nodes1) {
			if (b.v != -inf) {
				y_min = min(y_min, b.v);
				y_max = max(y_max, b.v);
			}
		}
		for (const board& b : nodes2) {
			if (b.v != -inf) {
				y_min = min(y_min, b.v);
				y_max = max(y_max, b.v);
			}
		}
		y_min = min(y_min, 0);
		y_max = max(y_max, 0);
		y_min = y_min - resolution + abs(y_min) % resolution;
		y_max = y_max + resolution - abs(y_max) % resolution;
		dy = size_y / (y_max - y_min);
		dx = size_x / 60;
		adj_y = size_y - dy * (y_max - y_min);
		adj_x = size_x - dx * 60;
	}

	void draw_graph(vector<board> nodes, Color color) {
		for (const board& b : nodes) {
			if (b.v != -inf) {
				int yy = sy + (y_max - b.v) * dy + adj_y * (y_max - b.v) / (y_max - y_min);
				Circle{ sx + (b.n - 4) * dx + (b.n - 4) * adj_x / 60, yy, 4 }.draw(color);
			}
			else {
				int yy = sy + y_max * dy + adj_y * y_max / (y_max - y_min);
				Circle{ sx + (b.n - 4) * dx + (b.n - 4) * adj_x / 60, yy, 4 }.draw(color);
			}
		}
		int idx1 = 0, idx2 = 0;
		while (idx2 < (int)nodes.size()) {
			while (idx1 < (int)nodes.size()) {
				if (nodes[idx1].v != -inf)
					break;
				++idx1;
			}
			if (idx1 >= (int)nodes.size())
				break;
			int xx1 = sx + (nodes[idx1].n - 4) * dx + (nodes[idx1].n - 4) * adj_x / 60;
			int yy1 = sy + (y_max - nodes[idx1].v) * dy + adj_y * (y_max - nodes[idx1].v) / (y_max - y_min);
			idx2 = idx1 + 1;
			while (idx2 < (int)nodes.size()) {
				if (nodes[idx2].v != -inf)
					break;
				++idx2;
			}
			if (idx2 >= (int)nodes.size())
				break;
			int xx2 = sx + (nodes[idx2].n - 4) * dx + (nodes[idx2].n - 4) * adj_x / 60;
			int yy2 = sy + (y_max - nodes[idx2].v) * dy + adj_y * (y_max - nodes[idx2].v) / (y_max - y_min);
			Line(xx1, yy1, xx2, yy2).draw(4, color);
			idx1 = idx2;
		}
	}
};
