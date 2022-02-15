#pragma once
#include <iostream>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"

using namespace std;

struct umigame_result {
	int b;
	int w;
	umigame_result operator+(const umigame_result& other) {
		umigame_result res;
		res.b = b + other.b;
		res.w = w + other.w;
		return res;
	}
};

#define UMIGAME_SEARCH_DEPTH 100

class Umigame{
	public:
		umigame_result get(Board *b) {
			umigame_result res;
			res.b = search(b, UMIGAME_SEARCH_DEPTH, BLACK);
			res.w = search(b, UMIGAME_SEARCH_DEPTH, WHITE);
			return res;
		}

	private:
		int search(Board *b, int depth, int player){
			if (depth == 0)
				return 1;
			if (!global_searching)
				return 0;
			Board nb;
			int val, max_val = -INF;
			vector<Board> boards;
			unsigned long long legal = b->mobility_ull();
			Mobility mob;
			for (int i = 0; i < HW2; ++i){
				if (1 & (legal >> i)) {
					calc_flip(&mob, b, i);
					nb = b->move_copy(&mob);
					val = book.get(&nb);
					if (val != -INF && val >= max_val) {
						if (val > max_val) {
							boards.clear();
							max_val = val;
						}
						boards.emplace_back(nb);
					}
				}
			}
			if (boards.size() == 0)
				return 1;
			int res;
			if (b->p == player) {
				res = INF;
				for (Board nnb : boards)
					res = min(res, search(&nnb, depth - 1, player));
			} else {
				res = 0;
				for (Board nnb : boards)
					res += search(&nnb, depth - 1, player);
			}
			return res;
		}
};

Umigame umigame;
