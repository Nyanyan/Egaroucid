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

#define umigame_search_depth 100

class umigame{
	public:
		umigame_result get(board* b) {
			umigame_result res;
			res.b = search(b, umigame_search_depth, black);
			res.w = search(b, umigame_search_depth, white);
			return res;
		}

	private:
		int search(board *b, int depth, int player){
			if (depth == 0)
				return 1;
			board nb;
			int val, max_val = -inf;
			vector<board> boards;
			for (int i = 0; i < hw2; ++i){
				if (b->legal(i)) {
					b->move(i, &nb);
					val = -book.get(&nb);
					if (val != inf && val >= max_val) {
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
				res = inf;
				for (board nnb : boards)
					res = min(res, search(&nnb, depth - 1, player));
			} else {
				res = 0;
				for (board nnb : boards)
					res += search(&nnb, depth - 1, player);
			}
			return res;
		}
};

umigame umigame;
