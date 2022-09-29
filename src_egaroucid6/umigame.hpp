#pragma once
#include <iostream>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"

using namespace std;

struct Umigame_result {
	int b;
	int w;
	Umigame_result operator+(const Umigame_result& other) {
		Umigame_result res;
		res.b = b + other.b;
		res.w = w + other.w;
		return res;
	}
};

#define UMIGAME_SEARCH_DEPTH 100

class Umigame{
	public:
		Umigame_result get(Board *b, int player) {
			Umigame_result res;
			res.b = search(b, UMIGAME_SEARCH_DEPTH, player, BLACK);
			res.w = search(b, UMIGAME_SEARCH_DEPTH, player, WHITE);
			return res;
		}

	private:
		int search(Board *b, int depth, int player, const int target_player){
			if (depth == 0)
				return 1;
			if (!global_searching)
				return 0;
			Board nb;
			int val, max_val = -INF;
			vector<Board> boards;
			uint64_t legal = b->get_legal();
			if (legal == 0ULL){
				player ^= 1;
				b->pass();
				legal = b->get_legal();
			}
			Flip flip;
			for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
				calc_flip(&flip, b, cell);
				b->move_board(&flip);
					val = book.get(b);
					if (val != -INF && val >= max_val) {
						if (val > max_val) {
							boards.clear();
							max_val = val;
						}
						boards.emplace_back(b->copy());
					}
				b->undo_board(&flip);
			}
			if (boards.size() == 0)
				return 1;
			int res;
			if (player == target_player) {
				res = INF;
				for (Board &nnb : boards)
					res = min(res, search(&nnb, depth - 1, player ^ 1, target_player));
			} else {
				res = 0;
				for (Board &nnb : boards)
					res += search(&nnb, depth - 1, player ^ 1, target_player);
			}
			return res;
		}
};

Umigame umigame;
