#pragma once
#include <iostream>

using namespace std;

struct History_elem {
	Board board;
	int player;
	int v;
	int policy;
	int next_policy;
	String transcript;
	int level;

	History_elem() {
		board.reset();
		player = 0;
		v = 0;
		policy = -1;
		next_policy = -1;
		level = -1;
	}
};
