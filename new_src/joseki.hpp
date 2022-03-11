#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "board.hpp"

using namespace std;



class Joseki {
private:
	vector<pair<Board, string>> arr;
public:
	bool init() {
		ifstream ifs("resources/joseki.txt");
		if (ifs.fail()) {
			cerr << "joseki file not found" << endl;
			return false;
		}
		string line;
		string name;
		int board_arr[HW2];
		int i;
		Board b;
		while (getline(ifs, line)) {
			for (i = 0; i < HW2; ++i) {
				if (line[i] == '0')
					board_arr[i] = BLACK;
				else if (line[i] == '1')
					board_arr[i] = WHITE;
				else
					board_arr[i] = VACANT;
			}
			b.translate_from_arr(board_arr, BLACK);
			name = line.substr(65, line.size());
			arr.push_back(make_pair(b, name));
			b.board_white_line_mirror();
			arr.push_back(make_pair(b, name));
			b.board_vertical_mirror();
			arr.push_back(make_pair(b, name));
			b.board_black_line_mirror();
			arr.push_back(make_pair(b, name));
		}
		return true;
	}

	inline string get(Board b) {
		int i, j;
		bool flag;
		for (i = 0; i < (int)arr.size(); ++i) {
			if (b.player == BLACK && arr[i].first.player == b.player && arr[i].first.opponent == b.opponent)
				return arr[i].second;
			if (b.player == WHITE && arr[i].first.player == b.opponent && arr[i].first.opponent == b.player)
				return arr[i].second;
		}
		return "";
	}

};

Joseki joseki;

bool joseki_init() {
	return joseki.init();
}
