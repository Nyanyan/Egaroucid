#pragma once
#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "board.hpp"

using namespace std;



class joseki {
private:
	vector<pair<board, string>> arr;
public:
	inline void init() {
		ifstream ifs("resources/joseki.txt");
		if (ifs.fail()) {
			cerr << "joseki file not found" << endl;
			exit(0);
		}
		string line;
		string name;
		int board_arr[hw2];
		int i;
		board b;
		while (getline(ifs, line)) {
			for (i = 0; i < hw2; ++i) {
				if (line[i] == '0')
					board_arr[i] = black;
				else if (line[i] == '1')
					board_arr[i] = white;
				else
					board_arr[i] = vacant;
			}
			b.translate_from_arr(board_arr, black);
			name = line.substr(65, line.size());
			int tmp[b_idx_num];
			arr.push_back(make_pair(b, name));
			for (i = 0; i < 8; ++i)
				swap(b.b[i], b.b[8 + i]);
			arr.push_back(make_pair(b, name));
			for (i = 0; i < 16; ++i)
				tmp[i] = b.b[i];
			for (i = 0; i < 8; ++i)
				b.b[i] = reverse_board[tmp[7 - i]];
			for (i = 0; i < 8; ++i)
				b.b[8 + i] = reverse_board[tmp[15 - i]];
			arr.push_back(make_pair(b, name));
			for (i = 0; i < 8; ++i)
				swap(b.b[i], b.b[8 + i]);
			arr.push_back(make_pair(b, name));
		}
	}

	inline string get(board b) {
		int i, j;
		bool flag;
		for (i = 0; i < (int)arr.size(); ++i) {
			flag = true;
			for (j = 0; j < hw; ++j)
				flag &= (b.b[j] == arr[i].first.b[j]);
			if (flag)
				return arr[i].second;
		}
		return "";
	}

};

joseki joseki;

inline void joseki_init() {
	joseki.init();
}
