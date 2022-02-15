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
			arr.push_back(make_pair(b, name));
			b.white_mirror();
			arr.push_back(make_pair(b, name));
			b.vertical_mirror();
			arr.push_back(make_pair(b, name));
			b.white_mirror();
			arr.push_back(make_pair(b, name));
		}
		return true;
	}

	inline string get(board b) {
		int i, j;
		bool flag;
		for (i = 0; i < (int)arr.size(); ++i) {
			if (arr[i].first.b == b.b && arr[i].first.w == b.w)
				return arr[i].second;
		}
		return "";
	}

};

joseki joseki;

bool joseki_init() {
	return joseki.init();
}
