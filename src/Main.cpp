#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <vector>
#include "board.hpp"

using namespace std;

inline int proc_coord(int y, int x) {
	return y * hw + x;
}

void Main() {
	constexpr int offset_y = 60;
	constexpr int offset_x = 60;
	constexpr int cell_hw = 50;
	constexpr Size cell_size{cell_hw, cell_hw};
	constexpr int stone_size = 20;
	constexpr int legal_size = 5;
	constexpr int first_board[hw2] = {
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,white,black,vacant,vacant,vacant,
		vacant,vacant,vacant,black,white,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
		vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant
	};

	board_init();

	Array<Rect> cells;
	Array<Circle> stones, legals;
	vector<int> cell_center_y, cell_center_x;
	board bd;
	int bd_arr[hw2];

	for (auto p : step(Size{ hw, hw }))
		cells << Rect{ (offset_x + p.x * cell_size.x), (offset_y + p.y * cell_size.y), cell_size };

	for (int i = 0; i < hw; ++i) {
		cell_center_y.push_back(offset_y + i * cell_size.y + cell_hw / 2);
		cell_center_x.push_back(offset_x + i * cell_size.y + cell_hw / 2);
	}

	for (int i = 0; i < hw2; ++i) {
		stones << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], stone_size };
		legals << Circle{ cell_center_x[i % hw], cell_center_y[i / hw], legal_size };
	}

	for (int i = 0; i < hw2; ++i)
		bd_arr[i] = first_board[i];
	bd.translate_from_arr(bd_arr, black);

	

	while (System::Update()) {
		for (const auto& cell : cells)
			cell.stretched(-1).draw(Palette::Green);

		bd.translate_to_arr(bd_arr);
		for (int y = 0; y < hw; ++y) {
			for (int x = 0; x < hw; ++x) {
				int coord = proc_coord(y, x);
				//Print << coord << bd_arr[coord];
				if (bd_arr[coord] == black)
					stones[coord].draw(Palette::Black);
				else if (bd_arr[coord] == white)
					stones[coord].draw(Palette::White);
				else if (bd.legal(coord)) {
					legals[coord].draw(Palette::Blue);
					if (cells[coord].leftClicked()) {
						bd = bd.move(coord);
					}
				}
			}
		}
	}
}
