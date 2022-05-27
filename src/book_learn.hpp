#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

#define BOOK_LEARN_UNDEFINED -INF

using namespace std;

struct Book_learn_node{
    Board board;
    int value;

    bool operator ==(const Book_learn_node& other) const noexcept{
		return board.player == other.board.player && board.opponent == other.board.opponent;
	}
}

// search value >= maximum value - expected_error
// register and search until depth
inline void learn_book(Board root_node, int expected_error, int depth){
    unordered_set<Book_learn_node> nodes;
    vector<Board> search_stack;
    Book_learn_node node;
    uint64_t legal;
    search_stack.emplace_back(root_node);
    while (search_stack.size()){
        Board board = search_stack.pop_back();
        node.board = board;
        legal = 
    }
}