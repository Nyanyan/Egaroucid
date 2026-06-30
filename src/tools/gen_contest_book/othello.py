from __future__ import annotations

from dataclasses import dataclass


BOARD_SIZE = 8
N_CELLS = BOARD_SIZE * BOARD_SIZE
PASS_MOVE = -1
DIRS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


def is_black_like(c: str) -> bool:
    return c in "XxB b0"


def is_white_like(c: str) -> bool:
    return c in "OoWw1"


def normalize_board_text(text: str) -> str:
    compact = "".join(text.split())
    if len(compact) != N_CELLS + 1:
        raise ValueError(f"invalid board length {len(compact)}")
    side = "O" if is_white_like(compact[N_CELLS]) else "X"
    cells = []
    for c in compact[:N_CELLS]:
        if is_black_like(c):
            cells.append("X")
        elif is_white_like(c):
            cells.append("O")
        else:
            cells.append("-")
    return "".join(cells) + " " + side


def coord_to_index(coord: str) -> int:
    if len(coord) != 2:
        raise ValueError(f"invalid coord {coord}")
    x = ord(coord[0].lower()) - ord("a")
    y = ord(coord[1]) - ord("1")
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"invalid coord {coord}")
    return y * BOARD_SIZE + x


def index_to_coord(index: int) -> str:
    x = index % BOARD_SIZE
    y = index // BOARD_SIZE
    return f"{chr(ord('a') + x)}{y + 1}"


@dataclass
class Board:
    cells: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Board":
        normalized = normalize_board_text(text)
        cells = list(normalized[:N_CELLS])
        side = normalized[N_CELLS + 1]
        if side == "O":
            cells = ["X" if c == "O" else "O" if c == "X" else "-" for c in cells]
        return cls(cells)

    def copy(self) -> "Board":
        return Board(self.cells.copy())

    def key(self) -> str:
        return "".join(self.cells) + " X"

    def n_discs(self) -> int:
        return sum(c != "-" for c in self.cells)

    def _flips_for_move(self, index: int) -> list[int]:
        if self.cells[index] != "-":
            return []
        x0 = index % BOARD_SIZE
        y0 = index // BOARD_SIZE
        flips: list[int] = []
        for dx, dy in DIRS:
            x = x0 + dx
            y = y0 + dy
            line: list[int] = []
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                idx = y * BOARD_SIZE + x
                c = self.cells[idx]
                if c == "O":
                    line.append(idx)
                elif c == "X":
                    if line:
                        flips.extend(line)
                    break
                else:
                    break
                x += dx
                y += dy
        return flips

    def legal_moves(self) -> list[int]:
        return [idx for idx, c in enumerate(self.cells) if c == "-" and self._flips_for_move(idx)]

    def play(self, index: int) -> None:
        flips = self._flips_for_move(index)
        if not flips:
            raise ValueError(f"illegal move {index_to_coord(index)} on {self.key()}")
        self.cells[index] = "X"
        for idx in flips:
            self.cells[idx] = "X"
        self.pass_turn()

    def pass_turn(self) -> None:
        self.cells = ["X" if c == "O" else "O" if c == "X" else "-" for c in self.cells]

    def is_end(self) -> bool:
        if self.legal_moves():
            return False
        board = self.copy()
        board.pass_turn()
        return not board.legal_moves()

    def score_player(self) -> int:
        n_player = self.cells.count("X")
        n_opponent = self.cells.count("O")
        n_empty = self.cells.count("-")
        if n_player > n_opponent:
            n_player += n_empty
        elif n_opponent > n_player:
            n_opponent += n_empty
        return n_player - n_opponent
