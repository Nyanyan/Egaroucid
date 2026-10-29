"""Independent, deliberately simple rules oracle (A1-first array, no bitboards)."""

from common import require


class Board:
    def __init__(self):
        self.cells = list("-" * 27 + "WB" + "-" * 6 + "BW" + "-" * 27)
        self.turn = "B"

    def flips(self, move):
        x, y = ord(move[0]) - ord("a"), int(move[1]) - 1
        if not (0 <= x < 8 and 0 <= y < 8) or self.cells[y * 8 + x] != "-":
            return []
        opponent = "W" if self.turn == "B" else "B"
        result = []
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            xx, yy, line = x + dx, y + dy, []
            while 0 <= xx < 8 and 0 <= yy < 8 and self.cells[yy * 8 + xx] == opponent:
                line.append(yy * 8 + xx)
                xx, yy = xx + dx, yy + dy
            if line and 0 <= xx < 8 and 0 <= yy < 8 and self.cells[yy * 8 + xx] == self.turn:
                result.extend(line)
        return result

    def legal(self):
        return {f"{chr(97 + x)}{y + 1}" for y in range(8) for x in range(8)
                if self.flips(f"{chr(97 + x)}{y + 1}")}

    def pass_turn(self):
        self.turn = "W" if self.turn == "B" else "B"

    def play(self, move):
        move = move.lower()
        if move == "ps":
            require(not self.legal(), "Unexpected pass")
            self.pass_turn()
            return
        flips = self.flips(move)
        require(flips, f"Illegal move from engine: {move}")
        self.cells[(int(move[1]) - 1) * 8 + ord(move[0]) - 97] = self.turn
        for cell in flips:
            self.cells[cell] = self.turn
        self.pass_turn()

    def replay(self, transcript):
        require(len(transcript) % 2 == 0, "Odd transcript length")
        for i in range(0, len(transcript), 2):
            move = transcript[i:i + 2]
            if move != "ps" and not self.legal():
                self.pass_turn()
            self.play(move)
        return self

    def ended(self):
        if self.legal():
            return False
        self.pass_turn()
        ended = not self.legal()
        self.pass_turn()
        return ended
