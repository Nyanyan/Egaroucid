from functools import lru_cache
from pathlib import Path

from othello import normalize_board_text


WORK_DIR = Path(__file__).resolve().parent
ROOT_DIR = WORK_DIR.parents[2]
DATA_DIR = WORK_DIR / "data"
START_DIR = DATA_DIR / "records321_14_random_setup"
GAME_RECORDS_DIR = DATA_DIR / "game_records"
BOOK_RECORDS_DIR = DATA_DIR / "book_records"
TRAINED_DIR = WORK_DIR / "trained"
BOOK_DIR = TRAINED_DIR
CONSOLE_EXE = ROOT_DIR / "bin" / "Egaroucid_for_Console.exe"

DEFAULT_LEVEL = 19
DEFAULT_THREADS = 1
DEFAULT_GAMES_PER_START = 512
DEFAULT_RECORD_BATCH_SIZE = 16
DEFAULT_MAX_LOSS_PER_MOVE = 2
DEFAULT_MAX_LOSS_TOTAL = 4
DEFAULT_BOOK_MAX_LOSS = 4
DEFAULT_CUT_EMPTY = 28
BOOK_EXTENSION = ".egcb"
START_INDEX_WIDTH = 7


def sanitize_board_name(board: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in board)


def iter_start_boards(start_dir: Path = START_DIR):
    for path in sorted(start_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                board = line.strip()
                if board:
                    yield board


@lru_cache(maxsize=None)
def start_board_indices(start_dir: Path = START_DIR) -> dict[str, int]:
    return {
        normalize_board_text(board): idx
        for idx, board in enumerate(iter_start_boards(start_dir))
    }


def board_name_for_start(board: str) -> str:
    normalized_board = normalize_board_text(board)
    name = sanitize_board_name(normalized_board)
    idx = start_board_indices().get(normalized_board)
    if idx is None:
        return name
    return f"{idx:0{START_INDEX_WIDTH}d}_{name}"


def record_dir_for_start(board: str) -> Path:
    return BOOK_RECORDS_DIR / board_name_for_start(board)


def book_path_for_start(board: str) -> Path:
    return BOOK_DIR / f"{board_name_for_start(board)}{BOOK_EXTENSION}"
