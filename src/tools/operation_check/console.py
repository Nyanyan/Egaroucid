"""Black-box release checks for a built Egaroucid console application."""

import re

from common import Case, parser_for, require, run_suite
from reference_board import Board


LAST_MOVE_BOARD = "-W" + "B" * 62


def engine(ctx, commands="", extra=(), *, allow_errors=False, book=False):
    exe = ctx.application()
    options = ["-quiet", "-noboard", "-thread", "1", "-hash", "18", "-l", "1"]
    options += ["-book", exe.parent / "resources/book.egbk3"] if book else ["-nobook"]
    result = ctx.run([exe, *options, *extra], input=commands, cwd=ctx.directory)
    require(result.returncode == 0, f"Exit code {result.returncode}; see stderr log")
    # Release packages contain seeds for large hashes only. The engine explicitly
    # supports generating the small test hash; retain the warning in raw logs.
    result.stderr = result.stderr.replace("[ERROR] can't open hash18.eghs\n", "").replace(
        "[ERROR] can't get hash. you can ignore this error\n", "")
    if not allow_errors:
        require("[ERROR]" not in result.stderr + result.stdout, "Engine reported an error; see logs")
    return result


def metadata(ctx):
    require(re.search(r"Egaroucid.*\d+\.\d+", engine(ctx, extra=["-v"]).stdout), "Version missing")
    help_text = engine(ctx, extra=["-help"]).stdout
    require(all(option in help_text for option in ("-gtp", "-solve", "-thread", "-selfplay")), "CLI help incomplete")
    commands = engine(ctx, "help\nexit\n").stdout
    require(all(command in commands for command in ("chthread", "transcript", "setboard", "hint")), "Command help incomplete")


def argument_error(ctx):
    result = ctx.run([ctx.application(), "-thread"])
    require(result.returncode != 0 and "requires 1 argument" in result.stderr, "Missing argument was not rejected")


def perft(ctx):
    for mode in (1, 2):
        result = engine(ctx, extra=["-perft", "5", str(mode)])
        require(f"perft mode {mode} depth 5 1396 leaves" in result.stdout, "Incorrect depth-5 move tree")


def history(ctx):
    result = engine(ctx, "play F5d6\ntranscript\nundo\nredo\ntranscript\nundo\nplay f6\ntranscript\ninit\ntranscript\nexit\n")
    require(result.stdout.splitlines() == ["f5d6", "f5d6", "f5f6", ""], "Play/undo/redo/branch/reset history differs")


def illegal_moves(ctx):
    result = engine(ctx, "play f5\nplay d6a1\ntranscript\nplay e4\ntranscript\nplay z9\nplay a\ntranscript\nexit\n", allow_errors=True)
    require(result.stdout.splitlines() == ["f5", "f5", "f5"], "Invalid transcript partially changed the board")
    require(result.stderr.count("[ERROR]") == 4, "Invalid input error count differs")


def search(ctx):
    result = engine(ctx, "go\ninit\nplay f5\ngo\nexit\n")
    moves = result.stdout.splitlines()
    require(len(moves) == 2, "Expected two quiet search results")
    require(moves[0] in Board().legal(), "Initial search returned illegal move")
    require(moves[1] in Board().replay("f5").legal(), "Search after play returned illegal move")
    hints = engine(ctx, "hint 4\nexit\n").stdout
    hinted = re.findall(r"\|\s*([a-h][1-8])\s*\|", hints)
    require(len(hinted) == 4 and set(hinted) == Board().legal(), "Hints must list all four legal moves once")


def endgame(ctx):
    result = engine(ctx, f"setboard {LAST_MOVE_BOARD} B\ngo\nnew\ngo\nexit\n", ["-showvalue"])
    require(result.stdout.splitlines() == ["a1 64", "a1 64"], "Exact endgame score or new/setboard behavior differs")
    result = engine(ctx, f"setboard {LAST_MOVE_BOARD} W\nplay ps\ngo\nexit\n", ["-noautopass", "-showvalue"])
    require(result.stdout.splitlines() == ["a1 64"], "Explicit pass failed")


def threads(ctx, ponder=False):
    commands = "play f5\nchthread 3\ngo\nundo\nchthread 1\ngo\nchthread 0\nchthread abc\nchthread 2x\nchthread 2\nexit\n"
    result = engine(ctx, commands, ["-noise"] + (["-ponder"] if ponder else []), allow_errors=True)
    require(re.findall(r"thread size = (\d+)", result.stderr) == ["1", "3", "1", "2"], "Thread counts differ")
    require(result.stderr.count("[ERROR] thread count must be a positive integer") == 3, "Invalid thread counts accepted")
    moves = result.stdout.splitlines()
    require(len(moves) == 2 and all(move in Board().replay("f5").legal() for move in moves), "Search failed after thread change")


def gtp(ctx):
    commands = ["protocol_version", "name", "known_command genmove", "known_command nonsense",
                "gogui-rules_legal_moves", "play black F5", "gogui-rules_side_to_move",
                "gogui-rules_legal_moves", "undo", "gogui-rules_legal_moves", "clear_board",
                "genmove black", "clear_board", "play black A1", "gogui-rules_legal_moves", "quit"]
    result = engine(ctx, "".join(f"{i} {command}\n" for i, command in enumerate(commands, 1)), ["-gtp"])
    responses = re.split(r"\n\s*\n", result.stdout.strip())
    require(len(responses) == len(commands), "GTP response framing/count differs")
    parsed = {}
    for response in responses:
        match = re.fullmatch(r"([=?])(\d+)\s*(.*)", response, re.DOTALL)
        require(match, "Malformed GTP response: " + response)
        require(int(match[2]) not in parsed, "Duplicate GTP response ID")
        parsed[int(match[2])] = (match[1], match[3].strip())
    require(set(parsed) == set(range(1, len(commands) + 1)), "Missing GTP response ID")
    require(all(status == ("?" if i == 14 else "=") for i, (status, _) in parsed.items()), "Unexpected GTP status")
    require(parsed[1][1] == "2.0" and "Egaroucid" in parsed[2][1], "GTP identity differs")
    require(parsed[3][1] == "true" and parsed[4][1] == "false", "known_command differs")
    for index in (5, 10, 15):
        require(set(parsed[index][1].lower().split()) == Board().legal(), f"GTP initial legal moves differ at {index}")
    require(parsed[7][1] == "white", "GTP side-to-move did not change")
    require(set(parsed[8][1].lower().split()) == Board().replay("f5").legal(), "GTP move/flip differs")
    require(parsed[12][1].lower() in Board().legal(), "GTP genmove illegal")


def solve_file(ctx):
    problem = ctx.directory / "endgame.txt"
    problem.write_text(LAST_MOVE_BOARD + " B\n", encoding="ascii")
    result = engine(ctx, extra=["-solve", problem])
    require(re.search(r"\|\s*a1\s*\|\s*\+?64\s*\|", result.stdout), "Problem file score/move differs")
    require("total " in result.stdout, "Problem file summary missing")


def selfplay(ctx):
    result = engine(ctx, extra=["-selfplay", "1", "0"])
    transcripts = result.stdout.splitlines()
    require(len(transcripts) == 1 and re.fullmatch(r"(?:[a-h][1-8]|ps)+", transcripts[0]), "Self-play transcript missing/malformed")
    require(Board().replay(transcripts[0]).ended(), "Self-play did not reach a terminal legal position")


def book(ctx):
    # The staged release book is read; user-edited document/book.egbk3 is never copied.
    result = engine(ctx, "go\nexit\n", ["-noise"], book=True)
    require(result.stdout.strip() in Board().legal(), "Book search returned illegal move")
    require(re.search(r"imported [1-9]\d* boards to book", result.stderr), "Release book did not load any positions")
    require("level Book" in result.stderr, "Initial search did not use the book")


CASES = [
    Case("metadata", "Version, CLI help, command help", metadata),
    Case("argument-error", "Missing CLI argument returns an error", argument_error),
    Case("perft", "Depth-5 move count in both pass modes", perft),
    Case("history", "Play, undo, redo, branch, reset, transcript", history),
    Case("illegal-moves", "Invalid moves roll back without changing history", illegal_moves),
    Case("search", "Legal quiet search results and all initial hints", search),
    Case("endgame", "Exact score, setboard/new, explicit pass", endgame),
    Case("threads", "Runtime thread changes and invalid counts", threads),
    Case("ponder", "Thread changes/search with pondering", lambda ctx: threads(ctx, True)),
    Case("gtp", "Response framing, IDs, play, undo, search, errors", gtp),
    Case("solve-file", "Read and solve an exact endgame file", solve_file),
    Case("selfplay", "Full game checked with independent Othello rules", selfplay),
    Case("book", "Load release book and return a legal move", book),
]


if __name__ == "__main__":
    raise SystemExit(run_suite("console", parser_for("console", "Egaroucid_for_Console.exe"), CASES))
