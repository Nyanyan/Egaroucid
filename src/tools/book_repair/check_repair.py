"""Exercise a real repair binary against an independent small Othello fixture."""
import argparse
import json
from pathlib import Path
import random
import struct
import subprocess

ROOT = Path(__file__).absolute().parents[3]
DIRECTIONS = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy, dx) != (0, 0)]


def flips(player, opponent, cell):
    if (player | opponent) >> cell & 1:
        return 0
    result = 0
    y, x = divmod(cell, 8)
    for dy, dx in DIRECTIONS:
        row, col = y + dy, x + dx
        line = 0
        while 0 <= row < 8 and 0 <= col < 8 and (opponent >> (row * 8 + col) & 1):
            line |= 1 << (row * 8 + col)
            row += dy
            col += dx
        if 0 <= row < 8 and 0 <= col < 8 and (player >> (row * 8 + col) & 1):
            result |= line
    return result


def legal(board):
    return [cell for cell in range(64) if flips(*board, cell)]


def moved(board, cell):
    player, opponent = board
    changed = flips(player, opponent, cell)
    assert changed
    return opponent & ~changed, player | changed | (1 << cell)


def transformed(board, symmetry):
    result = [0, 0]
    for side, bits in enumerate(board):
        for cell in range(64):
            if not (bits >> cell & 1):
                continue
            y, x = divmod(cell, 8)
            if symmetry & 1:
                y, x = x, y
            if symmetry & 2:
                y = 7 - y
            if symmetry & 4:
                x = 7 - x
            result[side] |= 1 << (8 * y + x)
    return tuple(result)


def canonical(board):
    return min(transformed(board, symmetry) for symmetry in range(8))


def fixture(path):
    random_source = random.Random(643)
    boards = set()
    passes = 0
    for _ in range(16):
        board = (0x0000000810000000, 0x0000001008000000)
        while True:
            boards.add(canonical(board))
            moves = legal(board)
            if not moves:
                board = board[::-1]
                boards.add(canonical(board))
                if not legal(board):
                    break
                passes += 1
            board = moved(board, random_source.choice(legal(board)))
    assert passes
    ordered = sorted(boards)
    data = b'DICUORAGE\x03' + struct.pack('<I', len(ordered))
    # Every leaf is deliberately damaged, including apparently legal moves.
    for i, (player, opponent) in enumerate(ordered):
        data += struct.pack('<QQbbIbbb', player, opponent, i % 33 - 16, 1,
                            i + 1, 127, i % 64, [1, -1, 70][i % 3])
    path.write_bytes(data)
    return ordered, passes


def validate(original, repaired, boards):
    before = original.read_bytes()
    after = repaired.read_bytes()
    assert len(before) == len(after) and before[:14] == after[:14]
    keys = set(boards)
    no_move = 0
    exact_checks = 0

    def solve(board):
        moves = legal(board)
        if moves:
            return max(-solve(moved(board, cell)) for cell in moves)
        if legal(board[::-1]):
            return -solve(board[::-1])
        player, opponent = (bin(bits).count('1') for bits in board)
        return 64 - 2 * opponent if player > opponent else 2 * player - 64 if player < opponent else 0

    for index, board in enumerate(boards):
        p = 14 + index * 25
        assert before[p:p+22] == after[p:p+22]
        remaining = []
        for cell in legal(board):
            child = moved(board, cell)
            if not legal(child):
                passed = child[::-1]
                registered = canonical(passed) in keys
                if not legal(passed):
                    registered |= canonical(child) in keys
            else:
                registered = canonical(child) in keys
            if not registered:
                remaining.append(cell)
        value, move, level = struct.unpack_from('bbb', after, p + 22)
        if remaining:
            assert -64 <= value <= 64 and move in remaining and level == 1
            if bin(board[0] | board[1]).count('1') >= 62:
                assert value == max(-solve(moved(board, cell)) for cell in remaining)
                assert value == -solve(moved(board, move))
                exact_checks += 1
            # Independently check all eight orientations, including 90/270 rotations.
            for symmetry in range(8):
                transformed_move = transformed((1 << move, 0), symmetry)[0].bit_length() - 1
                assert transformed_move in legal(transformed(board, symmetry))
        else:
            assert (value, move, level) == (-126, 65, -1)
            no_move += 1
    assert no_move and exact_checks
    print(f'PASS: independent rules, pass/terminal membership, 8 symmetries, {len(boards)} records, {no_move} no-move leaves, {exact_checks} exact scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.work_dir.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    original = directory / 'fixture.egbk3'
    output = directory / 'repaired.egbk3'
    checkpoint = directory / 'leaves.checkpoint'
    boards, passes = fixture(original)
    base = [str(args.exe.absolute()), '--input', str(original), '--output', str(output)]
    repair = base + ['--mode', 'repair', '--resources', str(ROOT / 'bin/resources'),
                     '--checkpoint', str(checkpoint), '--threads', '4', '--report', str(directory / 'report.json')]

    def run(arguments, success=True):
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=45)
        (directory / ('run-' + str(len(list(directory.glob('run-*.log')))) + '.log')).write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) == success, result.stdout + result.stderr

    run(repair + ['--limit', '7'])
    assert not output.exists()
    saved = checkpoint.read_bytes()
    checkpoint.write_bytes(saved + b'incomplete row')
    run(repair)
    assert output.exists()
    validate(original, output, boards)
    from run_repair import verify_checkpoint
    verify_checkpoint(output, checkpoint, json.loads((directory / 'report.json').read_text()))
    run(base + ['--mode', 'verify'])
    # Fail closed on a changed non-leaf byte or on an illegal leaf.
    repaired = output.read_bytes()
    broken = bytearray(repaired)
    broken[14 + 16] ^= 1
    output.write_bytes(broken)
    run(base + ['--mode', 'verify'], False)
    broken = bytearray(repaired)
    broken[14 + 22] = 127
    output.write_bytes(broken)
    run(base + ['--mode', 'verify'], False)
    output.write_bytes(repaired)
    # Duplicate completed rows and changed input identity cannot resume.
    journal = checkpoint.read_bytes()
    checkpoint.write_bytes(journal + journal.splitlines(keepends=True)[1])
    output.unlink()
    run(repair, False)
    checkpoint.write_bytes(journal)
    changed = bytearray(original.read_bytes())
    changed[14 + 16] ^= 1
    original.write_bytes(changed)
    run(repair, False)
    bad = directory / 'truncated.egbk3'
    bad.write_bytes(original.read_bytes()[:-1])
    run([str(args.exe.absolute()), '--mode', 'audit', '--input', str(bad)], False)
    # Installation verifies backups and cannot overwrite changes made during a run.
    from run_repair import install, sha256
    candidate = directory / 'candidate.egbk3'
    candidate.write_bytes(repaired)
    destination = directory / 'installed.egbk3'
    destination.write_bytes(b'original destination')
    expected = sha256(destination)
    destination.write_bytes(b'changed during calculation')
    try:
        install(candidate, destination, directory, expected)
        raise AssertionError('changed destination was overwritten')
    except RuntimeError:
        pass
    assert destination.read_bytes() == b'changed during calculation'
    destination.write_bytes(b'original destination')
    install(candidate, destination, directory, expected)
    assert destination.read_bytes() == repaired
    assert (directory / ('backup-' + expected + '.egbk3')).read_bytes() == b'original destination'
    print(f'PASS: resume, partial-write recovery, corruption rejection, identity checks; passes={passes}')


if __name__ == '__main__':
    main()
