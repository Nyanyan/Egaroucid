import random
import sys

from othello_py2 import Othello


MAX_MOVES = 60


def move_to_str(y, x):
    return chr(ord('a') + x) + str(y + 1)


def generate_random_opening(n_moves):
    othello = Othello()
    record = []

    while len(record) < n_moves:
        if not othello.has_legal():
            othello.move_pass()
            if not othello.has_legal():
                return None

        y, x = random.choice(othello.get_legal_moves())
        if not othello.move(y, x):
            return None
        record.append(move_to_str(y, x))

    return ''.join(record)


def parse_args(argv):
    if len(argv) != 3:
        print('usage: python generate_random_openings.py <n_moves> <n_games>', file=sys.stderr)
        return None

    try:
        n_moves = int(argv[1])
        n_games = int(argv[2])
    except ValueError:
        print('<n_moves> and <n_games> must be integers', file=sys.stderr)
        return None

    if not 0 <= n_moves <= MAX_MOVES:
        print(f'<n_moves> must be between 0 and {MAX_MOVES}', file=sys.stderr)
        return None
    if n_games < 0:
        print('<n_games> must be non-negative', file=sys.stderr)
        return None

    return n_moves, n_games


def main():
    args = parse_args(sys.argv)
    if args is None:
        return 1

    n_moves, n_games = args
    for _ in range(n_games):
        record = None
        while record is None:
            record = generate_random_opening(n_moves)
        print(record)

    return 0


if __name__ == '__main__':
    sys.exit(main())
