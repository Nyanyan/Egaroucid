from tqdm import tqdm

with open('eval_test_data_bestmove.txt', 'r') as f:
    data = f.read().splitlines()

with open('eval_test_data_bestmove_board.txt', 'w') as f:
    f.write('%\n')
    for datum in tqdm(data):
        board = datum[:64]
        bestmove = int(datum[64:])
        bestmove_str = chr(ord('A') + bestmove % 8) + str(bestmove // 8 + 1)
        board_proc = board.replace('0', 'O').replace('1', '*').replace('.', '-')
        s = board_proc + ' O' + '\n%\n'
        f.write(s)

with open('eval_test_data_bestmove_move.txt', 'w') as f:
    f.write('%\n')
    for datum in tqdm(data):
        board = datum[:64]
        bestmove = int(datum[64:])
        bestmove_str = chr(ord('A') + bestmove % 8) + str(bestmove // 8 + 1)
        board_proc = board.replace('0', 'O').replace('1', '*').replace('.', '-')
        s = bestmove_str + '\n%\n'
        f.write(s)