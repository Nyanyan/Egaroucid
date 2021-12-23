with open('learned_data/book.txt') as f:
    data = f.read().splitlines()
with open('learned_data/new_book.txt', 'w') as f:
    for datum in data:
        board, val = datum.split()
        val = float(val)
        val = round(val)
        f.write(board + ' ' + str(val) + '\n')