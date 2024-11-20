import telnetlib
import subprocess

# launch Egaroucid
egaroucid_cmd = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -noise -showvalue -ponder -logfile log/log.txt -hash 27 -time 120'
egaroucid = subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


with open('id/ggs_id.txt', 'r') as f:
    ggs_id = f.read()
with open('id/ggs_pw.txt', 'r') as f:
    ggs_pw = f.read()
with open('id/ggs_server.txt', 'r') as f:
    ggs_server = f.read()
with open('id/ggs_port.txt', 'r') as f:
    ggs_port = f.read()
print('GGS server', ggs_server, 'port', ggs_port)
print('GGS ID', ggs_id, 'GGS PW', ggs_pw)

# telnet
tn = telnetlib.Telnet(ggs_server, ggs_port)

def wait_ready():
    output = tn.read_until(b"READY", timeout=10).decode("utf-8")
    print(output)

def ggs_get_board():
    tn.read_until(b"A B C D E F G H", timeout=None).decode("utf-8")
    raw_board = tn.read_until(b"A B C D E F G H", timeout=None).decode("utf-8")
    print(raw_board)
    board = raw_board.replace('A B C D E F G H', '').replace('\r', '').replace('\n', '').replace('|', '').replace(' ', '')
    for i in range(1, 9):
        board = board.replace(str(i), '')
    board = board.replace('*', 'X')
    raw_player = tn.read_until(b"to move", timeout=None).decode("utf-8")
    print(raw_player)
    player_str = raw_player.replace('|', '').split()[-3]
    if player_str == '*':
        board += ' X'
    elif player_str == 'O':
        board += ' O'
    else:
        print('[ERROR]', 'cannot recognize player')
    print(board)
    return board

def ggs_play_move(coord, value):
    cmd = 't /os play ' + coord + '/' + value
    tn.write((cmd + '\n').encode('utf-8'))

def idx_to_coord_str_rev(coord):
    x = coord % 8
    y = coord // 8
    return chr(ord('a') + x) + str(y + 1)

def egaroucid_play_move(move):
    egaroucid.stdin.write(('play ' + move + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_setboard(board):
    egaroucid.stdin.write(('setboard ' + board + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_get_move_score():
    egaroucid.stdin.write(('go\n').encode('utf-8'))
    egaroucid.stdin.flush()
    line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '')
    coord = line.split()[0]
    value = line.split()[1]
    return coord, value

# login
tn.read_until(b": Enter login (yours, or one you'd like to use).")
tn.write((ggs_id + '\n').encode('utf-8'))
tn.read_until(b": Enter your password.")
tn.write((ggs_pw + '\n').encode('utf-8'))
wait_ready()

# alias
tn.write(b"ms /os\n")
wait_ready()

print('[INFO]', 'initialized!')

while True:
    # start game
    tn.write(b"ts ask 8w 05:00/00:00/02:00 nyanyan\n")
    wait_ready()
    wait_ready()

    egaroucid_turn = 'O'


    last_board = '---------------------------------------------------------------- ?'
    while True:
        board = ggs_get_board()
        print('[INFO]', 'got board from GGS', board)
        last_played_move = ''
        n_empties = 0
        n_diff = 0
        for i in range(64):
            if board[i] == '-':
                n_empties += 1
            if last_board[i] == '-' and board[i] != '-':
                last_played_move = idx_to_coord_str_rev(i)
                n_diff += 1
        if n_diff == 0:
            print('[INFO]', 'no move found. opponent passed?')
        elif n_diff == 1:
            print('[INFO]', 'last played', last_played_move)
        else:
            print('[INFO]', 'new board found')
        if n_empties == 0:
            print('[INFO]', 'game over')
        if board[-1] == egaroucid_turn:
            print('[INFO]', 'Egaroucid playing...')
            if n_diff == 0:
                pass
            elif n_diff == 1:
                egaroucid_play_move(last_played_move)
            else:
                egaroucid_setboard(board)
            coord, value = egaroucid_get_move_score()
            print('[INFO]', 'got move from Egaroucid', coord, value)
            ggs_play_move(coord, value)
        last_board = board

tn.close()