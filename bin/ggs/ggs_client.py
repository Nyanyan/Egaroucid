import telnetlib
import subprocess
import datetime

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

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

def ggs_wait_ready():
    output = tn.read_until(b"READY", timeout=None).decode("utf-8")
    print(output)
    return output

def ggs_ask_game(tl1, tl2, tl3, turn, user):
    turn_str = 'b' if turn == 'X' else 'w'
    tl1_str = fill0(tl1 // 60, 2) + ':' + fill0(tl1 % 60, 2)
    tl2_str = fill0(tl2 // 60, 2) + ':' + fill0(tl2 % 60, 2)
    tl3_str = fill0(tl3 // 60, 2) + ':' + fill0(tl3 % 60, 2)
    cmd = 'ts ask 8' + turn_str + ' ' + tl1_str + '/' + tl2_str + '/' + tl3_str + ' ' + user
    print('[INFO]', 'ask game', cmd)
    tn.write((cmd + '\n').encode('utf-8'))
    ggs_wait_ready()
    ggs_wait_ready()

def ggs_receive_game():
    output = ggs_wait_ready().splitlines()
    for line in output:
        if line[:4] == '/os:':
            data = line.split()
            if len(data) == 10:
                game_id = data[2]
                opponent = data[4]
                raw_tls = data[5].split('/')
                tls = [0, 0, 0]
                for i in range(3):
                    if raw_tls[i] != '':
                        tls[i] = int(raw_tls[i].split(':')[0]) + int(raw_tls[i].split(':')[1]) / 60 # minutes
                    else:
                        tls[i] = 0
                game_type = data[6]
                return game_id, tls[0], tls[1], tls[2], opponent, game_type
    print('[ERROR]', 'cannot receive game')

def ggs_accept_game(game_id):
    print('[INFO]', 'accept game', game_id)
    tn.write(('ts accept ' + game_id + '\n').encode('utf-8'))
    ggs_wait_ready()

def ggs_get_board():
    data = []
    while len(data) < 9:
        data = ggs_wait_ready().splitlines()
    for datum in data:
        if len(datum) >= 4:
            if datum[:4] == '/os:':
                line = datum.split()
                game_id = line[2]
    print('[INFO]', 'game_id', game_id)

    me_color = ''
    me_remaining_time = 0
    for datum in data:
        if len(datum):
            if datum[0] == '|':
                if datum.find(ggs_id) >= 0:
                    line = datum.split()
                    if line[2][0] == '*':
                        me_color = 'X'
                    elif line[2][0] == 'O':
                        me_color = 'O'
                    else:
                        print('[ERROR]', 'invalid color', datum, line)
                    raw_me_remaining_time = line[3].split(',')[0]
                    me_remaining_time = int(raw_me_remaining_time.split(':')[0]) * 60 + int(raw_me_remaining_time.split(':')[1])
    print('[INFO]', 'me_color', me_color, 'me_remaining_time', me_remaining_time)
    raw_player = ''
    for datum in data:
        if len(datum):
            if datum[0] == '|':
                player_info_place = datum.find(' to move')
                if player_info_place >= 1:
                    raw_player = datum[player_info_place - 1]
                    break
    #print('[INFO]', 'raw_player', raw_player)
    if raw_player == '*':
        color_to_move = 'X'
    elif raw_player == 'O':
        color_to_move = 'O'
    else:
        print('[ERROR]', 'cannot recognize player', raw_player)
    print('[INFO]', 'color_to_move', color_to_move)

    raw_board = ''
    got_coord = False
    for datum in data:
        if len(datum):
            if datum[0] == '|':
                if datum.find("A B C D E F G H") >= 0:
                    if got_coord:
                        break
                    else:
                        got_coord = True
                if got_coord:
                    raw_board += datum
    #print(raw_board)
    board = raw_board.replace('A B C D E F G H', '').replace('\r', '').replace('\n', '').replace('|', '').replace(' ', '')
    for i in range(1, 9):
        board = board.replace(str(i), '')
    board = board.replace('*', 'X') + ' ' + color_to_move
    print('[INFO]', 'board', board)

    return game_id, me_color, me_remaining_time, board, color_to_move

def ggs_play_move(game_id, coord, value):
    cmd = 't /os play ' + game_id  + ' ' + coord + '/' + value
    tn.write((cmd + '\n').encode('utf-8'))

def idx_to_coord_str_rev(coord):
    x = coord % 8
    y = coord // 8
    return chr(ord('a') + x) + str(y + 1)

def egaroucid_play_move(egaroucid_idx, move):
    egaroucid[egaroucid_idx].stdin.write(('play ' + move + '\n').encode('utf-8'))
    egaroucid[egaroucid_idx].stdin.flush()

def egaroucid_settime(egaroucid_idx, color, time_limit):
    egaroucid[egaroucid_idx].stdin.write(('settime ' + color + ' ' + str(time_limit) + '\n').encode('utf-8'))
    egaroucid[egaroucid_idx].stdin.flush()

def egaroucid_setboard(egaroucid_idx, board):
    egaroucid[egaroucid_idx].stdin.write(('setboard ' + board + '\n').encode('utf-8'))
    egaroucid[egaroucid_idx].stdin.flush()

def egaroucid_get_move_score(egaroucid_idx):
    egaroucid[egaroucid_idx].stdin.write(('go\n').encode('utf-8'))
    egaroucid[egaroucid_idx].stdin.flush()
    line = egaroucid[egaroucid_idx].stdout.readline().decode().replace('\r', '').replace('\n', '')
    coord = line.split()[0]
    value = line.split()[1]
    return coord, value

# login
tn.read_until(b": Enter login (yours, or one you'd like to use).")
tn.write((ggs_id + '\n').encode('utf-8'))
tn.read_until(b": Enter your password.")
tn.write((ggs_pw + '\n').encode('utf-8'))
ggs_wait_ready()
ggs_wait_ready()
ggs_wait_ready()
ggs_wait_ready()
ggs_wait_ready()

# setup
tn.write(b"ms /os\n")
ggs_wait_ready()
tn.write(b"ts client -\n")
ggs_wait_ready()

print('[INFO]', 'initialized!')

while True:
    
    '''
    tl1 = 300 # seconds
    tl2 = 0
    tl3 = 0
    egaroucid_turn = 'O'
    # start game
    opponent = input('who do you want to ask game?: ')
    ggs_ask_game(tl1, tl2, tl3, egaroucid_turn, opponent)
    #'''

    #'''
    # receive game
    print('[INFO]', 'waiting request...')
    game_id, tl1, tl2, tl3, opponent, game_type = ggs_receive_game()
    ggs_accept_game(game_id)
    #'''

    synchro = game_type[0] == 's'


    # launch Egaroucid
    d_today = str(datetime.date.today())
    t_now = str(datetime.datetime.now().time())
    logfile = 'log/' + d_today.replace('-', '') + '_' + t_now.split('.')[0].replace(':', '') + '.txt'
    print('log file', logfile)
    egaroucid_cmd = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -noise -showvalue -noautopass -ponder -hash 27 -logfile ' + logfile
    egaroucids = []
    egaroucids.append(subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    if synchro:
        egaroucids.append(subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))

    while True:
        game_id, me_color, me_remaining_time, board, color_to_move = ggs_get_board()
        print('[INFO]', 'got board from GGS', game_id, me_color, me_remaining_time, board, color_to_move)
        egaroucid_idx = 0
        if synchro:
            egaroucid_idx = int(game_id.split()[1])
        if me_color == color_to_move:
            print('[INFO]', 'Egaroucid playing...')
            egaroucid_setboard(egaroucid_idx, board)
            egaroucid_settime(egaroucid_idx, me_color, me_remaining_time)
            coord, value = egaroucid_get_move_score(egaroucid_idx)
            print('[INFO]', 'got move from Egaroucid', coord, value)
            ggs_play_move(game_id, coord, value)

    for egaroucid in egaroucids:
        egaroucid.kill()
    break

tn.close()