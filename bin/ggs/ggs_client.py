import telnetlib
import subprocess
import datetime
import time

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


# launch Egaroucid
d_today = str(datetime.date.today())
t_now = str(datetime.datetime.now().time())
logfile = 'log/' + d_today.replace('-', '') + '_' + t_now.split('.')[0].replace(':', '') + '.txt'
print('log file', logfile)
egaroucid_cmd = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -t 8 -quiet -noise -showvalue -noautopass -hash 27 -logfile ' + logfile
egaroucid = subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)



# telnet
tn = telnetlib.Telnet(ggs_server, ggs_port)




def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

def ggs_wait_ready(timeout=None):
    output = tn.read_until(b"READY", timeout=timeout).decode("utf-8")
    if len(output):
        print(output)
    return output



def ggs_os_ask_game(tl1, tl2, tl3, turn, user):
    turn_str = 'b' if turn == 'X' else 'w'
    tl1_str = fill0(tl1 // 60, 2) + ':' + fill0(tl1 % 60, 2)
    tl2_str = fill0(tl2 // 60, 2) + ':' + fill0(tl2 % 60, 2)
    tl3_str = fill0(tl3 // 60, 2) + ':' + fill0(tl3 % 60, 2)
    cmd = 'ts ask 8' + turn_str + ' ' + tl1_str + '/' + tl2_str + '/' + tl3_str + ' ' + user
    print('[INFO]', 'ask game', cmd)
    tn.write((cmd + '\n').encode('utf-8'))




def ggs_os_get_info(s):
    ss = s.splitlines()
    for line in ss:
        if line[:4] == '/os:':
            return line
    return ''

def ggs_os_is_game_request(s):
    ss = s.split()
    if len(ss) == 10:
        return (ss[1] == '+' or ss[1] == '-') and ss[9] == ggs_id
    return False

def ggs_os_get_received_game_info(s):
    print('received game info')
    print(s)
    data = s.split()
    if len(data) == 10:
        game_id = data[2]
        opponent = data[4]
        raw_tls = data[5].split('/')
        tls = [0, 0, 0]
        for i in range(3):
            if raw_tls[i] != '':
                tls[i] = int(raw_tls[i].split(':')[0]) * 60 + int(raw_tls[i].split(':')[1]) # seconds
            else:
                tls[i] = 0
        game_type = data[6]
        return game_id, tls[0], tls[1], tls[2], opponent, game_type
    print('[ERROR]', 'cannot receive game:', s)

def ggs_os_accept_game(request_id):
    print('[INFO]', 'accept game', request_id)
    tn.write(('ts accept ' + request_id + '\n').encode('utf-8'))



def ggs_os_is_start_game(s):
    ss = s.split()
    if len(ss) > 3:
        return ss[2] == 'match'
    return False

def ggs_os_start_game_get_id(s):
    ss = s.split()
    if len(ss) > 4:
        return ss[3]
    print('[ERROR]', 'cannot receive game id:', s)


def ggs_os_is_board_info(s):
    ss = s.split()
    if len(ss) > 2:
        return ss[1] == 'update' or ss[1] == 'join'
    return False

def ggs_os_board_info_get_id(s):
    ss = s.split()
    if len(ss) >= 3:
        return ss[2]
    print('[ERROR]', 'cannot receive game id:', s)

def ggs_os_get_board(s):
    data = s.splitlines()

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
    #print('[INFO]', 'me_color', me_color, 'me_remaining_time', me_remaining_time)
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
    #print('[INFO]', 'color_to_move', color_to_move)

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
    #print('[INFO]', 'board', board)

    return me_color, me_remaining_time, board, color_to_move









def ggs_os_play_move(game_id, coord, value):
    cmd = 't /os play ' + game_id  + ' ' + coord + '/' + value
    tn.write((cmd + '\n').encode('utf-8'))




def ggs_os_is_synchro(game_id):
    return len(game_id.split('.')) == 3

def ggs_os_get_synchro_id(game_id):
    return game_id.split()[2]






def idx_to_coord_str_rev(coord):
    x = coord % 8
    y = coord // 8
    return chr(ord('a') + x) + str(y + 1)

def egaroucid_play_move(move):
    cmd = 'play ' + move
    print('[INFO]', 'egaroucid play', ':', cmd)
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_settime(color, time_limit):
    cmd = 'settime ' + color + ' ' + str(time_limit)
    print('[INFO]', 'egaroucid settime', ':', cmd)
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_setboard(board):
    cmd = 'setboard ' + board
    print('[INFO]', 'egaroucid setboard', ':', cmd)
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_get_move_score():
    cmd = 'go'
    print('[INFO]', 'egaroucid play', ':', cmd)
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
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


playing_game_id = ''

last_board_move_time = time.time()
latest_boards = ['', '']
latest_moves = ['', '']


while True:
    received_data = ggs_wait_ready()
    os_info = ggs_os_get_info(received_data)
    if ggs_os_is_game_request(os_info):
        print('[INFO]', 'GGS Game Request', ':', os_info)
        request_id, tl1, tl2, tl3, opponent, game_type = ggs_os_get_received_game_info(os_info)
        ggs_os_accept_game(request_id)
    elif ggs_os_is_start_game(os_info):
        print('[INFO]', 'GGS Game Start', ':', os_info)
        game_id = ggs_os_start_game_get_id(os_info)
        playing_game_id = game_id
    elif ggs_os_is_board_info(os_info):
        game_id = ggs_os_board_info_get_id(os_info)
        if game_id[:len(playing_game_id)] == playing_game_id:
            me_color, me_remaining_time, board, color_to_move = ggs_os_get_board(received_data)
            print('[INFO]', 'got board from GGS', 'game id', game_id, 'egaroucid_color', me_color, 'remaining_time', me_remaining_time, 'board', board, 'color_to_move', color_to_move)
            print('[INFO]', 'game_id', game_id, 'set board')
            egaroucid_setboard(board)
            sub_idx = 0
            if ggs_os_is_synchro(game_id):
                sub_idx = ggs_os_get_synchro_id[game_id]
            latest_boards[sub_idx] = board
            last_board_move_time = time.time()
            if me_color == color_to_move:
                me_remaining_time_proc = max(1, me_remaining_time - 10)
                egaroucid_settime(me_color, me_remaining_time_proc)
                print('[INFO]', 'game_id', game_id, 'Egaroucid playing...')
                coord, value = egaroucid_get_move_score()
                print('[INFO]', 'got move from Egaroucid', coord, value)
                ggs_os_play_move(game_id, coord, value)
                latest_moves[sub_idx] = coord
                last_board_move_time = time.time()

tn.close()