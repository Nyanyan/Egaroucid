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

# launch Egaroucid
d_today = str(datetime.date.today())
t_now = str(datetime.datetime.now().time())
logfile_egaroucid = 'log/' + d_today.replace('-', '') + '_' + t_now.split('.')[0].replace(':', '') + '_egaroucid.txt'
logfile_client = 'log/' + d_today.replace('-', '') + '_' + t_now.split('.')[0].replace(':', '') + '_client.txt'

color_dic = {"black":"\033[30m", "red":"\033[31m", "green":"\033[32m", "yellow":"\033[33m", "blue":"\033[34m", "end":"\033[0m"}

def print_log(*args):
    txt = ' '.join([str(elem) for elem in args])
    with open(logfile_client, 'a') as f:
        f.write(txt + '\n')
    print(txt)

def print_log_color(*args, color='red'):
    txt = ' '.join([str(elem) for elem in args])
    with open(logfile_client, 'a') as f:
        f.write(txt + '\n')
    print(color_dic[color] + txt + color_dic["end"])


print_log('log files', logfile_egaroucid, logfile_client)
egaroucid_cmd = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -t 8 -quiet -noise -ponder -showvalue -noautopass -hash 27 -logfile ' + logfile_egaroucid
egaroucid = subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)




# telnet
tn = telnetlib.Telnet(ggs_server, ggs_port)




def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

def ggs_wait_ready(timeout=None):
    output = tn.read_until(b"READY", timeout=timeout).decode("utf-8")
    if len(output):
        print_log(output)
    return output



def ggs_os_ask_game(tl1, tl2, tl3, game_type, user):
    tl1_str = fill0(tl1 // 60, 2) + ':' + fill0(tl1 % 60, 2)
    tl2_str = fill0(tl2 // 60, 2) + ':' + fill0(tl2 % 60, 2)
    tl3_str = fill0(tl3 // 60, 2) + ':' + fill0(tl3 % 60, 2)
    cmd = 'ts ask ' + game_type + ' ' + tl1_str + '/' + tl2_str + '/' + tl3_str + ' ' + user
    print_log_color('[INFO] ask game ' + cmd, color='green')
    tn.write((cmd + '\n').encode('utf-8'))




def ggs_os_get_info(s):
    ss = s.splitlines()
    for line in ss:
        if line[:4] == '/os:':
            return line
    return ''

def ggs_os_is_game_request(s):
    ss = s.split()
    if len(ss) == 10 or len(ss) == 11:
        return ss[1] == '+' and ss[9] == ggs_id
    return False

def ggs_os_get_received_game_info(s):
    data = s.split()
    if len(data) == 10 or len(data) == 11:
        game_id = data[2]
        opponent = data[4]
        if len(data) == 10:
            raw_tls = data[5].split('/')
        else:
            raw_tls = data[10].split('/')
        tls = [0, 0, 0]
        for i in range(3):
            if raw_tls[i] != '':
                tls[i] = int(raw_tls[i].split(':')[0]) * 60 + int(raw_tls[i].split(':')[1]) # seconds
            else:
                tls[i] = 0
        game_type = data[6]
        return game_id, tls[0], tls[1], tls[2], opponent, game_type
    print_log_color('[ERROR] cannot receive game : ' + s, color='red')

def ggs_os_accept_game(request_id):
    print_log_color('[INFO] accept game : ' + request_id, color='green')
    tn.write(('ts accept ' + request_id + '\n').encode('utf-8'))


def ggs_os_is_game_end(s):
    ss = s.split()
    if len(ss) >= 2:
        return ss[1] == 'end'
    return False

def ggs_os_is_start_game(s):
    ss = s.split()
    if len(ss) > 3:
        return ss[1] == '+' and ss[2] == 'match'
    return False

def ggs_os_is_game_terminated(s):
    ss = s.split()
    if len(ss) > 3:
        return ss[1] == '-'
    return False

def ggs_os_start_game_get_id(s):
    ss = s.split()
    if len(ss) > 4:
        return ss[3]
    print_log_color('[ERROR] cannot receive game id: ' + s, color='red')


def ggs_os_is_board_info(s):
    ss = s.split()
    if len(ss) > 2:
        return ss[1] == 'update' or ss[1] == 'join'
    return False

def ggs_os_board_info_get_id(s):
    ss = s.split()
    if len(ss) >= 3:
        return ss[2]
    print_log_color('[ERROR] cannot receive game id: ' + s, color='red')

def ggs_os_get_board(s):
    raw_data = s.splitlines()
    data = []
    for datum in raw_data:
        if len(datum):
            if datum[0] == '|':
                data.append(datum)

    try:
        last_move = data[0].split()[2][:2]
    except:
        last_move = ''

    me_color = ''
    me_remaining_time = 0
    for datum in data:
        if datum.find(ggs_id) >= 0:
            line = datum.split()
            if line[2][0] == '*':
                me_color = 'X'
            elif line[2][0] == 'O':
                me_color = 'O'
            else:
                print_log_color('[ERROR] invalid color: ' + datum, color='red')
            raw_me_remaining_time = line[3].split(',')[0]
            me_remaining_time = int(raw_me_remaining_time.split(':')[0]) * 60 + int(raw_me_remaining_time.split(':')[1])
    #print_log('[INFO]', 'me_color', me_color, 'me_remaining_time', me_remaining_time)
    raw_player = ''
    for datum in data:
        player_info_place = datum.find(' to move')
        if player_info_place >= 1:
            raw_player = datum[player_info_place - 1]
            break
    #print_log('[INFO]', 'raw_player', raw_player)
    if raw_player == '*':
        color_to_move = 'X'
    elif raw_player == 'O':
        color_to_move = 'O'
    else:
        print_log_color('[ERROR] cannot recognize player: ' + raw_player, color='red')
    #print_log('[INFO]', 'color_to_move', color_to_move)

    raw_board = ''
    got_coord = False
    for datum in data:
        if datum.find("A B C D E F G H") >= 0:
            if got_coord:
                break
            else:
                got_coord = True
        if got_coord:
            raw_board += datum
    #print_log(raw_board)
    board = raw_board.replace('A B C D E F G H', '').replace('\r', '').replace('\n', '').replace('|', '').replace(' ', '')
    for i in range(1, 9):
        board = board.replace(str(i), '')
    board = board.replace('*', 'X') + ' ' + color_to_move
    #print_log('[INFO]', 'board', board)

    return last_move, me_color, me_remaining_time, board, color_to_move









def ggs_os_play_move(game_id, coord, value):
    cmd = 't /os play ' + game_id  + ' ' + coord + '/' + value
    tn.write((cmd + '\n').encode('utf-8'))




def ggs_os_is_synchro(game_id):
    return len(game_id.split('.')) == 3

def ggs_os_get_synchro_id(game_id):
    return game_id.split('.')[2]






def idx_to_coord_str_rev(coord):
    x = coord % 8
    y = coord // 8
    return chr(ord('a') + x) + str(y + 1)

def egaroucid_play_move(move):
    cmd = 'play ' + move
    #print_log_color('[INFO] Egaroucid play : ' + cmd, color='green')
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_settime(color, time_limit):
    cmd = 'settime ' + color + ' ' + str(time_limit)
    #print_log_color('[INFO] Egaroucid settime : ' + cmd, color='green')
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_setboard(board):
    cmd = 'setboard ' + board
    #print_log_color('[INFO] Egaroucid setboard : ' + cmd, color='green')
    egaroucid.stdin.write((cmd + '\n').encode('utf-8'))
    egaroucid.stdin.flush()

def egaroucid_get_move_score():
    cmd = 'go'
    #print_log_color('[INFO] Egaroucid go : ' + cmd, color='green')
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

print_log_color('[INFO] Initialized!', color='green')


playing_game_id = ''
game_playing = False
asking_game = False

ponder_boards = []

transcripts = [['', []], ['', []]]


while True:
    '''
    if (not game_playing) and (not asking_game):
        ggs_os_ask_game(300, 0, 0, 's8r18', 'nyanyan')
        asking_game = True
    #'''
    received_data = ggs_wait_ready()
    os_info = ggs_os_get_info(received_data)
    if os_info != '':
        print_log_color('[INFO] GGS /os info : ' + os_info, color='green')
    if ggs_os_is_game_request(os_info):
        print_log_color('[INFO] GGS Game Request : ' + os_info, color='green')
        request_id, tl1, tl2, tl3, opponent, game_type = ggs_os_get_received_game_info(os_info)
        ggs_os_accept_game(request_id)
    elif ggs_os_is_start_game(os_info):
        print_log_color('[INFO] GGS Game Start : ' + os_info, color='green')
        game_id = ggs_os_start_game_get_id(os_info)
        playing_game_id = game_id
        print_log_color('[INFO] GGS Playing Game ID : ' + playing_game_id, color='green')
        game_playing = True
        asking_game = False
    elif ggs_os_is_game_end(os_info):
        print_log_color('[INFO] GGS Game End : ' + os_info, color='green')
        game_playing = False
        playing_game_id = ''
    elif ggs_os_is_game_terminated(os_info):
        print_log_color('[INFO] GGS Game Terminated : ' + os_info, color='green')
        game_playing = False
        asking_game = False
    elif ggs_os_is_board_info(os_info):
        game_id = ggs_os_board_info_get_id(os_info)
        game_id_nosub = game_id
        sub_idx = 0
        if ggs_os_is_synchro(game_id):
            sub_idx = ggs_os_get_synchro_id(game_id)
            game_id_nosub = game_id[:-len(str(sub_idx)) - 1]
        print_log_color('[INFO] GGS Received board game_id : ' + game_id_nosub, color='green')
        if game_id_nosub == playing_game_id:
            last_move, me_color, me_remaining_time, board, color_to_move = ggs_os_get_board(received_data)
            print_log_color('[INFO] GGS Got board from GGS game_id : ' + game_id, color='green')
            print_log_color('[INFO] GGS Got board from GGS egaroucid_color : ' + me_color, color='green')
            print_log_color('[INFO] GGS Got board from GGS remaining_time : ' + str(me_remaining_time), color='green')
            print_log_color('[INFO] GGS Got board from GGS board : ' + board, color='green')
            egaroucid_setboard(board)
            logfile_transcript = 'log/' + d_today.replace('-', '') + '_' + t_now.split('.')[0].replace(':', '') + '_transcript_' + game_id + '.txt'
            with open(logfile_transcript, 'a') as f:
                if last_move == '':
                    f.write('game_id ' + game_id + '\n')
                    f.write('egaroucid_color ' + me_color + '\n')
                    f.write('egaroucid_remaining_time ' + str(me_remaining_time) + '\n')
                    f.write('initial_board' + board + '\n')
                else:
                    f.write(last_move + '\n')
            if me_color == color_to_move:
                me_remaining_time_proc = max(1, me_remaining_time - 10)
                egaroucid_settime(me_color, me_remaining_time_proc)
                print_log_color('[INFO] Egaroucid thinking... game_id : ' + game_id, color='green')
                coord, value = egaroucid_get_move_score()
                if coord == 'ps':
                    coord = 'PA' # pass for GGS
                print_log_color('[INFO] Egaroucid moved : ' + coord + ' score ' + value, color='green')
                ggs_os_play_move(game_id, coord, value)
                if len(ponder_boards):
                    latest_board, latest_move = ponder_boards.pop()
                    print_log_color('[INFO] Egaroucid set board for ponder : ' + latest_board + ' ' + latest_move, color='green')
                    egaroucid_setboard(latest_board)
                    if latest_move != '':
                        egaroucid_play_move(latest_move)
                    ponder_boards.append([board, coord])
            else:
                ponder_boards.append([board, ''])
tn.close()