import telnetlib
import subprocess

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

def get_board():
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

def play_move(coord, value):
    cmd = 't /os play ' + coord + '/' + value
    tn.write((cmd + '\n').encode('utf-8'))

# login
tn.read_until(b": Enter login (yours, or one you'd like to use).")
tn.write((ggs_id + '\n').encode('utf-8'))
tn.read_until(b": Enter your password.")
tn.write((ggs_pw + '\n').encode('utf-8'))
wait_ready()

# alias
tn.write(b"ms /os\n")
wait_ready()

print('[INFO]', 'set up')

# start game
tn.write(b"ts ask 8w 05:00/00:00/02:00 nyanyan\n")
wait_ready()
wait_ready()

egaroucid_cmd = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -showvalue -ponder -time 295'
egaroucid = subprocess.Popen(egaroucid_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
while True:
    board = get_board()
    print('[INFO]', 'got board', board)
    egaroucid.stdin.write(('setboard ' + board + '\n').encode('utf-8'))
    egaroucid.stdin.flush()
    egaroucid.stdin.write(('go\n').encode('utf-8'))
    egaroucid.stdin.flush()
    line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '')
    coord = line.split()[0]
    value = line.split()[1]
    print('[INFO]', 'got move from Egaroucid', coord, value)
    play_move(coord, value)

tn.close()