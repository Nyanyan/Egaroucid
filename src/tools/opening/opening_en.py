from othello_py import *

def translate_transcript(transcript):
    if transcript[:2] == 'f5':
        return transcript
    res = ''
    for i in range(0, len(transcript), 2):
        x = ord(transcript[i]) - ord('a')
        y = int(transcript[i + 1]) - 1
        if transcript[:2] == 'c4':
            res += chr(ord('a') + 7 - x)
            res += str(8 - y)
        elif transcript[:2] == 'd3':
            res += chr(ord('a') + 7 - y)
            res += str(8 - x)
        elif transcript[:2] == 'e6':
            res += chr(ord('a') + y)
            res += str(x + 1)
    return res

data = []
names = set([])
transcripts = set([])
with open('data/openings_english.txt', 'r', encoding='utf-8') as f:
    for datum in f.read().splitlines():
        transcript = datum[:41].replace(' ', '').lower()
        name = datum[41:].split(',')[0].replace('**', '').replace('(t3)', '').replace('"', '')
        if len(transcript):
            while name[-1] == ' ':
                name = name[:-1]
            if not name in names:
                names.add(name)
                transcript_f5 = translate_transcript(transcript)
                print(transcript_f5, name)
                data.append([len(transcript_f5) + 10000, name, transcript_f5])
                transcripts.add(transcript_f5)

data.sort()

for i in range(len(data)):
    transcript = data[i][2]
    n_spaces = 0
    for j in reversed(range(0, len(transcript) - 1, 2)):
        t_sub = transcript[:j]
        #print(t_sub)
        if t_sub in transcripts:
            for k in range(i):
                if data[k][2] == t_sub:
                    n_spaces = data[k][0] + 1
            break
    data[i][0] = n_spaces
    print(n_spaces, transcript)


joseki = {}
joseki_many = {}

for n_spaces, name, record in data:
    o = othello()
    o.check_legal()
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('A')
        if x >= hw:
            x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        o.move(y, x)
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        s = ''
        for i in range(hw):
            for j in range(hw):
                if o.grid[i][j] == 0:
                    s += '0'
                elif o.grid[i][j] == 1:
                    s += '1'
                else:
                    s += '.'
        flag = not (s in joseki_many)
        if (not flag):
            if joseki_many[s][0] == n_spaces:
                joseki_many[s].append(name)
        else:
            joseki_many[s] = [n_spaces, name]
    #if not (s in joseki.keys()):
    #    joseki[s] = name
    joseki[s] = name

print(len(joseki))
print(len(joseki_many))
with open('output/openings.txt', 'w', encoding='utf-8') as f:
    for board in joseki.keys():
        f.write(board + ' ' + joseki[board] + '\n')
with open('output/openings_fork.txt', 'w', encoding='utf-8') as f:
    for board in joseki_many.keys():
        f.write(board + ' ' + '|'.join(joseki_many[board][1:]) + '\n')