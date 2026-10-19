# 読み切り (endgame search with MPC) test on GGS positions
# Positions: GGS games where egrcd decided a move by an endgame search with MPC (problem/yomikiri_ggs.json,
# made by reports/20261019_3_yomikiri_build.py). Every position has its exact value and the exact values
# of the moves known so far (the GGS moves, the exact best move and the screen move).
#
# Every position is solved to the end with the given MPC probability (-dpr 1 60 60 <prob>), like
# mpcendtest.py. A move whose exact value is not known is reported as UNKNOWN (solve it with
# "python reports/20261019_3_yomikiri_build.py extend" to add it).
# prob: 74, 88, 93, 98, 99, 99.9, 100

import json
import os
import subprocess
import sys
import time

PROBS = ['74', '88', '93', '98', '99', '99.9', '100']


def parse_args(args, names):
    # positional (in the order of names) or name=value
    n_positional = 0
    for arg in sys.argv[1:]:
        name, sep, value = arg.partition('=')
        if sep and name.isidentifier():
            if not (name in names):
                raise ValueError('unknown option ' + name)
        else:
            if n_positional >= len(names):
                raise ValueError('too many arguments')
            name = names[n_positional]
            value = arg
            n_positional += 1
        args[name] = type(args[name])(value)


args = {
    'prob': '93',
    'n_threads': 1,
    'procs': 1,
    'hash_level': 23,
    'exe': 'Egaroucid_for_Console.exe',
    'problem_file': 'problem/yomikiri_ggs.json',
    'subset': 'all',
    'out': '',
}
try:
    parse_args(args, ['prob', 'n_threads', 'procs', 'hash_level', 'exe', 'problem_file', 'subset', 'out'])
    if not (args['prob'] in PROBS):
        raise ValueError('prob must be one of ' + ', '.join(PROBS))
    if args['subset'] not in ('all', 'mistakes', 'others'):
        raise ValueError('subset must be all, mistakes (positions where the GGS move lost discs) or others')
except Exception as e:
    print(e)
    print('usage: python yomikiritest.py [prob=93] [n_threads=1] [procs=1] [hash_level=23] [exe=Egaroucid_for_Console.exe] [problem_file=problem/yomikiri_ggs.json] [subset=all|mistakes|others] [out=]')
    print('procs > 1 splits the positions over that many engine processes (n_threads each); out= saves the per-position results as JSON')
    sys.exit(2)

script_dir = os.path.dirname(os.path.abspath(__file__))
exe = args['exe'] if os.path.isabs(args['exe']) else os.path.join(script_dir, args['exe'])
problem_file = args['problem_file'] if os.path.isabs(args['problem_file']) else os.path.join(script_dir, args['problem_file'])
with open(problem_file, 'r', encoding='utf-8') as f:
    problems = json.load(f)
if args['subset'] == 'mistakes':
    problems = [p for p in problems if p['ggs_mistake']]
elif args['subset'] == 'others':
    problems = [p for p in problems if not p['ggs_mistake']]

version = subprocess.run([exe, '-v'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, check=True).stdout.decode().strip()
print(version)
cmd_base = [exe, '-dpr', '1', '60', '60', args['prob'], '-nobook', '-quiet', '-thread', str(args['n_threads']), '-hash', str(args['hash_level'])]
print(subprocess.list2cmdline(cmd_base).replace(script_dir, 'script_dir') + ' -solve <shard>', '(' + str(args['procs']) + ' processes)', flush=True)

tmp_dir = os.path.join(script_dir, 'tmp_yomikiritest_' + str(os.getpid()))
os.makedirs(tmp_dir, exist_ok=True)
n_procs = max(1, min(args['procs'], len(problems)))
shards = [list(range(k, len(problems), n_procs)) for k in range(n_procs)]
procs = []
t0 = time.time()
try:
    for k, ids in enumerate(shards):
        pf = os.path.join(tmp_dir, 'shard_' + str(k) + '.txt')
        with open(pf, 'w', encoding='utf-8') as f:
            for i in ids:
                f.write(problems[i]['board'] + '\n')
        procs.append(subprocess.Popen(cmd_base + ['-solve', pf], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    outputs = [p.communicate()[0].decode() for p in procs]
finally:
    for p in procs:
        if p.poll() is None:
            p.kill()
wall = time.time() - t0

rows = [None] * len(problems)
for ids, text in zip(shards, outputs):
    lines = [l for l in text.splitlines() if l.startswith('|') and 'Level' not in l]
    if len(lines) != len(ids):
        raise RuntimeError('engine returned ' + str(len(lines)) + ' results for ' + str(len(ids)) + ' positions')
    for i, line in zip(ids, lines):
        c = [x.strip() for x in line.split('|') if x.strip()]
        if c[0] != 'custom':
            raise RuntimeError('engine did not use the requested probability: ' + line)
        h, m, s = c[4].split(':')
        rows[i] = {'depth': c[1], 'move': c[2].lower(), 'value': int(c[3]), 'time': int(h) * 3600 + int(m) * 60 + float(s), 'nodes': int(c[5])}

n = len(problems)
wrong = unknown = total_loss = big_loss = 0
value_abs = 0
total_time = total_nodes = 0
results = []
for i, (p, r) in enumerate(zip(problems, rows)):
    q = p['move_values'].get(r['move'])
    tag = ''
    if q is None:
        unknown += 1
        tag = ' UNKNOWN MOVE'
        loss = None
    else:
        loss = p['best_value'] - q
        if loss > 0:
            wrong += 1
            total_loss += loss
            big_loss += loss >= 4
            tag = ' WRONG POLICY loss ' + str(loss)
    value_abs += abs(r['value'] - p['best_value'])
    total_time += r['time']
    total_nodes += r['nodes']
    results.append({**r, 'board': p['board'], 'best_value': p['best_value'], 'loss': loss, 'ggs_mistake': p['ggs_mistake']})
    print('#{:03d} e{} {} move {} value {:+d} best {:+d} time {:.3f} nodes {}{}{}'.format(
        i, p['empties'], 'M' if p['ggs_mistake'] else '-', r['move'], r['value'], p['best_value'], r['time'], r['nodes'],
        tag, ' WRONG SCORE' if r['value'] != p['best_value'] else ''), flush=True)
print('positions {} wrong_moves {} unknown_moves {} total_loss {} loss_ge_4 {} value_mae {:.3f} engine_time {:.3f}s nodes {} wall {:.1f}s'.format(
    n, wrong, unknown, total_loss, big_loss, value_abs / max(1, n), total_time, total_nodes, wall), flush=True)
if args['out']:
    with open(args['out'], 'w', encoding='utf-8') as f:
        json.dump({'args': args, 'results': results}, f)
for k in range(n_procs):
    try:
        os.remove(os.path.join(tmp_dir, 'shard_' + str(k) + '.txt'))
    except OSError:
        pass
try:
    os.rmdir(tmp_dir)
except OSError:
    pass
