"""Compress opening-book policies into a binary file from transcripts or legacy packed books.

Compare breadth/depth-first spanning trees and raw/adaptive arithmetic bitstreams.
Every candidate is decoded and compared with the complete retained policy map.
"""
import argparse
import hashlib
import json
import struct
import sqlite3
import time
from pathlib import Path
from codec import PolicyGraph, encode_tree, pack_book, decode_book

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--legacy-dir', type=Path, help='directory containing legacy black.txt and white.txt packed C++ strings')
    ap.add_argument('--transcripts', type=Path, nargs='*', default=[], help='SQLite database (games.payload JSON) or JSONL with transcript and side')
    ap.add_argument('--max-discs', type=int, default=60)
    ap.add_argument('--max-bytes', type=int, default=99999, help='maximum binary file size in bytes, including the header')
    ap.add_argument('--output', type=Path, default=ROOT/'out'/'book_data.bin')
    ap.add_argument('--report', type=Path, default=ROOT/'out'/'book_compression.json')
    ap.add_argument('--export-reference', type=Path, help='sorted <P,O,policy> uint64 triples for the C++ decoder regression')
    args = ap.parse_args()
    if not 4 <= args.max_discs <= 64 or args.max_bytes <= 0:
        ap.error('max-discs must be 4..64 and max-bytes must be positive')
    if args.legacy_dir is None and not args.transcripts:
        ap.error('provide --transcripts or --legacy-dir')
    graph, inputs, start = PolicyGraph(), [], time.perf_counter()
    if args.legacy_dir is not None:
        for side, name in enumerate(('black.txt', 'white.txt')):
            path = args.legacy_dir/name
            metadata = graph.import_legacy(path.read_text(encoding='utf-8'), side)
            metadata.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), side=side)
            inputs.append(metadata)
            print('imported', name, 'policy positions', len(graph.policy), flush=True)
    for path in args.transcripts:
        if path.suffix in ('.sqlite3', '.sqlite', '.db'):
            with sqlite3.connect('file:'+path.resolve().as_posix()+'?mode=ro',uri=True) as database:
                lines = [json.loads(row[0]) for row in database.execute('SELECT payload FROM games ORDER BY id')]
        else:
            lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in lines:
            side = int(row['side'])
            if side not in (0, 1):
                raise ValueError('side must be black=0 or white=1')
            graph.import_transcript(row['transcript'], side, int(row.get('start_ply',0)), row.get('policy_plies'))
        inputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'transcripts': len(lines),
                       'transcript_snapshot_sha256': hashlib.sha256(json.dumps(lines,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()})
    if not graph.policy:
        raise ValueError('no policy positions imported')
    candidates = []
    for max_discs in range(args.max_discs, 3, -1):
        winners = []
        for dfs in (False, True):
            nodes, retained = graph.tree(max_discs, dfs)
            if not retained:
                continue
            for arithmetic,parent_context in ((False,False),(True,False),(True,True)):
                data = encode_tree(nodes, arithmetic,parent_context)
                packed = pack_book(data, len(nodes), len(retained), arithmetic, dfs, parent_context)
                decoded = decode_book(packed)
                assert decoded == retained, 'book codec changed an imported policy'
                record = {'max_discs': max_discs, 'dfs': dfs, 'arithmetic': arithmetic,'parent_context':parent_context,
                          'nodes': len(nodes), 'positions': len(retained), 'bits': len(data),
                          'packed_bytes': (len(data)+7)//8, 'file_bytes': len(packed),
                          'complete_policy_roundtrip': True}
                candidates.append(record)
                print(json.dumps(record), flush=True)
                if len(packed) <= args.max_bytes:
                    winners.append((len(packed), len(data), packed, record, retained))
        if winners:
            _, _, packed, selected, reference = min(winners, key=lambda item: item[:2])
            break
    else:
        raise ValueError('book does not fit the byte budget at any disc limit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(packed)
    assert decode_book(args.output.read_bytes()) == reference, 'saved book changed an imported policy'
    if args.export_reference:
        args.export_reference.parent.mkdir(parents=True, exist_ok=True)
        with args.export_reference.open('wb') as stream:
            for (p, o), policy in sorted(reference.items()):
                stream.write(struct.pack('<QQQ', p, o, policy))
    report = {'inputs': inputs, 'source_policy_positions': len(graph.policy),
              'imported_moves': graph.imported_moves, 'max_bytes': args.max_bytes,
              'selected': selected, 'candidates': candidates,
              'binary_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'elapsed_s': time.perf_counter()-start,
              'method': '8 symmetries; full-board transpositions; pruned spanning tree; joint policy/tree edges; combinatorial legal-move ranks; adaptive arithmetic coding; parent-policy context option',
              'lossless_for_retained_positions': True,
              'scope_note': 'Searches the implemented codec/traversal choices. This is not a proof of globally shortest encoding.'}
    if args.export_reference:
        report.update(reference=str(args.export_reference), reference_sha256=hashlib.sha256(args.export_reference.read_bytes()).hexdigest())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', args.output, 'selected', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
