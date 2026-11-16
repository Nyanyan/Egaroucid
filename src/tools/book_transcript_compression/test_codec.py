"""Binary codec and optional C++ reader regression checks."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import compress_book
from codec import (BOOK_HEADER, INITIAL, PolicyGraph, bits, canonical, decode_book,
                   encode_tree, legal, pack_bits, pack_book, play, transform, unpack_bits)

ROOT = Path(__file__).resolve().parent
READER = Path(os.environ.get('BOOK_READER_EXE', str(ROOT/'out'/'read_book.exe')))


def random_transcript(seed):
    rng, board, moves = random.Random(seed), INITIAL, []
    while True:
        if not legal(*board):
            board = board[::-1]
            if not legal(*board):
                break
        move = rng.choice(list(bits(legal(*board))))
        cell = 63-(move.bit_length()-1)
        moves.append(chr(ord('a')+cell % 8)+str(1+cell//8))
        board = play(*board, move)
    return ''.join(moves)


def reference_bytes(policies):
    return b''.join(struct.pack('<QQQ', p, o, policy)
                    for (p, o), policy in sorted(policies.items()))


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = PolicyGraph()
        cls.transcripts = [random_transcript(seed) for seed in range(4)]
        for transcript in cls.transcripts:
            for side in (0, 1):
                cls.graph.import_transcript(transcript, side)

    def test_byte_packing_and_padding(self):
        values = bytes(range(256))
        self.assertEqual(pack_bits(unpack_bits(values)), values)
        self.assertEqual(pack_bits([]), b'')
        for length in range(1, 24):
            data = [i % 2 for i in range(length)]
            packed = pack_bits(data)
            self.assertIsInstance(packed, bytes)
            self.assertEqual(len(packed), (length+7)//8)
            self.assertEqual(unpack_bits(packed), data+[0]*((-length) % 8))

    def candidates(self):
        for dfs in (False, True):
            nodes, retained = self.graph.tree(64, dfs)
            for arithmetic, parent_context in ((False, False), (True, False), (True, True)):
                data = encode_tree(nodes, arithmetic, parent_context)
                yield pack_book(data, len(nodes), len(retained), arithmetic, dfs, parent_context), retained

    def test_all_codec_modes_roundtrip(self):
        for packed, retained in self.candidates():
            with self.subTest(flags=packed[5]):
                self.assertGreater(len(retained), 200)
                self.assertEqual(decode_book(packed), retained)

    def test_invalid_binary(self):
        graph = PolicyGraph()
        graph.import_transcript('d3', 0)
        nodes, retained = graph.tree()
        data = encode_tree(nodes, False)
        packed = pack_book(data, len(nodes), len(retained), False)
        invalid = [packed[:10], b'FAIL'+packed[4:], packed[:4]+b'\x02'+packed[5:],
                   packed[:5]+b'\x80'+packed[6:], packed[:-1], packed+b'\0',
                   packed[:12]+struct.pack('<I', 999)+packed[16:]]
        if len(data) % 8:
            invalid.append(packed[:-1]+bytes([packed[-1] | 1]))
        for value in invalid:
            with self.subTest(binary=value):
                with self.assertRaises(ValueError):
                    decode_book(value)

    def run_main(self, root, *options):
        with patch.object(compress_book, 'ROOT', root), \
             patch.object(sys, 'argv', ['compress_book.py', '--transcripts', str(ROOT/'sample.jsonl'), *options]), \
             contextlib.redirect_stdout(io.StringIO()):
            compress_book.main()

    def test_default_output_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_main(root)
            binary = (root/'out'/'book_data.bin').read_bytes()
            report = json.loads((root/'out'/'book_compression.json').read_text())
            self.assertEqual(len(decode_book(binary)), 3)
            self.assertEqual(report['selected']['file_bytes'], len(binary))
            self.assertEqual(report['selected']['packed_bytes'], len(binary)-BOOK_HEADER.size)
            self.assertEqual(report['binary_sha256'], hashlib.sha256(binary).hexdigest())

    def test_byte_budget_prunes_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_main(root, '--max-bytes', '22')
            binary = (root/'out'/'book_data.bin').read_bytes()
            self.assertLessEqual(len(binary), 22)
            self.assertLess(len(decode_book(binary)), 3)
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                self.run_main(root, '--max-bytes', '20')

    @unittest.skipUnless(READER.is_file(), 'build read_book.exe or set BOOK_READER_EXE')
    def test_cpp_reader_all_modes_and_lookup(self):
        query = self.transcripts[0][:40]
        board = INITIAL
        for i in range(0, len(query), 2):
            if not legal(*board):
                board = board[::-1]
            cell = (int(query[i+1])-1)*8+ord(query[i])-ord('a')
            board = play(*board, 1 << (63-cell))
        if not legal(*board) and legal(*board[::-1]):
            board = board[::-1]
        key, symmetries = canonical(*board)
        with tempfile.TemporaryDirectory() as directory:
            binary, reference = Path(directory)/'book.bin', Path(directory)/'decoded.bin'
            for packed, retained in self.candidates():
                with self.subTest(flags=packed[5]):
                    binary.write_bytes(packed)
                    result = subprocess.run([str(READER), str(binary), query, '--export-reference', str(reference)],
                                            capture_output=True, text=True, check=True)
                    self.assertEqual(reference.read_bytes(), reference_bytes(retained))
                    coords = result.stdout.split('moves: ', 1)[1].strip().split()
                    mask = sum(1 << (63-((int(c[1])-1)*8+ord(c[0])-ord('a'))) for c in coords)
                    self.assertEqual(transform(mask, symmetries[0]), retained[key])
                    self.assertEqual(mask & ~legal(*board), 0)


if __name__ == '__main__':
    unittest.main()
