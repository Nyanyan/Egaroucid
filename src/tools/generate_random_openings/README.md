# Generate Random Openings

ランダム打ちでオセロの序盤棋譜を生成するツールです。

`othello_py2.py` の `Othello` クラスを使い、各手番で合法手の中からランダムに 1 手を選びます。

## Usage

```sh
python generate_random_openings.py <n_moves> <n_games>
```

例:

```sh
python generate_random_openings.py 12 1000
```

`12` 手のランダム序盤棋譜を `1000` 局分、標準出力に出力します。

ファイルに保存する場合:

```sh
python generate_random_openings.py 12 1000 > random12.txt
```

## Output

1 行に 1 棋譜を出力します。

棋譜は `f5d6c3...` のように、小文字の列 `a`-`h` と行番号 `1`-`8` を 2 文字ずつ連結した形式です。

出力例:

```txt
f5f4f3g4
c4c5f6d3
c4c3e6f4
```

## Notes

- `n_moves` は `0` 以上 `60` 以下です。
- パスは棋譜には出力せず、内部で手番だけ進めます。
- 途中で終局して指定手数に届かない場合、その棋譜は破棄して作り直します。
- 乱数シードは指定していないため、実行ごとに出力は変わります。
