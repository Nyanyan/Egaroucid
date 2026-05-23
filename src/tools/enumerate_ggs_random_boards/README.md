# Enumerate Random Boards

GGS Othello のランダム開始局面生成が作り得る盤面を列挙するツール。

出力は 1 行 1 局面で、64 マスの `X/O/-` と手番文字を空白区切りで書く。盤面は点対称・線対称な形を同一視して代表形だけを出力する。

## random_setup.cpp

GGS の `Board::random_setup(int ra)` に対応する。

アルゴリズムの概要:

- 角と角隣接マスを避ける。8x8 では各隅の 2x2、合計 16 マスを使わない。
- 残りのマスを中心からのチェビシェフ距離でグループ化する。
- 内側のグループから順に使い、最後の途中グループだけランダムに一部を選ぶ。
- 白石数はおおむね半分だが、GGS は `floor(white / 3)` までの偏りを許す。奇数石数では、半分切り上げのケースも 50% で発生する。
- 手番は通常のオセロの偶奇に合わせ、偶数石なら黒番、奇数石なら白番。

使い方:

```sh
g++ -std=c++20 -O2 -march=native random_setup.cpp -o random_setup.out
./random_setup.out <n_discs>
```

出力先は実行ファイルと同じディレクトリにある `output` フォルダ内:

```txt
output/<n_discs>_random_setup/0000000.txt
```

## random_setup_2.cpp

GGS の `Board::random_setup_2(int ra)` に対応する。GGS コメントでは Stephane Nicolet の alternate algorithm。

アルゴリズムの概要:

- まず `random_setup(5)` で 5 石の開始局面を作る。
- そこから `ra - 5` 手、合法手をランダムに進める。
- この合法手選択でも角と角隣接マスは避ける。
- 途中で合法手がなくなった試行は失敗扱いで捨てる。パスはしない。
- 最後に、黒石数または白石数が `max(ra / 4, 3)` 以下なら片寄りすぎとして捨てる。

このツールは `random_setup_2` が成功した場合に作り得る局面を列挙する。GGS 本体の `random_setup_2` は最大 5 回試して全て失敗した場合に `random_setup(ra)` へフォールバックするため、そのフォールバック分は `random_setup.cpp` 側でカバーする。

使い方:

```sh
g++ -std=c++20 -O2 -march=native random_setup_2.cpp -o random_setup_2.out
./random_setup_2.out <n_discs>
```

出力先は実行ファイルと同じディレクトリにある `output` フォルダ内:

```txt
output/<n_discs>_random_setup_2/0000000.txt
```

## GGS での混合割合

GGS の Othello では `NICOLET_RANDOM` が有効になっている。

- `ra <= 3`: 通常初期局面。
- `4 <= ra < 10`: `random_setup(ra)` を 100% 使用。
- `10 <= ra`: 50% で `random_setup(ra)`、50% で `random_setup_2(ra)` を試す。
- `random_setup_2(ra)` 側は最大 5 回試行し、全て失敗したら `random_setup(ra)` にフォールバックする。

そのため `ra >= 10` の最終的な比率は、厳密には次のようになる。

```txt
random_setup   : 50% + 50% * P(random_setup_2 が 5 回連続で失敗)
random_setup_2 : 50% * P(random_setup_2 が 5 回以内に成功)
```

8x8 の GGS 合法 rand type は `r4` から `r48`。
