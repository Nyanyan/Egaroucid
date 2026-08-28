# Edax / Neural Reversi 外部調査メモ

調査日: 2026-08-28。以下では事実の根拠を混ぜないため、記述を次の4区分に分ける。

- **公開ソースで確定**: 指定commitの実装または公式READMEから直接確認できる。
- **作者・論文資料に記載**: 作者記事またはEdaxを直接調査した論文に書かれているが、現行実装と同一とは限らない。
- **実測可能／実測で確認**: 配布重みや実行ファイルを測れば確認できる性質。学習方法の証明にはならない。
- **不明**: 調査範囲の公開資料からは確定できない。

## 調査対象を固定したcommit

| 対象 | 調査時HEAD | commit日時 | 一次資料 |
|---|---|---:|---|
| Edax 公式リポジトリ | `14f048c05ddfa385b6bf954a9c2905bbe677e9d3` | 2025-03-10 | [abulmo/edax-reversi](https://github.com/abulmo/edax-reversi/tree/14f048c05ddfa385b6bf954a9c2905bbe677e9d3) |
| Neural Reversi | `6556d90e8f2cf705a92386205b70a361822586d2` | 2026-08-22 | [natsutteatsuiyone/neural-reversi](https://github.com/natsutteatsuiyone/neural-reversi/tree/6556d90e8f2cf705a92386205b70a361822586d2) |
| Neural Reversi Training | `b1170a6cc7f2baaef270241f94f2f817dde32c15` | 2026-08-20 | [natsutteatsuiyone/neural-reversi-training](https://github.com/natsutteatsuiyone/neural-reversi-training/tree/b1170a6cc7f2baaef270241f94f2f817dde32c15) |

## Edax

### 公開ソースで確定したこと

- 公式READMEは実行ファイルとは別に `data/eval.dat` をリリースから取得する構成を説明している。Gitリポジトリには評価重みの生成器や学習データは同梱されていない。[README](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/README.md)
- 評価器は12種類の3値盤面パターンと定数項を使う。展開後は46個の盤面上パターン出現と定数項を加算する。対称圧縮前はパターン226,314個＋定数1個、1局面段階あたり226,315パラメータである。[パターン座標](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c#L36-L87)、[サイズ定義](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c#L440-L481)
- `eval.dat` は各局面段階につき対称圧縮後114,364個の符号付き16-bit重みを保持し、読み込み時に対称形と手番色を展開する。公開ソースのローダーは `ply=0..60` の61ブロックを読む。[eval.dat loader](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c#L572-L738)
- 静的評価は重み和を符号付きで128単位に丸めて石差尺度へ変換し、非終局値を `-63..+63` に制限する。[midgame.c](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c#L29-L43)
- 評価値の加算に明示的な着手可能数、自石数、相手石数の項はない。着手可能数は探索の手順並べ替え等には使われるが、それを静的評価関数の特徴と数えてはいけない。[eval_accumulate](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c#L1030-L1134)
- このcommitのNBoard `learn` は `play_store()` を呼び、`play_store()` は明示的に対局をopening bookへ保存する処理である。したがって、この `learn` は `eval.dat` の評価重み学習ではない。公開ツリーで確認できる評価関数処理は主に読み込み・展開・推論である。[NBoard `learn`](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/nboard.c#L173-L176)、[`play_store()`](https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/play.c#L941-L972)

### 作者・論文資料に記載されたこと

- Edaxを直接扱う査読論文は、Edaxを「12パターン＋定数、盤面4～63石に対応する60局面段階、合計13,578,900パラメータ、16-bit、対称圧縮後約13.3 MiB」と整理している。ソースの61読込ブロックとの違いは、使用される60非終局段階とファイル内ブロック数を区別して扱う必要がある。[Yamana and Hoshino, IEEE Transactions on Games, DOI 10.1109/TG.2025.3624825](https://doi.org/10.1109/TG.2025.3624825)、[著者版全文](https://tsukuba.repo.nii.ac.jp/records/2023333)
- Richard Delorme本人による公開README/Doxygenはpattern-based evaluationであることとファイル形式・推論実装を示すが、今回の調査範囲では評価重みの教師ラベル、データ生成、損失関数、最適化法を記した作者資料は見つからなかった。
- Logistello/Buroの一般的なパターン評価学習法は、Edaxが同じ学習データ、教師、目的関数、optimizerを使った証拠ではない。本調査ではEdax固有の事実として採用しない。

### 配布 `eval.dat` から実測できること

- **実測で確認**: 今回比較に使った `bin/versions/edax_4_5_5/bin/data/eval.dat` は13,952,436 bytes、SHA-256は `f8b2299612d9fa4414157e70e932636e33111c2602d0c2fc382a7d90ef21b792`。ヘッダは評価重みversion `3.2.5`、時刻値はUTC `2002-11-28T17:59:03.701034`。サイズは28-byte header＋61×114,364×2 bytesと一致する。
- **実測で確認**: 比較実行ファイル `wEdax-x86-64-v3.exe` は566,272 bytes、SHA-256は `52d57d89a956817672730e575a4cb67a1e0e38db4de8754a7b4cf15c4a473410`、bannerはEdax 4.5.5だった。
- **実測可能**: 重みの局面段階別分布、飽和、隣接段階相関、および同一局面集合上のEdax評価のbias・傾き・兄弟順位は測定できる。これらは「学習済みモデルの性質」であり、元ラベル分布を一意に復元するものではない。

### 不明

- Edax評価重みの元棋譜・局面分布、ランダム開始法、教師が終局石差か探索値か、その混合方法。
- train/validation分割、重複処理、サンプル重み、損失、正則化、optimizer、収束判定、量子化前の浮動小数重み。
- したがって「Edaxの教師ラベルは縮んでいない」「Edaxは特定の学習法だから探索後に改善する」とは公開情報だけでは言えない。同一Egaroucid学習局面上での実測比較が必要である。

## Neural Reversi: 現行コード

### データ生成

- **公開ソースで確定**: 通常self-playのopening長は `min(U[10,50), U[10,50))`、すなわち10～49手で小さい値に偏る。READMEの「10～30手」は現行実装と不一致なので、実装を採用する。[selfplay.rs constants/generation](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/selfplay.rs#L27-L30)、[sampling](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/selfplay.rs#L177-L183)、[README](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/README.md)
- opening中の局面も捨てず、各局面を探索して `record.score` と探索最善手を保存する。ただし実際に指す手はランダム手または指定opening手で、`is_random=true` になる。指定openingも同じ処理を通るため、`is_random` は厳密には「探索最善手でなくopening列の手を指した」を表す。[selfplay.rs play_game](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/selfplay.rs#L335-L405)
- opening終了後は探索最善手で自己対局し、`record.score` はその局面の探索値、`is_random=false`。終局後、各recordの手番視点に合わせた最終石差を `record.game_score` に格納する。[selfplay.rs final score](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/selfplay.rs#L407-L418)、[27-byte record](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/record.rs#L20-L53)

### 教師選択とshuffle filter

- **公開ソースで確定**: 現行loaderの教師は `ply<=1` なら0、そうでなく `is_random=true` なら `record.score`（探索値）、`is_random=false` なら `record.game_score`（最終石差）。現在は両者の加重混合式ではない。[bin_dataset.cpp](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/dataset/csrc/bin_dataset.cpp#L105-L138)
- `--max-score-diff T` は `|record.score-record.game_score|>T` を除く。`--drop-random` はopening列中のrecordを除く。`--keep-above-ply P` は `ply>=P` についてこの2フィルタを迂回するが、`--min-ply` は常に独立して適用される。[shuffle.rs](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/shuffle.rs#L355-L392)、[CLI定義](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/main.rs#L95-L117)
- 公式READMEのfilter例は `--min-ply 8 --max-score-diff 12 --drop-random --keep-above-ply 50` だが、これは使用例であり、配布モデルの実際の起動引数だという証拠はない。[datagen README filter example](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/README.md#L110-L152)

### 現行large modelと学習

- **公開ソースで確定**: 入力は32個のパターン出現（20×`3^8`、8×`3^9`、4×`3^7`）と明示的な現在手番の着手可能数。[model_common.py](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/models/model_common.py#L22-L61)、[model_lg.py](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/models/model_lg.py#L20-L145)
- `base_input` は全局面段階共有、`pa_input` は10手刻みの6 bucket、後段のL1/L2/outputは1手刻みの60 bucketである。着手可能数は `min(mobility*7/255,1)` としてL1へ明示入力される。
- 教師石差を64で正規化し、損失はMSE。既定はbatch 65,536、400 epoch、learning rate `5e-4`、weight decay `1e-2`、AdamW＋Lookahead、CosineAnnealing、gradient norm clip 1.0である。[model_common.py loss/optimizer](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/models/model_common.py#L114-L168)、[train.py CLI](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/scripts/train.py#L49-L153)
- 現行forwardには重みを毎回整数へ丸める処理は見当たらない。活性値のclamp、量子化scale補正、隠れ層重みの範囲clippingを学習中に行い、整数への丸めはserialize時に行う。[weight clipping](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/models/model_common.py#L257-L262)、[serialize_lg.py](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/models/serialization/serialize_lg.py)

### 現行学習について不明なこと

- training repositoryにある起動例はtrain/validation directoryを指定する最小例だけで、配布済み重みを作った実コマンド、設定ファイル、shuffle filter値はcommitされていない。[training README](https://github.com/natsutteatsuiyone/neural-reversi-training/blob/b1170a6cc7f2baaef270241f94f2f817dde32c15/README.md#L32-L73)
- 調査した両repositoryには実学習dataset、checkpoint、データmanifestがない。したがって使用ゲーム数、各ply・正負・石差帯の割合、重複率、ラベル対完全読みのbias/傾き、train/validation漏洩は不明であり、コードのdefaultやREADME例から推測しない。
- self-play探索深度のCLI defaultはmidgame 12、endgame 21、生成ゲーム数default 1億だが、配布重み生成時に実際に使われた値は不明である。[datagen CLI](https://github.com/natsutteatsuiyone/neural-reversi/blob/6556d90e8f2cf705a92386205b70a361822586d2/crates/datagen/src/main.rs#L22-L67)

## Neural Reversi: 過去Zenn記事との分離

**作者・過去資料に記載**: 2025-03-23公開（2025-05-07更新）の記事は当時の方式として、8個のパターン＋着手可能数、最大30手のランダム進行、非ランダム局面に対する「手数×最終石差＋残り×評価値」の混合教師、Edax 12手読み100万棋譜→自モデル150万棋譜、順伝播中の丸め、AdamW＋CosineAnnealingを説明している。[Zenn: 浅いニューラルネットワークで作る高精度なオセロの評価関数](https://zenn.dev/natsuatsui/articles/c1d9aa50acb347)

これは現行HEADの説明として使えない。現行コードは32パターン出現、10～49手から偏り付きで選ぶopening長、`is_random` による探索値／最終石差の切替、AdamW＋Lookahead、serialize時丸めであり、過去記事の混合教師と順伝播丸めは確認できない。一方、配布中モデルが現行HEADのどの時点・引数で学習されたかも公開資料だけでは不明である。
