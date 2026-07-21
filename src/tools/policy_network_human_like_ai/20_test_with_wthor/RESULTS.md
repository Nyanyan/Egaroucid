# WTHORテストデータにおける人間着手一致率

関連issue: #613

更新日: 2026-07-21

## 実験の定義

WTHORのtest分割からseed付きで局面を無作為抽出し、次を同じ標本で測定する。

- Policy NetworkとEgaroucid level 21を、`alpha=0.0, 0.2, ..., 1.0`で幾何ブレンドした方策のtop-1・top-3一致率
- Egaroucidの奇数level 1から21までのtop-1・top-3一致率
- 各一致率のWilson法による95%信頼区間

一致判定では、手番側と相手側の石配置をそれぞれ不変に保つ、正方形盤面の回転・鏡映による8通りの変換を考慮する。人間の実着手からこれらの変換で移る合法手のいずれかが上位N手に入れば一致と数える。

Policy Networkが手`m`を選ぶ確率を`p(m)`、Egaroucidの評価値を確率へ変換したものを`q(m)`とする。Egaroucid評価値の尺度を調整する係数を`temperature`と呼ぶ。合成後の方策は従来どおり、次の重み付き幾何平均で定義する。

```text
r(m) = p(m)^alpha * q(m)^(1 - alpha)
```

最後に全合法手の`r(m)`の合計が1になるよう正規化する。ただし、top-1・top-3の測定に必要なのは確率の値そのものではなく着手の順位だけである。

小さな確率を直接累乗・乗算すると、計算機上で0へ丸められることがある。そこで実装では、上の式の自然対数を取った次の値を比較する。

```text
alpha * log(p(m)) + (1 - alpha) * log(q(m))
```

対数は単調増加関数なので、この値で並べても`r(m)`で並べても着手順位は同じになる。さらに、Policy Networkの出力やEgaroucid評価値を確率へ変換するときに加わる、全着手に共通の項は順位に影響しないため省略できる。その結果、実際の順位計算は次の式になる。

```text
alpha * Policy Networkの変換前出力
    + (1 - alpha) * (Egaroucidの評価値 / temperature)
```

つまり、合成方式を変更したのではない。従来の重み付き幾何平均とまったく同じ順位を、数値が0へ丸められない形で計算しているだけである。

## `hint 3`と`hint 64`の使い分け

ブレンド方策を作るには全合法手のEgaroucid評価値が必要である。上位3手だけを取得し、残りの合法手を確率0としてブレンドした旧結果は無効とする。

このため、旧100,000局面評価を入力にした`../50_rating_vs_human_match/`のブレンド系列も、全合法手による再評価が終わるまでは正式な結果として使わない。`../40_test_strength/`は従来から全合法手を取得していたため、この不具合の影響を受けない。

新実装では用途を分ける。

- ブレンド用level 21: `hint 64`を要求し、返された手の集合が全合法手と完全一致することを検証する。
- Egaroucid level 1から19の単体top-1・top-3: Consoleへ直接`hint 3`を要求する。上位3手より下の評価値はこの指標に不要である。
- Egaroucid level 21単体: ブレンド用`hint 64`の先頭3手を再利用し、同じ探索を重複実行しない。

したがって、`hint 3`が誤りなのは「ブレンド方策の生成に使う場合」であり、Egaroucid単体のtop-1・top-3を測る用途では正しい。

Consoleはlevel間で共有しない。v5で16 workersを使う場合、開始時はlevel 1から19の各level専用Consoleを1つずつ、level 21専用Consoleを6つ起動する。低levelのConsoleが完了したら、空いた枠にlevel 21専用Consoleを追加する。level 21の全actorが完了した後も他levelの局面が残っていれば、全ての空き枠にそのlevel専用Consoleを追加する。同levelの複数actorはlevel別のatomic cursorから次の局面を1件ずつ取得し、各局面を厳密に1回だけclaimする。置換表は同一Console内だけで保持し、異なるlevel間だけでなく、同levelの別actor間でもConsoleと置換表を共有しない。worker数、1 worker当たりの探索スレッド数、標本、分配方式を実験identityへ保存する。

## 書き直した実装

実行入口は`run_random_wthor_blend_experiment.py`である。実装は、科学計算を担当する`wthor_human_match_evaluation.py`、Console探索・キャッシュを担当する`wthor_hint_pipeline.py`、CLI・成果物を担当する`wthor_human_match_experiment.py`へ分けた。旧実装の二重の並列プール、Manager経由のhot path、全プロセス共通の0.1秒staggerは廃止した。

主な処理は次の通り。

- N局面を先に抽出し、その後で同一の`(盤面, 手番)`をまとめる。人間着手ごとの出現回数は保持し、評価分母は元のN局面のままにする。
- 同時実行数を制限した単一のspawn ProcessPoolを使う。各actorは終了までlevelを固定し、専用Consoleを1つ常駐させる。
- 16 workersのcold cacheでは、level 1から19の10 actorとlevel 21の6 actorを同時に開始する。level 21が未完了の間は低levelの完了で空いた枠へlevel 21 actorを追加する。level 21の全actor完了後は、残存levelの推定残作業に応じて全ての空き枠を再分配する。同時起動数は最大16である。
- 各levelは未処理局面列とatomic cursorを持つ。同levelのactorが増えても、cursorのclaimによって重複・取りこぼしなく各局面を1回だけ計算する。
- workerは数状態ごとに探索結果だけを親へ送り、SQLiteは親プロセスだけがtransactionで保存する。Consoleと置換表はactorごとに独立し、異なるactorと共有しない。
- 最終JSONの`level_timing`には各actorのID、worker PID、処理した局面index列、開始・終了時刻、処理時間を保存し、実際の動的割当を後から追跡できるようにする。
- timeout、異常終了、不完全なhint出力はConsoleを再起動して再試行する。
- 失敗やCtrl+Cでは未開始actorを取り消し、実行中workerへ協調キャンセルを通知する。actor完了前に得られた結果も随時キャッシュへ保存する。再開時にあるlevelのキャッシュが不完全なら、そのlevelの全局面を新しいatomic cursorへ載せ直して再計算し、中断前後の異なる置換表履歴を混在させない。
- Policy Networkは一意状態だけをbatch推論し、全alphaでlogitを共有する。
- 実行ファイル、重み、抽出位置、抽出レコード内容、並列条件などをhash付きidentityへ保存し、条件の異なる出力先の再利用を拒否する。
- 標本読込、Policy Network推論、Console探索、集計、保存の各段階を開始時に即時表示する。Console探索中は既定で30秒ごとに全体進捗、Console actorの累計起動数・完了数、level別の稼働Console数、CPU・メモリ、再試行回数を表示する。続けて値が得られた各モデルを1モデル1行で表示し、top-1・top-3とそれぞれの95% Wilson信頼区間、評価分母`n`、当該levelのhint進捗を示す。値がまだないモデルの行と注意書きは表示しない。完走後の標準出力でも、ブレンド方策とConsole単体の両方に同じ信頼区間を表示する。

既定値は、この32スレッドPCで実測の速かった16 workers × 2探索スレッドである。探索スレッド総数は32となる。

## 検証

- 関連unit test 54件がすべて成功した。
- 回帰テストでは、level 21の最後のactor完了直後に稼働中の低level actorを残したまま同level actorが追加されること、level 21が全件キャッシュ済みなら開始直後から低levelへ複数actorを割り当てること、8並列actorが100局面を重複・欠落なく1回ずつclaimすることを確認した。
- v5を実バイナリ、1局面、11 workers × 1 thread、cold cacheで実行し、11個のlevel固定actor、Windows spawnでのatomic cursor共有、全level・全alpha・CSV・JSON・SQLiteキャッシュの生成まで完走することを確認した。所要時間は約6秒だった。
- level 19の同じ12局面を1 actorと4 actorsで比較すると、置換表履歴が分かれるため2局面でtop-3順が異なった。一方、4 actors条件をcold cacheで2回実行した比較では12局面すべてのtop-3順が一致した。actor数と分配方式をv5 identityへ含め、実際の局面割当も成果物へ保存する。
- 旧v4方式を実バイナリ、1局面、11 workers × 1 thread、cold cacheで実行し、level 1から21までの11個の専用Consoleが同時に起動すること、全level・全alpha・CSV・JSON・SQLiteキャッシュの生成まで完走することを確認した。所要時間は約4.8秒だった。これはv5の動的actor方式の実測値ではなく、歴史値とする。
- 同じ旧v4実機確認で進捗表示を1秒間隔にし、最初のtaskが完了する前からlevel別の稼働プロセス数が表示され、値が得られたモデルだけが1モデル1行で更新されることを確認した。最終の途中集計値と全hintからの再集計値も一致した。これもv4の歴史的な確認結果である。
- v3では同じ100局面について、全levelで`hint 64`を使った修正版と、level 1から19だけ`hint 3`へ短縮した実装を比較し、全levelのtop-1・top-3集計値と全alphaのブレンド集計値が一致した。
- v3では同じ出力先を再実行した場合、保存済みhintをすべて検証して再利用し、100局面を約1秒で再集計できた。

## 30,000局面の最終結果

2026-07-21にWTHORのtest分割からseed 613で30,000局面を抽出し、v5をcold cache、16 workers × 2探索スレッドで実行した。同一の盤面・手番をまとめると24,340状態で、5,660局面分の重複計算を省略した。一致率の分母は重複を含む元の30,000局面である。角括弧内はWilson法による95%信頼区間を表す。

### ブレンド方策

| alpha | top-1一致率 | top-3一致率 |
| ---: | ---: | ---: |
| 0.0 | 63.393% [62.847%, 63.937%] | 90.733% [90.400%, 91.056%] |
| 0.2 | 64.220% [63.676%, 64.761%] | 91.203% [90.878%, 91.519%] |
| 0.4 | 64.563% [64.020%, 65.103%] | 91.593% [91.274%, 91.902%] |
| 0.6 | **65.147% [64.606%, 65.684%]** | **91.810% [91.494%, 92.115%]** |
| 0.8 | 64.173% [63.629%, 64.714%] | 91.623% [91.304%, 91.932%] |
| 1.0 | 57.147% [56.586%, 57.706%] | 87.610% [87.232%, 87.978%] |

試したalphaの中では、`alpha=0.6`がtop-1・top-3の両方で最高だった。alpha間の差は同じ局面に対する対応ありデータなので、差そのものの信頼区間や有意差を求める場合は別途対応あり解析を行う。

### Console単体

| level | top-1一致率 | top-3一致率 |
| ---: | ---: | ---: |
| 1 | 53.270% [52.705%, 53.834%] | 86.440% [86.048%, 86.823%] |
| 3 | 59.453% [58.897%, 60.008%] | 89.500% [89.148%, 89.842%] |
| 5 | 61.363% [60.811%, 61.913%] | 90.347% [90.007%, 90.676%] |
| 7 | 62.350% [61.800%, 62.897%] | 90.837% [90.505%, 91.158%] |
| 9 | 62.783% [62.235%, 63.329%] | 90.637% [90.302%, 90.961%] |
| 11 | 63.310% [62.763%, 63.854%] | 90.677% [90.342%, 91.000%] |
| 13 | 62.963% [62.415%, 63.508%] | 90.710% [90.376%, 91.033%] |
| 15 | 62.943% [62.395%, 63.488%] | 90.683% [90.349%, 91.007%] |
| 17 | 62.920% [62.372%, 63.465%] | 90.843% [90.512%, 91.164%] |
| 19 | 62.740% [62.191%, 63.285%] | 90.683% [90.349%, 91.007%] |
| 21 | 63.393% [62.847%, 63.937%] | 90.733% [90.400%, 91.056%] |

成果物は`output/random_wthor_test_n30000_seed613_workers16_threads2_blendhint64_consolehint3_v5/`へ保存した。最終JSON・CSVに加え、標準出力を`stdout.log`、30秒ごとの全進捗を含む標準エラーを`stderr.log`へUTF-8で保存した。

## 実行時間

| 条件 | 標本 | 一意状態 | 実行時間 | 平均CPU | 最大関連メモリ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧実装、同時hint上限4、各8 threads | 100,000 | 78,653 | 29.10時間 | 13.4% | - |
| v3実装、32 workers × 1 thread | 100 | 97 | 84.5秒 | 99.3% | 40.2 GiB |
| v3実装、16 workers × 2 threads | 100 | 97 | 67.8秒 | 78.0% | 20.1 GiB |
| v3実装、16 workers × 2 threads | 300 | 283 | 176.1秒 | 73.2% | 20.1 GiB |
| v5実装、16 workers × 2 threads、cold cache | 1,000 | 900 | 6分2秒 | 80.3% | 20.2 GiB |
| v5実装、16 workers × 2 threads、cold cache | 30,000 | 24,340 | 2時間42分0秒 | 79.6% | 21.0 GiB |

v3の短い標本では16 × 2が32 × 1より約20%速く、使用メモリも約半分だったため、同時探索スレッド数の既定値は16 × 2の32 threadsを維持する。旧実装・v3・v4の値は歴史値であり、今後の所要時間見積もりにはv5の実測を使う。

30,000局面ではhint探索が2時間38分41秒、最終検算・集計が1分40秒、総実行時間が2時間42分0秒だった。hint探索中の平均CPU使用率は79.6%、最大Egaroucidプロセス数は16、最大関連メモリは21.0 GiB、Console再試行は0回だった。合計40 actorを起動し、level 21には15 actor、level 19には最終的に16 actorを割り当てた。他の各levelは1 actorで完了した。

95.9%時点の定期ログでは、level 21の未claim局面が尽きた直後に稼働数が一時的に7へ見えたが、その7秒後にlevel 21完了を検知して残りのlevel 19を16 actorへ増員した。以後99.9%時点まで16プロセスを維持した。旧実装で見られた、処理対象が大量に残ったまま終盤の少数プロセス状態が継続する現象は再現しなかった。

## 30,000局面の統計精度

最終結果のWilson 95%信頼区間の半幅は、top-1で約±0.54から±0.56ポイント、top-3で約±0.31から±0.39ポイントだった。一致率50%という最悪条件でも半幅は約±0.57ポイントである。

100,000局面と比べると、標本誤差による区間幅は理論上約`sqrt(100000 / 30000) = 1.83`倍になる。通常のWilson区間は30,000標本を独立なBernoulli試行として扱い、同一対局内や重複局面間の相関は考慮しない。24,340一意状態を保守的な有効標本数とみなす場合、区間の半幅は約11%広くなる。また、同じ標本上のalpha間・level間の差を検定するには、各モデル単独の区間の重なりではなく対応あり解析が別途必要である。

## 修正済み1,000局面の参考結果

| alpha | top-1 | top-3 |
| ---: | ---: | ---: |
| 0.0 | 60.1% | 91.5% |
| 0.2 | 62.6% | 91.8% |
| 0.4 | 62.8% | 92.2% |
| 0.6 | 63.1% | 92.8% |
| 0.8 | 63.2% | 92.0% |
| 1.0 | 56.4% | 88.6% |

これはv5の実装修正確認用の固定標本であり、正式な一致率には上の30,000局面結果を使う。

## 実行方法

この32スレッドPCで30,000局面を実行するコマンドは次の通り。

```powershell
python src/tools/policy_network_human_like_ai/20_test_with_wthor/run_random_wthor_blend_experiment.py 30000 --workers 16 --egaroucid-threads 2
```

この並列条件は既定値と同じなので、次の短いコマンドでもよい。

```powershell
python src/tools/policy_network_human_like_ai/20_test_with_wthor/run_random_wthor_blend_experiment.py 30000
```

進捗は開始直後と、Console探索中は既定で30秒ごとに標準エラーへ表示される。表示間隔を10秒へ変更する場合は`--progress-interval-sec 10`を追加する。

標準出力と標準エラーをUTF-8の別ファイルへそのまま残すPowerShell例は次の通り。Windows PowerShellの`Tee-Object`へ日本語を含むnative stderrを直接通すと文字コード変換が入る場合があるため、`Start-Process`のリダイレクトを使う。

```powershell
$script = (Resolve-Path "src/tools/policy_network_human_like_ai/20_test_with_wthor/run_random_wthor_blend_experiment.py").Path
$out = Join-Path (Split-Path $script) "output/random_wthor_test_n30000_seed613_workers16_threads2_blendhint64_consolehint3_v5"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$arguments = @(
    "-u", $script, "30000",
    "--workers", "16",
    "--egaroucid-threads", "2",
    "--progress-interval-sec", "30",
    "--output-dir", $out
)
$process = Start-Process -FilePath "python" -ArgumentList $arguments `
    -WorkingDirectory (Split-Path $script) -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $out "stdout.log") `
    -RedirectStandardError (Join-Path $out "stderr.log") `
    -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "実験が終了コード $($process.ExitCode) で失敗しました"
}
```

別のPowerShellから`Get-Content -Encoding UTF8 -Wait <出力先>\stderr.log`を実行すれば、保存中の進捗ログをそのまま監視できる。

再試行可能なConsoleエラーは発生時点で警告として表示し、再試行上限に達したエラーやworkerの異常終了は直ちに標準エラーへ表示して実験を停止する。同じ実験条件・出力先でコマンドを再実行すると、利用可能なキャッシュを検証して再利用する。

30,000局面での既定の出力先は`output/random_wthor_test_n30000_seed613_workers16_threads2_blendhint64_consolehint3_v5/`である。v5は局面の分配とConsoleの置換表履歴がv4と異なるため、v4以前のキャッシュは自動再利用しない。主な出力は`random_wthor_blend_summary.json`、`random_wthor_blend_summary.csv`、`random_wthor_console_level_summary.csv`である。
