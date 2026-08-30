# 終盤 MPC の浅い探索深度を候補別に再学習したオフライン比較

## 比較の概要

現行の浅い探索深度から `-2`、`0`、`+2`、`+4` 手分を比較した。
各候補について、その候補の浅い探索結果だけを使って標準偏差と上下の分位点を学習し直した。現行モデルの誤差幅を別の深度へ流用した比較ではない。

- `-2` は浅い探索そのものは安いが、枝を切れる回数が減り、開発用データと 6 月 holdout の両方で総ノード数が増えた。
- `+4` は浅い探索が高価であるうえ、6 月 holdout で 4 石の誤 cut が 1 件あった。
- 4 開発セットの root-set 交差検証では、推定総ノード比は `0` が 0.2439、`+2` が 0.2613 だった。
- 独立の 6 月 holdout では、推定総ノード比は `0` が 0.3566、`+2` が 0.3466 だった。
- 深さ 10～18 を含む補助 root holdout では、推定総ノード比は `0` が 0.8349、`+2` が 0.8364 だった。
- `shallow_nodes` は full-window 探索の値であり、実際の MPC の null-window 探索とはノード数が異なる。コンパイル済みエンジンの実時間測定は別表で扱う。

## 比較した深度

| 終盤の深い探索深度 | -2 | 現行 0 | +2 | +4 |
|---:|---:|---:|---:|---:|
| 10 | 2 | 4 | 6 | 8 |
| 11 | 3 | 5 | 7 | 9 |
| 12 | 2 | 4 | 6 | 8 |
| 13 | 3 | 5 | 7 | 9 |
| 14 | 2 | 4 | 6 | 8 |
| 15 | 5 | 7 | 9 | 11 |
| 16 | 4 | 6 | 8 | 10 |
| 17 | 5 | 7 | 9 | 11 |
| 18 | 4 | 6 | 8 | 10 |

## データ分割

`root_id` を分割単位とした。下表の 6 セットは、互いの `root_id` の交差がすべて 0 である。

| 用途 | データ | root 数 | MPC context 数 | 候補ごとの一意な評価誤差標本数 |
|---|---|---:|---:|---:|
| 開発 | `end_mpc_enriched_dev_743` | 65 | 800 | 573 |
| 開発 | `end_mpc_enriched_dev_202607` | 103 | 1,015 | 754 |
| 開発 | `end_mpc_enriched_dev_202608a` | 105 | 1,000 | 790 |
| 開発 | `end_mpc_enriched_dev_202608b` | 105 | 1,000 | 765 |
| 最終確認 | `end_mpc_enriched_holdout_202606` | 105 | 1,000 | 758 |
| 深さ 18 の補助データ | `end_mpc_v2_admission_dev` | 35 | 2,648 | 631～632 |

4 開発セットは合計 378 root、3,815 context、2,882 個の一意な評価誤差標本を持つ。6 月 holdout は候補選択用の fit に入れていない。

既存の 4 開発セットと 6 月 holdout が持つ context は深さ 10～17 であり、深さ 18 は 0 件だった。`end_mpc_v2_admission_dev` には深さ 10～18 が各およそ 300 context あり、深さ 18 は 303 context、70 個の一意な評価誤差標本を持つ。この 35 root を `root_id` の辞書順で 24 root の学習側と 11 root の確認側に分けた。確認側は全深度 837 context、うち深さ 18 が 98 context である。

ただし `end_mpc_v2_admission_dev` 自体は過去の方針調整で既に調べたデータである。この 24/11 分割は深さ 18 の補助確認であり、新しい独立 holdout とは扱わない。

## 学習方法

候補ごとに次を行った。

1. 各 context から静的評価の行と、その候補深度の浅い探索の行だけを選んだ。
2. 浅い探索のモデルは現行実装と同じく `exact_value = shallow_value + residual` とした。
3. `fit_sigma(..., "gap_shallow", 64.0)` により、選んだ候補だけから標準偏差を再学習した。
4. 正規化した residual の分位点を、選択率 74%、88%、93% の各上下端について候補別に再計算した。
5. 実装へ入る整数誤差幅は、丸め済みの現行表を倍率変換せず、次式で浮動小数点値から計算し直した。

   - high 側: `ceil(1.10 * lower_tail[level] * sigma[depth])`
   - low 側: `ceil(1.10 * upper_tail[level] * sigma[depth])`

6. 静的評価 cut のモデルと、浅い探索を開始する静的評価の余裕 4 は候補間で同じにした。

現行の `evaluate_end_probcut_final_policy_v2.py` は実行時深度表が固定されているため、入力コピー内の `trace_shallow_depth` を候補深度へ置換した。レポートでは `fitted_identity` を候補の結果として使用した。`runtime_final` は現行の hard-code 表を評価する欄であり、offset が 0 以外の比較には使っていない。同じ理由で `incomplete` は offset が 0 以外では実際の欠損数を表さない。

## 標準偏差モデルの精度

下表は 4 開発セットを 1 セットずつ除外して学習した結果の加重平均である。bias は `exact_value - shallow_value` の平均で、単位は石数である。

| 候補 | RMSE | MAE | bias |
|---|---:|---:|---:|
| -2 | 6.032 | 4.589 | +1.434 |
| 0 | 4.851 | 3.709 | +1.082 |
| +2 | 3.783 | 2.877 | +0.679 |
| +4 | 2.964 | 2.197 | +0.236 |

6 月 holdout でも同じ順序だった。

| 候補 | RMSE | MAE | bias |
|---|---:|---:|---:|
| -2 | 6.181 | 4.737 | +1.540 |
| 0 | 5.043 | 3.873 | +1.443 |
| +2 | 3.870 | 2.999 | +0.846 |
| +4 | 3.051 | 2.265 | +0.402 |

浅い探索を深くすれば値の予測精度は上がる。しかし、MPC 全体では浅い探索に掛かる費用との釣り合いが必要である。

候補別に再学習した上下の分位点は次の通りである。各欄は 74% / 88% / 93% の順である。

| 候補 | lower tail | upper tail |
|---|---|---|
| -2 | 0.7843 / 1.2466 / 1.5851 | 1.2897 / 1.7436 / 2.0314 |
| 0 | 0.7925 / 1.2227 / 1.6066 | 1.2641 / 1.7118 / 1.9813 |
| +2 | 0.9209 / 1.3061 / 1.5673 | 1.2279 / 1.7096 / 2.0756 |
| +4 | 1.0382 / 1.4855 / 1.7854 | 1.1153 / 1.7304 / 1.8588 |

各深度の標準偏差、上下の分位点、整数誤差幅の全表は [parameters.csv](parameters.csv) にある。深さ 18 の補助学習を加えた表は [parameters_with_admission.csv](parameters_with_admission.csv) にある。

深さ 18 を補助データで直接学習した場合は次の値になった。

| 候補 | 浅い探索深度 | sigma | 74% high/low | 88% high/low | 93% high/low |
|---|---:|---:|---:|---:|---:|
| -2 | 4 | 6.300 | 6 / 9 | 9 / 13 | 11 / 15 |
| 0 | 6 | 4.979 | 5 / 7 | 7 / 10 | 9 / 11 |
| +2 | 8 | 4.578 | 5 / 7 | 7 / 9 | 8 / 11 |
| +4 | 10 | 3.605 | 5 / 5 | 6 / 7 | 8 / 8 |

現行の深さ 18 の sigma は、深さ 18 の標本がない状態で同じ浅い探索深度の親分散へ fallback した値である。深さ 18 の表を変更する場合は、上の補助データまたは新しい独立データを使って直接 fit する必要がある。

## cut と推定総ノード数

`推定ノード比` は次の値である。

`(静的評価で候補外にならなかった浅い探索のノード数 + cut できなかった場合の深い探索のノード数) / 全 context の深い探索ノード数`

浅い探索のノード数は置換表を空にした 100% の full-window 探索で収集されている。実際の MPC の null-window 探索と同じノード数ではないため、候補選別用の近似値である。

### 4 開発セットの root-set 交差検証

| 候補 | context | cut | 誤 cut | 2 石以上 | 4 石以上 | 推定ノード比 |
|---|---:|---:|---:|---:|---:|---:|
| -2 | 3,815 | 2,182 | 13 | 7 | 5 | 0.3171 |
| 0 | 3,815 | 2,563 | 9 | 2 | 1 | 0.2439 |
| +2 | 3,815 | 2,796 | 4 | 1 | 0 | 0.2613 |
| +4 | 3,815 | 2,958 | 4 | 2 | 1 | 0.3607 |

### 独立の 6 月 holdout

| 候補 | context | cut | 誤 cut | 2 石以上 | 4 石以上 | 推定ノード比 |
|---|---:|---:|---:|---:|---:|---:|
| -2 | 1,000 | 579 | 4 | 0 | 0 | 0.4139 |
| 0 | 1,000 | 662 | 3 | 0 | 0 | 0.3566 |
| +2 | 1,000 | 730 | 3 | 0 | 0 | 0.3466 |
| +4 | 1,000 | 777 | 2 | 1 | 1 | 0.4022 |

### 深さ 10～18 を含む補助 root holdout

| 候補 | context | cut | 誤 cut | 2 石以上 | 4 石以上 | 推定ノード比 |
|---|---:|---:|---:|---:|---:|---:|
| -2 | 837 | 156 | 2 | 1 | 0 | 0.8389 |
| 0 | 837 | 183 | 0 | 0 | 0 | 0.8349 |
| +2 | 837 | 200 | 0 | 0 | 0 | 0.8364 |
| +4 | 837 | 229 | 1 | 0 | 0 | 0.8402 |

このうち深さ 18 の 98 context だけを取り出すと、推定ノード比は `-2: 0.8893`、`0: 0.8835`、`+2: 0.8827`、`+4: 0.8856` だった。全候補で誤 cut は 0 件だった。`0` と `+2` の差は 0.09% で、この近似値から優劣を確定できる大きさではない。

詳細値は [summary.csv](summary.csv)、[folds.csv](folds.csv)、[admission_holdout_summary.csv](admission_holdout_summary.csv)、[admission_deep18_summary.csv](admission_deep18_summary.csv) にある。

## 再実行コマンド

offset `+2` の候補別 fit と 6 月 holdout 評価は次のコマンドで再実行できる。他の候補は `offset_p2` を `offset_m2`、`offset_p0`、`offset_p4` に置き換える。

```powershell
python src/tools/probcut/evaluate_end_probcut_final_policy_v2.py `
  --input dev743=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev743.jsonl `
  --input dev202607=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202607.jsonl `
  --input dev202608a=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202608a.jsonl `
  --input dev202608b=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202608b.jsonl `
  --evaluation-input benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/holdout202606.jsonl `
  --output benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/report.json
```

浅い探索の残差モデルを root-set 交差検証するコマンドは次である。

```powershell
python src/tools/probcut/screen_end_probcut_v2.py `
  --mode shallow `
  --input dev743=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev743.jsonl `
  --input dev202607=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202607.jsonl `
  --input dev202608a=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202608a.jsonl `
  --input dev202608b=benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/dev202608b.jsonl `
  --output benchmark/mpc_depth_aggressiveness_20260830/end_offline_refit/offset_p2/model_screen.json
```

## 新しい深さ 18 の独立 holdout を収集する場合

既存 holdout には深さ 18 がない。別 root から追加収集する場合は、baseline の trace executable を使い、既存の全データを除外する。

```powershell
python src/tools/probcut/collect_end_probcut_samples.py `
  --positions benchmark/mpc_quantile_stage1_202608_fresh_pool/corpus.jsonl `
  --exe bin/end_mpc_context_tool_v2.exe `
  --output benchmark/mpc_depth_aggressiveness_20260830/end_depth18_independent_holdout `
  --seed 20260830 `
  --descendant-empty 20 `
  --descendants-per-root 1 `
  --mpc-levels 0,1,2 `
  --min-deep 10 `
  --max-deep 18 `
  --trace-contexts-per-depth 4 `
  --shallow-offsets=-4,-2,0,2,4 `
  --max-samples 15000 `
  --workers 20 `
  --trace-timeout 45 `
  --score-timeout 90 `
  --max-roots-per-empty 15 `
  --exclude-samples benchmark/mpc_quantile_stage1_74314921_descendant_samples/samples.jsonl `
  --exclude-samples benchmark/mpc_quantile_stage1_202607_descendant_samples/samples.jsonl `
  --exclude-samples benchmark/mpc_quantile_stage1_202608_fresh_final_samples/samples.jsonl `
  --exclude-samples benchmark/mpc_quantile_stage1_202608_fresh2_final_samples/samples.jsonl `
  --exclude-samples benchmark/mpc_quantile_stage1_202606_final_samples/samples.jsonl `
  --exclude-samples benchmark/end_mpc_v2_admission_dev/samples.jsonl
```

中断後は同じコマンドに `--resume` を加える。

## 実装上の注意

`END_MPC_SHALLOW_DEPTH_OFFSET` は汎用の `mpc_impl<true>` にだけ効く。深さ 10～18、選択率 74/88/93 の再較正済み終盤 MPC は `mpc_end_recalibrated_shallow()` から `END_MPC_SHALLOW_DEPTH[]` を直接参照するため、この macro を変えても今回の対象経路の浅い探索深度は変わらない。

この経路を比較するには、`END_MPC_SHALLOW_DEPTH[]` の候補表と、その候補専用に再学習した `END_MPC_SHALLOW_SIGMA`、上下 tail、整数誤差幅を一組としてコンパイルする必要がある。
