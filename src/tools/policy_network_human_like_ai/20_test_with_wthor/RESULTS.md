# WTHORテストデータにおけるブレンド方策の人間着手一致率

関連issue: #613

実験日: 2026-07-17

## 評価対象

WTHORの全8,035,282局面を、学習時と同じseed `613`でシャッフルし、8:1:1に分割しました。
このうちテストデータは803,529局面です。

Egaroucid for Console level 21による全合法手評価は計算時間が長いため、今回はテストデータからseed `613`で無作為抽出した1,000局面を評価しました。
したがって、以下はテストデータ全件の結果ではなく、固定された1,000局面標本の結果です。

| 項目 | 値 |
| --- | --- |
| Policy Network | 幅512、隠れ層4層、50エポック終了後 |
| Policy Network重み | `../10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin` |
| WTHOR全局面数 | 8,035,282 |
| WTHORテスト局面数 | 803,529 |
| 評価した局面数 | 1,000 |
| データ分割seed | 613 |
| テストデータ内の抽出seed | 613 |
| Egaroucid | Egaroucid for Console 7.8.1、level 21、bookなし |
| α | 0.0から1.0まで0.1刻み |
| 同時実行した評価処理数 | 16 |
| 1評価処理当たりのEgaroucidスレッド数 | 2 |
| hint未計算状態からの実行時間 | 約323秒 |

## 評価方法

ブレンド方策は次の式で計算しました。

```text
ブレンド方策 = (Policy Networkの方策)^α × (Egaroucidの方策)^(1 - α)
```

各局面で合法手以外を除外した後、確率が最も高い手とWTHORの実着手が一致した場合を1位一致としました。
上位3手一致率は、WTHORの実着手が確率上位3手に入った割合です。

95%信頼区間は、1位一致率についてWilson法で計算しました。

## 結果

| α | 1位一致数 | 1位一致率 | 1位一致率の95%信頼区間 | 上位3手一致率 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 682 | 68.2% | 65.2%から71.0% | 93.7% |
| 0.1 | 619 | 61.9% | 58.8%から64.9% | 92.0% |
| 0.2 | 622 | 62.2% | 59.2%から65.2% | 92.0% |
| 0.3 | 623 | 62.3% | 59.3%から65.3% | 92.1% |
| 0.4 | 623 | 62.3% | 59.3%から65.3% | 92.4% |
| 0.5 | 627 | 62.7% | 59.7%から65.6% | 92.9% |
| 0.6 | 631 | 63.1% | 60.1%から66.0% | 92.6% |
| 0.7 | 635 | 63.5% | 60.5%から66.4% | 92.3% |
| 0.8 | 633 | 63.3% | 60.3%から66.2% | 92.1% |
| 0.9 | 624 | 62.4% | 59.4%から65.3% | 91.8% |
| 1.0 | 564 | 56.4% | 53.3%から59.4% | 88.6% |

この1,000局面では、全条件中ではα=0.0が最も高い1位一致率でした。
αが0より大きい条件ではα=0.7が最も高く、α=1.0より7.1ポイント高い結果でした。
ただし、1,000局面の標本評価であるため、近い値を示したα同士の小さな差は確定的な優劣を意味しません。

## hint計算の共有

各局面について、Policy Networkの方策とEgaroucidのhint結果をそれぞれ1回だけ計算し、その2つの分布から11個のαをまとめて評価しました。
1,000局面の評価におけるhintキャッシュ参照回数は1,000回であり、11,000回ではありません。

1回目の実行では、重複局面の再利用を含めて900種類のhint結果をSQLiteへ保存しました。
同じキャッシュを使った2回目の集計では1,000局面すべてがキャッシュに一致し、約1.3秒で完了しました。

実行コマンドは次の通りです。

```powershell
python src/tools/policy_network_human_like_ai/20_test_with_wthor/evaluate_wthor_blend_human_match.py `
  --data-split test `
  --split-seed 613 `
  --sample-positions 1000 `
  --sample-seed 613 `
  --blend-params 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 `
  --jobs 16 `
  --egaroucid-threads 2 `
  --hint-cache-db src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000/hint_score_cache.sqlite3 `
  --output-dir src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000
```

## 出力

```text
src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000/wthor_blend_human_match.json
src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000/wthor_blend_human_match_topn.csv
src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000/wthor_blend_human_match_by_move10.csv
src/tools/policy_network_human_like_ai/20_test_with_wthor/output/wthor_test_alpha_0p1_sample1000/hint_score_cache.sqlite3
```
