# Policy Networkブレンド強さ測定結果

関連issue: #613

実験日: 2026-07-17

## ブレンド式

Policy Networkの出力とEgaroucid for Consoleの方策は、次の式で合法手上に正規化して使います。

```text
ブレンド方策 = (Policy Networkの方策)^α × (Egaroucid for Consoleの方策)^(1 - α)
```

αは0.0から1.0まで指定できます。既定の実験設定では、0.0, 0.1, ..., 1.0を使います。
α=0.0はEgaroucid for Consoleの方策のみ、α=1.0はPolicy Networkの方策のみです。

## 今回の暫定対戦条件

| 項目 | 値 |
| --- | --- |
| Policy Network | 幅512、4層、WTHOR学習、50エポック終了後モデル |
| Policy Network重み | `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin` |
| 対戦相手 | Egaroucid for Console 7.8.1 level 21 |
| book | 使用しない |
| Egaroucidスレッド数 | 1 |
| 対戦数 | 各αにつき10局 |
| opening | `bin/problem/xot/openingslarge.txt` から先後入れ替えペアで使用 |
| 並列対戦数 | 2 |

勝率は、勝ちを1、引き分けを0.5、負けを0として計算しました。
Eloは各2者対戦内で平均1500になるように推定した相対値です。

## 結果

| α | α側の勝敗 | α側勝率 | α側平均石差 | α側Elo | level 21側Elo | Elo差 | 備考 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0勝1分9敗 | 5.0% | -6.6 | 1244.2 ± 247.0 | 1755.8 ± 247.0 | -511.5 | 10局の暫定値 |
| 1.0 | 0勝0分10敗 | 0.0% | -55.8 | 300.0 ± 53753.7 | 2700.0 ± 53753.7 | -2400.0 | 全敗のためElo推定は飽和しており、数値としては信頼しにくい |

## 出力ファイル

```text
src/tools/policy_network_human_like_ai/40_test_strength/output/alpha_0_5_vs_l21_10g
src/tools/policy_network_human_like_ai/40_test_strength/output/alpha_1_0_vs_l21_10g
```

