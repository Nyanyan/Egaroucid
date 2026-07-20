# 推定Eloと人間との着手一致率

関連issue: #613

作成日: 2026-07-20

![推定Eloレーティングと人間との1位着手一致率](rating_vs_human_top1.png)

## 図の定義

- 横軸は、総当たり戦から推定したEloレーティングである。
- 縦軸は、WTHORテストデータ100,000局面における人間との1位着手一致率である。
- 横線は推定Eloレーティングの68%信頼区間、縦線は1位着手一致率の95%信頼区間を表す。
- 青い丸はEgaroucid for Console単体、赤い四角はブレンド方策を表す。
- 推定Eloの信頼区間は対称なので、計算した半幅を点推定値の左右へ加えた。
- 1位着手一致率の信頼区間は上下で幅が異なるため、表に記録された下限と上限をそのまま使用した。
- 各点のラベルは、他のラベル、点、信頼区間、凡例、注記との重なりがない候補位置を自動探索して配置した。重ならない位置を見つけられない場合は、重なった図を保存せずエラーとして終了する。

強さと人間との着手一致率の両方が提示されている16モデルだけを作図対象とした。ランダム打ちは人間との着手一致率がなく、Egaroucid for Console level 21は総当たり戦の推定Eloが提示されていないため、図に含めていない。

## 入力データ

作図に使用した値は[`rating_vs_human_top1_data.csv`](rating_vs_human_top1_data.csv)に保存した。元の表との対応は次のとおりである。

- Egaroucid for Console: level 1、3、5、7、9、11、13、15、17、19
- ブレンド方策: α=0.0、0.2、0.4、0.6、0.8、1.0

## 推定Eloの信頼区間

強さ測定で使用した[`elo_rating_backcal.py`](../../../../bin/elo_rating_backcal.py)は、Elo推定時のヘッセ行列から各点推定値の標準誤差を求め、正規分布に基づく対称な信頼区間を計算する。元の総当たり戦では信頼水準に0.95を指定していた。

保存済みの`../40_test_strength/output/xot_50sets_17players/strength_win_rate_matrix.tsv`から勝率行列と対戦数行列を復元し、同じ関数で信頼水準0.68の区間を再計算した。併せて信頼水準0.95でも再計算し、保存済みの全17参加者の95%信頼区間半幅を誤差0で再現できることを確認した。

作図スクリプトでは、入力CSVに保存した95%信頼区間半幅を標準誤差へ戻し、指定した信頼水準の正規分布の係数を掛けて表示用の半幅を計算する。既定値は0.68である。例えば、Egaroucid level 1の68%信頼区間半幅は389.679、α=0.0は362.178、α=1.0は390.467である。

## 再生成

```powershell
python src/tools/policy_network_human_like_ai/50_rating_vs_human_match/plot_rating_vs_human_match.py
```

このコマンドは同じフォルダにPNG画像を出力する。

横方向の信頼水準は次のように変更できる。

```powershell
python src/tools/policy_network_human_like_ai/50_rating_vs_human_match/plot_rating_vs_human_match.py `
  --rating-confidence 0.95
```
