# MPC係数候補 回帰検査

- 実行開始: 2026-08-30T19:37:09.7634987+09:00
- Git commit: `6e3fb11b8b7987d085db89b780a8ca5bdaa32a17`
- コンパイラ: `clang version 22.1.8 (https://github.com/llvm/llvm-project.git ca7933e47d3a3451d81e72ac174dcb5aa28b59d1)`
- コンパイル条件: `-O2 -std=c++20 -march=native -pthread -lws2_32`
- 結果: 23 / 23 件成功、0 件失敗

| ID | 検査内容 | マクロ定義 | compile | run | compile時間 (ms) | run時間 (ms) |
|---|---|---|---:|---:|---:|---:|
| `coefficient_production_precalc_0` | 係数候補: production, USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 7286 | 663 |
| `coefficient_mid_variant_1_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 7463 | 652 |
| `coefficient_mid_variant_2_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 7147 | 546 |
| `coefficient_mid_variant_3_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6176 | 556 |
| `coefficient_mid_variant_4_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 5785 | 549 |
| `coefficient_mid_variant_5_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 5464 | 549 |
| `coefficient_end_sigma_variant_1_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 5564 | 554 |
| `coefficient_end_sigma_variant_2_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 5532 | 545 |
| `coefficient_production_precalc_1` | 係数候補: production, USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5525 | 563 |
| `coefficient_mid_variant_1_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5602 | 556 |
| `coefficient_mid_variant_2_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5662 | 560 |
| `coefficient_mid_variant_3_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5525 | 547 |
| `coefficient_mid_variant_4_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5518 | 535 |
| `coefficient_mid_variant_5_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5558 | 554 |
| `coefficient_end_sigma_variant_1_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5526 | 554 |
| `coefficient_end_sigma_variant_2_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5541 | 546 |
| `mid_sigma_scale_precalc_0` | production中盤MPC: USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 5566 | 1916 |
| `mid_sigma_scale_precalc_1` | production中盤MPC: USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 5567 | 1902 |
| `end_recalibrated_variant_0` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=0 | `END_MPC_RECALIBRATED_VARIANT=0` | 成功 | 成功 | 5519 | 530 |
| `end_recalibrated_variant_1` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=1 | `END_MPC_RECALIBRATED_VARIANT=1` | 成功 | 成功 | 5453 | 546 |
| `end_recalibrated_variant_2` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=2 | `END_MPC_RECALIBRATED_VARIANT=2` | 成功 | 成功 | 5477 | 547 |
| `end_recalibrated_variant_3` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=3 | `END_MPC_RECALIBRATED_VARIANT=3` | 成功 | 成功 | 5459 | 558 |
| `end_recalibrated_variant_4` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=4 | `END_MPC_RECALIBRATED_VARIANT=4` | 成功 | 成功 | 5486 | 601 |
