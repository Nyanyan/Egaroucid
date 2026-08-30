# MPC係数候補 回帰検査

- 実行開始: 2026-08-30T19:16:40.3826776+09:00
- Git commit: `2dd111a2f48bc56f91586ef39574e9227fc1f94f`
- コンパイラ: `clang version 22.1.8 (https://github.com/llvm/llvm-project.git ca7933e47d3a3451d81e72ac174dcb5aa28b59d1)`
- コンパイル条件: `-O2 -std=c++20 -march=native -pthread -lws2_32`
- 結果: 23 / 23 件成功、0 件失敗

| ID | 検査内容 | マクロ定義 | compile | run | compile時間 (ms) | run時間 (ms) |
|---|---|---|---:|---:|---:|---:|
| `coefficient_production_precalc_0` | 係数候補: production, USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6385 | 1582 |
| `coefficient_mid_variant_1_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 27607 | 11623 |
| `coefficient_mid_variant_2_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 10128 | 838 |
| `coefficient_mid_variant_3_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 7092 | 625 |
| `coefficient_mid_variant_4_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6578 | 613 |
| `coefficient_mid_variant_5_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6450 | 610 |
| `coefficient_end_sigma_variant_1_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6796 | 768 |
| `coefficient_end_sigma_variant_2_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 7383 | 654 |
| `coefficient_production_precalc_1` | 係数候補: production, USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 8491 | 1097 |
| `coefficient_mid_variant_1_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 7274 | 1543 |
| `coefficient_mid_variant_2_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 7466 | 645 |
| `coefficient_mid_variant_3_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 9642 | 2753 |
| `coefficient_mid_variant_4_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 8088 | 766 |
| `coefficient_mid_variant_5_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 6744 | 585 |
| `coefficient_end_sigma_variant_1_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 8971 | 716 |
| `coefficient_end_sigma_variant_2_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 6740 | 611 |
| `mid_sigma_scale_precalc_0` | production中盤MPC: USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 6946 | 2809 |
| `mid_sigma_scale_precalc_1` | production中盤MPC: USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 7599 | 2202 |
| `end_recalibrated_variant_0` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=0 | `END_MPC_RECALIBRATED_VARIANT=0` | 成功 | 成功 | 9664 | 713 |
| `end_recalibrated_variant_1` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=1 | `END_MPC_RECALIBRATED_VARIANT=1` | 成功 | 成功 | 7273 | 1348 |
| `end_recalibrated_variant_2` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=2 | `END_MPC_RECALIBRATED_VARIANT=2` | 成功 | 成功 | 7279 | 671 |
| `end_recalibrated_variant_3` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=3 | `END_MPC_RECALIBRATED_VARIANT=3` | 成功 | 成功 | 7476 | 908 |
| `end_recalibrated_variant_4` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=4 | `END_MPC_RECALIBRATED_VARIANT=4` | 成功 | 成功 | 7340 | 1394 |
