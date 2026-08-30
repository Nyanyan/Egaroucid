# MPC係数候補 回帰検査

- 実行開始: 2026-08-30T19:41:45.2942415+09:00
- Git commit: `ef88935f6006579b0e519fa6c65063add6328b8f`
- コンパイラ: `clang version 22.1.8 (https://github.com/llvm/llvm-project.git ca7933e47d3a3451d81e72ac174dcb5aa28b59d1)`
- コンパイル条件: `-O2 -std=c++20 -march=native -pthread -lws2_32`
- 結果: 24 / 24 件成功、0 件失敗

| ID | 検査内容 | マクロ定義 | compile | run | compile時間 (ms) | run時間 (ms) |
|---|---|---|---:|---:|---:|---:|
| `coefficient_production_precalc_0` | 係数候補: production, USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4902 | 517 |
| `coefficient_mid_variant_1_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4928 | 492 |
| `coefficient_mid_variant_2_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4859 | 475 |
| `coefficient_mid_variant_3_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4843 | 498 |
| `coefficient_mid_variant_4_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4839 | 480 |
| `coefficient_mid_variant_5_precalc_0` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4873 | 501 |
| `coefficient_end_sigma_variant_1_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4902 | 479 |
| `coefficient_end_sigma_variant_2_precalc_0` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4907 | 506 |
| `coefficient_production_precalc_1` | 係数候補: production, USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4892 | 496 |
| `coefficient_mid_variant_1_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4915 | 498 |
| `coefficient_mid_variant_2_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4913 | 478 |
| `coefficient_mid_variant_3_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=3, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4894 | 486 |
| `coefficient_mid_variant_4_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=4, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4901 | 518 |
| `coefficient_mid_variant_5_precalc_1` | 係数候補: MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1 | `MID_MPC_RECALIBRATED_VARIANT=5, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4934 | 507 |
| `coefficient_end_sigma_variant_1_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=1, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4901 | 503 |
| `coefficient_end_sigma_variant_2_precalc_1` | 係数候補: END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1 | `END_MPC_SIGMA_MODEL_VARIANT=2, USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4866 | 498 |
| `mid_sigma_scale_precalc_0` | production中盤MPC: USE_MPC_PRE_CALCULATION=0 | `USE_MPC_PRE_CALCULATION=0` | 成功 | 成功 | 4901 | 1697 |
| `mid_sigma_scale_precalc_1` | production中盤MPC: USE_MPC_PRE_CALCULATION=1 | `USE_MPC_PRE_CALCULATION=1` | 成功 | 成功 | 4869 | 1737 |
| `end_recalibrated_variant_0` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=0 | `END_MPC_RECALIBRATED_VARIANT=0` | 成功 | 成功 | 4796 | 470 |
| `end_recalibrated_variant_1` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=1 | `END_MPC_RECALIBRATED_VARIANT=1` | 成功 | 成功 | 4816 | 500 |
| `end_recalibrated_variant_2` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=2 | `END_MPC_RECALIBRATED_VARIANT=2` | 成功 | 成功 | 4846 | 507 |
| `end_recalibrated_variant_3` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=3 | `END_MPC_RECALIBRATED_VARIANT=3` | 成功 | 成功 | 4829 | 509 |
| `end_recalibrated_variant_4` | 終盤MPC: END_MPC_RECALIBRATED_VARIANT=4 | `END_MPC_RECALIBRATED_VARIANT=4` | 成功 | 成功 | 4819 | 516 |
| `coefficient_refit_rejects_global_sigma_scale` | 係数再推定候補と全体sigma倍率の併用をコンパイル時に拒否する | `MID_MPC_RECALIBRATED_VARIANT=1, MPC_SIGMA_SCALE=0.9` | 意図通り拒否 | 対象外 | 1513 | - |
