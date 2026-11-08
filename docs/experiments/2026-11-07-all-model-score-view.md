# 全モデルの測定スコア（151実行）

元catalog SHA256: e45bc9c5d097d04cbdfaca9e0dde3a3ac8a95e3a227fff6839714f2109a4c648。151行を全件使用、146 completed / 4 partial / 1 historical unfinished。記録上のlabelは75種類（物理重み、overlay identity、baseline別名を含むため、75種類の修正モデルという意味ではない）。

同じlabelの複数測定を列挙し、平均・最良値選択をしない。code90、95項目screen内code（HE/MBPP）、hard30（LCB/Sci）、known64、full14内codeは異なる集合。全14総合は次節へ分離。番号は末尾の全151行表と対応。旧protocolのnative ON/OFFがcatalogにない場合は推測せず、確証のない統合比較を避ける。

## Code・部分集合（full14以外）

| 記録モデル名 / label先頭 | 各測定 code %（protocol/split、#run） |
|---|---|
| 原モデル calibration / calib-base-d | calib/dev 71.073716/partial [#1]; calib/dev — [#2]; calib/dev — [#5]; calib/dev — [#7] |
| 原モデル calibration / calib-base-t | calib/test — [#3]; calib/test 74.629671/partial [#4]; calib/test — [#6]; calib/test —/partial [#8] |
| O 原モデル / O 原モデル・観測ON / cab440f871a3 | screen95/dev 85.000000 [#16]; livecodebench20/dev 70.000000 [#69]; reserve1491/test 77.428571/partial [#77]; code90/dev 72.575397 [#88]; known64_test30/test 78.261905 [#89]; code90/dev 71.325397 [#92]; code90/dev 69.242063 [#96]; code90/dev 69.103175 [#98]; code90/dev 70.769841 [#101]; code90/dev 70.700397 [#105]; code90/dev 71.186508 [#108] |
| scalar-l28-minus4 / 9832d5bdd6af | screen95/dev 85.000000 [#17] |
| scalar-l28-plus4 / cf51e7e0cf5b | screen95/dev 85.000000 [#18] |
| scalar-l23-minus4 / 47e907583fbc | screen95/dev 85.000000 [#19] |
| scalar-l23-plus4 / d16d1c7a5f33 | screen95/dev 85.000000 [#20] |
| scalar-l41-minus4 / 8f2a101edd53 | screen95/dev 85.000000 [#21] |
| scalar-l41-plus4 / e1d557d7dd91 | screen95/dev 85.000000 [#22] |
| sparse-code-margin3 / 011d2df0de91 | screen95/dev 85.000000 [#23] |
| sparse-action-margin3 / c32c5c24d2dd | screen95/dev 85.000000 [#24] |
| joint-purpose-balanced-small / 003dc42c0bc0 | screen95/dev 90.000000 [#25]; screen95/dev 85.000000 [#27]; hard_code30/dev 50.150794 [#28]; subset162/dev — [#29] |
| joint-purpose-programming-set / b01c699175e6 | screen95/dev 85.000000 [#30]; screen95/dev 85.000000 [#32]; hard_code30/dev 55.150794 [#33] |
| joint-purpose-algorithm-set / 46692e90c87d | screen95/dev 85.000000 [#34]; hard_code30/dev 53.365079 [#36] |
| joint-purpose-workflow-language / 01f83a9479a4 | screen95/dev 85.000000 [#35] |
| sparse-code-action6 / d56d26a97391 | screen95/dev 85.000000 [#37] |
| joint-purpose-balanced-medium / 23bcae0f0d67 | screen95/dev 90.000000 [#38]; screen95/dev 85.000000 [#39] |
| joint-purpose-chat-set / 14a9fafa06c1 | screen95/dev 85.000000 [#40]; hard_code30/dev 57.650794 [#42] |
| finalnorm-minus1pct / 5b761d754aa6 | screen95/dev 85.000000 [#43]; screen95/dev 90.000000 [#46] |
| finalnorm-plus1pct / c5be8132c454 | screen95/dev 90.000000 [#44]; screen95/dev 90.000000 [#45]; hard_code30/dev 58.206349 [#48] |
| donor-text-c1 / 3353278c3610 | screen95/dev 70.000000 [#49]; hard_code30/dev 55.984127 [#50] |
| joint-programming-finalnorm-plus1pct / b626f1189623 | screen95/dev 90.000000 [#52]; hard_code30/dev 57.650794 [#53] |
| finalnorm-plus0p5pct / fb494e89a52e | screen95/dev 90.000000 [#55]; hard_code30/dev 55.150794 [#56] |
| donor-text-c0p25 / 44a68b0f35b8 | screen95/dev 85.000000 [#58]; hard_code30/dev 50.150794 [#59] |
| joint-donor-quarter-finalnorm-plus1pct / 81981febd92b | subset107/dev 85.000000 [#61] |
| hs0020 / 6d098298bbf5 | screen95/dev 85.000000 [#64] |
| 0397 / ba5e6ab526c8 | screen95/dev 85.000000 [#65]; hard_code30/dev 57.650794 [#66]; livecodebench20/dev 75.000000 [#68]; reserve1490/test 82.547619 [#78]; code90/dev 70.075397 [#79]; code90/dev 70.492063 [#80]; known64_test30/test 77.011905 [#84] |
| 0485 / b9617ac7de12 | screen95/dev 85.000000 [#70]; hard_code30/dev 60.150794 [#71] |
| localG-C / cc3bf6d50a1a | screen95/dev 90.000000 [#73]; hard_code30/dev 55.150794 [#74] |
| 0397＋localG-C / cc824166b483 | hard_code30/dev 60.150794 [#75]; screen95/dev 85.000000 [#76] |
| B native-contracts / 564ae5fc1efd | code90/dev 73.408730 [#81]; known64_test30/test 76.476190 [#85]; code90/dev 70.075397 [#102] |
| G native＋matrix / b006bf7e344a | code90/dev 70.769841 [#82] |
| F matrix-balanced / 9d2bfb78a08b | code90/dev 67.575397 [#83] |
| correctLL balanced donor / f5137ab7e45d | code90/dev 71.325397 [#90] |
| correctLL JOINT / 3d45f526266c | code90/dev 70.492063 [#91] |
| H coherent donor1/16 / 0f9775ba6a24 | code90/dev 72.158730 [#93]; code90/dev 70.769841 [#103] |
| I math GRPO c1 / 9acfbe674fe1 | code90/dev 70.908730 [#97] |
| J weighted-reasoning c1 / dbf0a54c2788 | code90/dev 72.158730 [#99] |
| K＝B＋H / fd8ba3ccb923 | code90/dev 70.908730 [#100] |
| L＝O＋K1 / f8f936eee953 | code90/dev 67.182540 [#104]; code90/dev 68.075397 [#109] |
| P 公開base・元protocol / P 公開改変base・保持確認 / P・Nの同時対照 / heretic-public-base / 3dfb3bd1db0a | code90/dev 73.825397 [#106]; code90/dev 74.242063 [#112]; code90/dev 73.269841 [#114]; code90/dev 72.575397 [#117] |
| M＝P＋K1 / a09697f0f7cd | code90/dev 67.908730 [#107] |
| N＝P＋K1 text1/16 / 355629b43ad5 | code90/dev 71.603175 [#111] |
| heretic-public-base-math-mlp-c1 / incumbent-r / ea3d27ababa4 | code90/dev 73.626984 [#113] |
| current-p-native-s / candidate-s / current-s / 60520ea1e63b | code90/dev 74.242063 [#116]; code90/dev 73.269841 [#120] |
| current-s-code-priority-ac / candidate-ac / current-ac / d34fcba755ee | code90/dev 73.825397 [#119]; code90/dev 71.408730 [#122] |
| repair-ae / afa211da029e | code90/dev 72.158730 [#123] |
| native-ad / candidate-ad / current-ad / AD / 27629fd3cd4c | code90/dev 73.825397 [#124]; code90/dev 74.242063 [#127]; code90/dev 72.158730 [#130]; code90/dev 74.742063 [#141] |
| finalist-af / 496886f1f822 | code90/dev 72.658730 [#126] |
| finalist-ag / b5c80f1e73c4 | code90/dev 73.408730 [#128] |
| library-al / e89f4534b6a6 | code90/dev 68.825397 [#129] |
| an / 24754bba84f8 | code90/dev 74.242063 [#131] |
| control / 42db468937d5 | code90/dev 72.158730 [#132]; code90/dev 73.769841 [#135] |
| ao / 60115082facd | code90/dev 71.325397 [#133] |
| ap / 734747981fc0 | code90/dev 71.742063 [#134] |
| as / 618331541796 | code90/dev 71.325397 [#136] |
| aq / da2708b1b2a9 | code90/dev 72.436508 [#137] |
| ar / bca397eacfa6 | code90/dev 74.742063 [#138] |
| AN / 10a72fc4cb4d | code90/dev 72.436508 [#139] |
| AR / 23a4f622315b | code90/dev 73.075397 [#140] |
| AW overlay / 05a65f83c07e | code90/dev 72.158730 [#142] |
| AX overlay / 453a82b66f4e | code90/dev 72.242063 [#143] |
| AV overlay / e73c7262dbf8 | code90/dev 72.932540 [#144] |
| AD empty overlay / 0c6589f6cccc | code90/dev 72.575397 [#145] |
| AV-primary-overlay / AV-repeat-overlay / e9de1bc2ba1d | code90/dev 71.742063 [#148]; code90/dev 73.075397 [#149] |
| AZ-overlay / 5676422f4331 | code90/dev 69.742063 [#150] |
| BA-overlay / 937ddbb70102 | code90/dev 73.408730 [#151] |

## Full14（code % / 全14総合 %）

full14内codeはcode90ではない。tier10ではcode26、全体778（能力746＋尤度probe32）。dev/test、tier別に保持。

| 記録モデル名 / label先頭 | 各測定 code / overall（tier/split、#run） |
|---|---|
| O 原モデル / base-10-dev- | F10/dev 84.444444 / 81.223722 [#9] |
| O 原モデル / base-10-dev- | F10/dev — / —/旧未完 [#10] |
| O 原モデル / base-10-test | F10/test 91.111111 / 85.130499 [#11] |
| O 原モデル / base-30-dev- | F30/dev 69.658730 / 78.768381 [#12] |
| O 原モデル / base-30-test | F30/test 78.083333 / 78.020991 [#13] |
| O 原モデル / base-60-dev- | F60/dev 70.421769 / 76.604069 [#14] |
| O 原モデル / base-60-test | F60/test 75.952381 / 77.417560 [#15] |
| O 原モデル / O 原モデル・観測ON / cab440f871a3 | F10/dev 87.777778 / 80.214802 [#63]; F10/dev 87.777778 / 80.463802 [#86]; F10/dev 87.777778 / 80.164156 [#95] |
| joint-purpose-balanced-small / 003dc42c0bc0 | F10/dev 87.777778 / 81.490386 [#26] |
| joint-purpose-programming-set / b01c699175e6 | F10/dev 87.777778 / 80.705754 [#31] |
| joint-purpose-chat-set / 14a9fafa06c1 | F10/dev 87.777778 / 80.476941 [#41] |
| finalnorm-plus1pct / c5be8132c454 | F10/dev 87.777778 / 81.300496 [#47]; F10/test 91.111111 / 82.360074 [#62] |
| donor-text-c1 / 3353278c3610 | F10/dev 74.444444 / 76.589114 [#51] |
| joint-programming-finalnorm-plus1pct / b626f1189623 | F10/dev 84.444444 / 80.897718 [#54] |
| finalnorm-plus0p5pct / fb494e89a52e | F10/dev 75.555556 / 79.121497 [#57] |
| donor-text-c0p25 / 44a68b0f35b8 | F10/dev 84.444444 / 81.194800 [#60] |
| 0397 / ba5e6ab526c8 | F10/dev 87.777778 / 81.919633 [#67] |
| 0485 / b9617ac7de12 | F10/dev 84.444444 / 82.163967 [#72] |
| B native-contracts / 564ae5fc1efd | F10/dev 84.444444 / 79.291973 [#87] |
| H coherent donor1/16 / 0f9775ba6a24 | F10/dev 84.444444 / 81.972528 [#94] |
| P 公開base・元protocol / P 公開改変base・保持確認 / P・Nの同時対照 / heretic-public-base / 3dfb3bd1db0a | F10/dev 87.777778 / 78.475099 [#110] |
| heretic-public-base-math-mlp-c1 / incumbent-r / ea3d27ababa4 | F10/dev 84.444444 / 76.382319 [#115] |
| current-p-native-s / candidate-s / current-s / 60520ea1e63b | F10/dev 84.444444 / 78.129386 [#118] |
| current-s-code-priority-ac / candidate-ac / current-ac / d34fcba755ee | F10/dev 87.777778 / 78.934942 [#121] |
| native-ad / candidate-ad / current-ad / AD / 27629fd3cd4c | F10/dev 87.777778 / 77.576379 [#125]; F10/dev 87.777778 / 77.209316 [#146] |
| AV / 5d056eb03f95 | F10/dev 87.777778 / 77.154948 [#147] |

## DeepSWE（別台帳）

113課題、各1試行、32並列。元R1/R2の2実行はincompleteで点数不明。以下3件と合わせて5実行。

| モデル | 公式成功／採点 | 未採点 | agent steps | agent記録API calls |
|---|---:|---:|---:|---:|
| 元R3 | 1/113 | 0 | 8,274 | 8,306 |
| 凍結0397 | 1/113 | 0 | 8,071 | 8,098 |
| 旧finalnorm＋1%・ユーザー停止 | 0/109 | 4/113 | 7,844 | 未集計 |

元と0397は双方pass@1=0.88495575%。新成功1・旧成功1の喪失で同点。
旧停止runを0/113完走と表示しない。API列はagentの記録counterで、独立HTTP telemetryではない。
0397ではF2P625→473、P2P226,371→78,067の退行もあり、同点のみをもって改善としない。



## 全151実行（正確なrun IDと状態）

| # | Run ID | protocol / split | 状態 | 採点 / 予定 | code % | full14 overall % |
|---:|---|---|---|---:|---:|---:|
| 1 | wsp-calib-dev-chat-20261104 | calibration_subset / dev | partial | 6265 / 6266 | 71.073716 | — |
| 2 | wsp-calib-dev-agents-20261104 | calibration_subset / dev | completed | 30 / 30 | — | — |
| 3 | wsp-calib-test-agents-20261104 | calibration_subset / test | completed | 30 / 30 | — | — |
| 4 | wsp-calib-test-chat-20261104 | calibration_subset / test | partial | 6264 / 6266 | 74.629671 | — |
| 5 | wsp-calib-dev-swe-20261104 | calibration_subset / dev | completed | 12 / 12 | — | — |
| 6 | wsp-calib-test-swe-20261104 | calibration_subset / test | completed | 12 / 12 | — | — |
| 7 | wsp-calib-dev-agents2-20261104 | calibration_subset / dev | completed | 30 / 30 | — | — |
| 8 | wsp-calib-test-agents2-20261104 | calibration_subset / test | partial | 29 / 30 | — | — |
| 9 | wsp-base-10-dev-1-20261104 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 81.223722 |
| 10 | wsp-base-10-dev-2-20261104 | full14_tier10 / dev | running | 不明 / 778 | — | — |
| 11 | wsp-base-10-test-1-20261104 | full14_tier10 / test | completed | 778 / 778 | 91.111111 | 85.130499 |
| 12 | wsp-base-30-dev-1-20261104 | full14_tier30 / dev | completed | 2338 / 2338 | 69.658730 | 78.768381 |
| 13 | wsp-base-30-test-1-20261104 | full14_tier30 / test | completed | 2338 / 2338 | 78.083333 | 78.020991 |
| 14 | wsp-base-60-dev-1-20261104 | full14_tier60 / dev | completed | 4841 / 4841 | 70.421769 | 76.604069 |
| 15 | wsp-base-60-test-1-20261104 | full14_tier60 / test | completed | 4841 / 4841 | 75.952381 | 77.417560 |
| 16 | gemma4-base-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 17 | gemma4-scalar-l28-minus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 18 | gemma4-scalar-l28-plus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 19 | gemma4-scalar-l23-minus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 20 | gemma4-scalar-l23-plus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 21 | gemma4-scalar-l41-minus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 22 | gemma4-scalar-l41-plus4-dev10-screen-20261104 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 23 | gemma4-sparse-code-margin3-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 24 | gemma4-sparse-action-margin3-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 25 | gemma4-joint-purpose-balanced-small-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 26 | gemma4-joint-purpose-balanced-small-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 81.490386 |
| 27 | gemma4-joint-purpose-balanced-small-dev10-screen-20261105-r2 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 28 | gemma4-joint-purpose-balanced-small-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 50.150794 | — |
| 29 | gemma4-joint-purpose-balanced-small-dev10-observed-losses-repeat-20261105 | selected_subset_162 / dev | completed | 162 / 162 | — | — |
| 30 | gemma4-joint-purpose-programming-set-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 31 | gemma4-joint-purpose-programming-set-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 80.705754 |
| 32 | gemma4-joint-purpose-programming-set-dev10-screen-20261105-r2 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 33 | gemma4-joint-purpose-programming-set-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 55.150794 | — |
| 34 | gemma4-joint-purpose-algorithm-set-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 35 | gemma4-joint-purpose-workflow-language-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 36 | gemma4-joint-purpose-algorithm-set-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 53.365079 | — |
| 37 | gemma4-sparse-code-action6-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 38 | gemma4-joint-purpose-balanced-medium-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 39 | gemma4-joint-purpose-balanced-medium-dev10-screen-20261105-r2 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 40 | gemma4-joint-purpose-chat-set-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 41 | gemma4-joint-purpose-chat-set-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 80.476941 |
| 42 | gemma4-joint-purpose-chat-set-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 57.650794 | — |
| 43 | gemma4-finalnorm-minus1pct-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 44 | gemma4-finalnorm-plus1pct-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 45 | gemma4-finalnorm-plus1pct-dev10-screen-20261105-r2 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 46 | gemma4-finalnorm-minus1pct-dev10-screen-20261105-r2 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 47 | gemma4-finalnorm-plus1pct-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 81.300496 |
| 48 | gemma4-finalnorm-plus1pct-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 58.206349 | — |
| 49 | gemma4-donor-text-c1-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 70.000000 | — |
| 50 | gemma4-donor-text-c1-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 55.984127 | — |
| 51 | gemma4-donor-text-c1-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 74.444444 | 76.589114 |
| 52 | gemma4-joint-programming-finalnorm-plus1pct-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 53 | gemma4-joint-programming-finalnorm-plus1pct-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 57.650794 | — |
| 54 | gemma4-joint-programming-finalnorm-plus1pct-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 80.897718 |
| 55 | gemma4-finalnorm-plus0p5pct-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 56 | gemma4-finalnorm-plus0p5pct-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 55.150794 | — |
| 57 | gemma4-finalnorm-plus0p5pct-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 75.555556 | 79.121497 |
| 58 | gemma4-donor-text-c0p25-dev10-screen-20261105 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 59 | gemma4-donor-text-c0p25-dev30-hard-code-20261105 | hard_code30_dev30 / dev | completed | 30 / 30 | 50.150794 | — |
| 60 | gemma4-donor-text-c0p25-dev10-full-20261105 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 81.194800 |
| 61 | gemma4-joint-donor-quarter-finalnorm-plus1pct-dev10-core-long-20261105 | selected_subset_107 / dev | completed | 107 / 107 | 85.000000 | — |
| 62 | gemma4-frozen-c5be8132c454-test10-20261105 | full14_tier10 / test | completed | 778 / 778 | 91.111111 | 82.360074 |
| 63 | gemma4-base-observed-dev10-20261106 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 80.214802 |
| 64 | gemma4-resume-hs0020-dev10-screen-20261106 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 65 | gemma4-resume-h0397-dev10-screen-20261106 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 66 | gemma4-resume-h0397-dev30-hard-code-20261106 | hard_code30_dev30 / dev | completed | 30 / 30 | 57.650794 | — |
| 67 | gemma4-resume-h0397-dev10-full-20261106 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 81.919633 |
| 68 | gemma4-resume-h0397-dev30-lcb-repeat-20261106 | livecodebench20_dev30 / dev | completed | 20 / 20 | 75.000000 | — |
| 69 | gemma4-resume-base-dev30-lcb-matched-20261106 | livecodebench20_dev30 / dev | completed | 20 / 20 | 70.000000 | — |
| 70 | gemma4-resume-h0485-dev10-screen-20261106 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 71 | gemma4-resume-h0485-dev30-hard-code-20261106 | hard_code30_dev30 / dev | completed | 30 / 30 | 60.150794 | — |
| 72 | gemma4-resume-h0485-dev10-full-20261106 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 82.163967 |
| 73 | gemma4-resume-localgc-dev10-screen-20261106 | screen95_dev10 / dev | completed | 95 / 95 | 90.000000 | — |
| 74 | gemma4-resume-localgc-dev30-hard-code-20261106 | hard_code30_dev30 / dev | completed | 30 / 30 | 55.150794 | — |
| 75 | gemma4-resume-composed-dev30-hard-code-20261106 | hard_code30_dev30 / dev | completed | 30 / 30 | 60.150794 | — |
| 76 | gemma4-resume-composed-dev10-screen-20261106 | screen95_dev10 / dev | completed | 95 / 95 | 85.000000 | — |
| 77 | gemma4-confirmation-base-1491-20261106 | confirmation_reserve_1491 / test | partial | 1490 / 1491 | 77.428571 | — |
| 78 | gemma4-confirmation-h0397-1490-20261106 | confirmation_reserve_1490 / test | completed | 1490 / 1490 | 82.547619 | 80.311317 |
| 79 | gemma4-code90-0397-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.075397 | — |
| 80 | gemma4-code90-h0397-control-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.492063 | — |
| 81 | gemma4-code90-native-contracts-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 73.408730 | — |
| 82 | gemma4-code90-joint-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.769841 | — |
| 83 | gemma4-code90-matrix-balanced-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 67.575397 | — |
| 84 | gemma4-code90-h0397-control-c32-known64-r1-20261106 | known64_test30 / test | completed | 64 / 64 | 77.011905 | — |
| 85 | gemma4-code90-native-contracts-c32-known64-r1-20261106 | known64_test30 / test | completed | 64 / 64 | 76.476190 | — |
| 86 | gemma4-code90-original-base-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 80.463802 |
| 87 | gemma4-code90-native-contracts-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 79.291973 |
| 88 | gemma4-code90-original-base-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 72.575397 | — |
| 89 | gemma4-code90-original-base-c32-known64-r1-20261106 | known64_test30 / test | completed | 64 / 64 | 78.261905 | — |
| 90 | gemma4-code90-public-final-balanced-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 71.325397 | — |
| 91 | gemma4-code90-public-final-joint-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.492063 | — |
| 92 | gemma4-code90-original-base-c32-dev30-code-pairh-20261106 | code90_dev30 / dev | completed | 90 / 90 | 71.325397 | — |
| 93 | gemma4-code90-coherent-c00625-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 94 | gemma4-code90-coherent-c00625-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 81.972528 |
| 95 | gemma4-code90-original-base-c32-dev10-full14-pairh-20261106 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 80.164156 |
| 96 | gemma4-code90-original-base-c32-dev30-code-pairi-20261106 | code90_dev30 / dev | completed | 90 / 90 | 69.242063 | — |
| 97 | gemma4-code90-math-grpo-c1-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.908730 | — |
| 98 | gemma4-code90-original-base-c32-dev30-code-pairj-20261106 | code90_dev30 / dev | completed | 90 / 90 | 69.103175 | — |
| 99 | gemma4-code90-weighted-reasoning-c1-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 100 | gemma4-code90-native-b-plus-coherent-h-c32-dev30-code-pairk-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.908730 | — |
| 101 | gemma4-code90-original-base-c32-dev30-code-pairk-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.769841 | — |
| 102 | gemma4-code90-native-b-c32-dev30-code-pairk-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.075397 | — |
| 103 | gemma4-code90-coherent-h-c32-dev30-code-pairk-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.769841 | — |
| 104 | gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairl-20261106 | code90_dev30 / dev | completed | 90 / 90 | 67.182540 | — |
| 105 | gemma4-code90-original-base-c32-dev30-code-pairl-20261106 | code90_dev30 / dev | completed | 90 / 90 | 70.700397 | — |
| 106 | gemma4-code90-heretic-public-base-c32-dev30-code-pairm-20261106 | code90_dev30 / dev | completed | 90 / 90 | 73.825397 | — |
| 107 | gemma4-code90-heretic-public-base-k1-c1-c32-dev30-code-pairm-20261106 | code90_dev30 / dev | completed | 90 / 90 | 67.908730 | — |
| 108 | gemma4-code90-original-base-c32-dev30-code-pairm-20261106 | code90_dev30 / dev | completed | 90 / 90 | 71.186508 | — |
| 109 | gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairm-20261106 | code90_dev30 / dev | completed | 90 / 90 | 68.075397 | — |
| 110 | gemma4-code90-incumbent-p-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 78.475099 |
| 111 | gemma4-code90-heretic-public-base-k1-c00625-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 71.603175 | — |
| 112 | gemma4-code90-heretic-public-base-c32-dev30-code-pairn-20261106 | code90_dev30 / dev | completed | 90 / 90 | 74.242063 | — |
| 113 | gemma4-code90-heretic-public-base-math-mlp-c1-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 73.626984 | — |
| 114 | gemma4-code90-heretic-public-base-c32-dev30-code-pairr-20261106 | code90_dev30 / dev | completed | 90 / 90 | 73.269841 | — |
| 115 | gemma4-code90-incumbent-r-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 76.382319 |
| 116 | gemma4-code90-current-p-native-s-c32-dev30-code-20261106 | code90_dev30 / dev | completed | 90 / 90 | 74.242063 | — |
| 117 | gemma4-code90-heretic-public-base-c32-dev30-code-pairs-20261106 | code90_dev30 / dev | completed | 90 / 90 | 72.575397 | — |
| 118 | gemma4-code90-candidate-s-c32-dev10-full14-20261106 | full14_tier10 / dev | completed | 778 / 778 | 84.444444 | 78.129386 |
| 119 | gemma4-code90-current-s-code-priority-ac-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 73.825397 | — |
| 120 | gemma4-code90-current-s-c32-dev30-code-pairac-20261107 | code90_dev30 / dev | completed | 90 / 90 | 73.269841 | — |
| 121 | gemma4-code90-candidate-ac-c32-dev10-full14-20261107 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 78.934942 |
| 122 | gemma4-code90-current-ac-c32-dev30-code-pairadae-20261107 | code90_dev30 / dev | completed | 90 / 90 | 71.408730 | — |
| 123 | gemma4-code90-repair-ae-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 124 | gemma4-code90-native-ad-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 73.825397 | — |
| 125 | gemma4-code90-candidate-ad-c32-dev10-full14-20261107 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 77.576379 |
| 126 | gemma4-code90-finalist-af-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 72.658730 | — |
| 127 | gemma4-code90-current-ad-c32-dev30-code-pairafag-20261107 | code90_dev30 / dev | completed | 90 / 90 | 74.242063 | — |
| 128 | gemma4-code90-finalist-ag-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 73.408730 | — |
| 129 | gemma4-code90-library-al-c32-dev30-code-20261107 | code90_dev30 / dev | completed | 90 / 90 | 68.825397 | — |
| 130 | gemma4-code90-current-ad-c32-dev30-code-pairal-20261107 | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 131 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-an | code90_dev30 / dev | completed | 90 / 90 | 74.242063 | — |
| 132 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-control | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 133 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-ao | code90_dev30 / dev | completed | 90 / 90 | 71.325397 | — |
| 134 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-ap | code90_dev30 / dev | completed | 90 / 90 | 71.742063 | — |
| 135 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-control | code90_dev30 / dev | completed | 90 / 90 | 73.769841 | — |
| 136 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-as | code90_dev30 / dev | completed | 90 / 90 | 71.325397 | — |
| 137 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-aq | code90_dev30 / dev | completed | 90 / 90 | 72.436508 | — |
| 138 | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-ar | code90_dev30 / dev | completed | 90 / 90 | 74.742063 | — |
| 139 | code90-current-ad-an-ar-an-20261107-v1 | code90_dev30 / dev | completed | 90 / 90 | 72.436508 | — |
| 140 | code90-current-ad-an-ar-ar-20261107-v1 | code90_dev30 / dev | completed | 90 / 90 | 73.075397 | — |
| 141 | code90-current-ad-an-ar-ad-20261107-v1 | code90_dev30 / dev | completed | 90 / 90 | 74.742063 | — |
| 142 | gemma4-code90-ad-helper-qk-overlay-c32-dev30-20261107-r1-aw | code90_dev30 / dev | completed | 90 / 90 | 72.158730 | — |
| 143 | gemma4-code90-ad-helper-qk-overlay-c32-dev30-20261107-r1-ax | code90_dev30 / dev | completed | 90 / 90 | 72.242063 | — |
| 144 | gemma4-code90-ad-helper-qk-overlay-c32-dev30-20261107-r1-av | code90_dev30 / dev | completed | 90 / 90 | 72.932540 | — |
| 145 | gemma4-code90-ad-helper-qk-overlay-c32-dev30-20261107-r1-control | code90_dev30 / dev | completed | 90 / 90 | 72.575397 | — |
| 146 | gemma4-ad-av-ad-c32-dev10-full14-20261107-v1 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 77.209316 |
| 147 | gemma4-ad-av-av-c32-dev10-full14-20261107-v1 | full14_tier10 / dev | completed | 778 / 778 | 87.777778 | 77.154948 |
| 148 | gemma4-code90-av-az-ba-overlay-c32-dev30-20261107-v1-r1-control | code90_dev30 / dev | completed | 90 / 90 | 71.742063 | — |
| 149 | gemma4-code90-av-az-ba-overlay-c32-dev30-20261107-v1-r1-control-repeat | code90_dev30 / dev | completed | 90 / 90 | 73.075397 | — |
| 150 | gemma4-code90-av-az-ba-overlay-c32-dev30-20261107-v1-r1-az | code90_dev30 / dev | completed | 90 / 90 | 69.742063 | — |
| 151 | gemma4-code90-av-az-ba-overlay-c32-dev30-20261107-v1-r1-ba | code90_dev30 / dev | completed | 90 / 90 | 73.408730 | — |

既存catalogに全source分母・未丸め値・run/report SHA/statを保持。生成物・問題本文・回答本文は読み取っていない。未実施モデルの点数は作らない。


## Current parent: highest recorded code90 score

The user's explicit policy now selects the highest measured score on the fixed four-source official code90 cohort, superseding the earlier local-pair gain rule. The151-run ledger contains59 code90 runs. Saved AD's74.742063% is the highest ordinary code90 score; the AR logical overlay ties that value, while separately saved AR scores73.075397%. Different cohorts (screen95, hard30, known64, full14 code26) are not ranked together.

The working-parent metadata was restored to saved AD, tree27629fd3, with no weight or service change. AV's prior accepted state and all AV-anchored AZ/BA measurements remain historical evidence. BA materialization and the prepared AV/BA full14 comparison are HELD: no copy, deployment, or full14 launch occurred. The old AV-pinned templates remain unchanged and cannot validate against the new parent state.

Current AD state: b70fe5e9d519c7b4459af26f505471649c3313483058558e0796be3dad64440f. Policy: working-incumbent-policy-highest-code-v2.json eaf2112552ba45b26c913756ee128111848b64b7184afd74da3ae425886e85de. Decision: current-ad-highest-code-policy-v1/working-decision.json a9eaa1fd499b7363a2f44f375fbcf7bd3d2732ca28c8b79129f26ee228e5bc7b. This selects the observed record; it does not prove a stable expected-score ordering or code>90.
