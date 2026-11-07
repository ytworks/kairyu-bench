# AV behavior12 native感度の実測結果

**12件×2目的を各1回、全421 native tensor/1,331,516座標で完了した。新候補・solve・checkpoint編集は0。** 元gemma-2のexact CID `fd4ec2eb…` はhealthy/queue0へ復帰し、他26containerは不変。owner PID841081は終了しglobal flockは解放済み。

[完全closureと全raw hash](records/2026-11-07-native-gpu4-owner/measurement-v2/measurement-closure.json) SHA `e889b331d80e724c4313cda5f9cec672d3f1752364e268e4c9b415d12eb897c3`。承認plan SHA `5783f8bce04355bcc57d8c2648659e92c9c107c71de9d01c20b0e39dde6fa732`、owner source SHA `9a8d9255312a97ae2762ec97c34c9701ac5d64dcb0dc273e01f8d7f6700b7183`。

| 目的 | 完了case | forward | collector秒 | process秒（load含む） | peak allocated GiB | peak reserved GiB | CPU退避累積byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct-logprob | 12/12 | 12 | 29.386511 | 51.437405 | 62.256745 | 62.697266 | 65,238,857,728 |
| margin | 12/12 | 24 | 47.426942 | 68.458518 | 62.256745 | 63.066406 | 130,603,544,576 |

owner全体は **237.897175秒**（stop・2回のload・計測・exact復帰を含む）。各objectiveのCPU→GPU unpack累積byteは上表の退避量と同じ。値は重複save/unpackを含む累積転送量で、unique活性量やhost RAM peakではない。host peakは未測定。GPU4だけを露出し、GPU5は計測に使わなかった。失敗file0、retry0、切詰め/backend fallback0。両summaryはsource_values_unchanged=true、全12frame一致、unused tensor0。

| case | correct token平均LL | correct−foil margin | 先行native生成の数値fixture |
|---|---:|---:|---:|
| behavior-A1 | -1.373584032 | -0.260095477 | 0/16 |
| behavior-A2 | -1.647462606 | 0.228590012 | 16/16 |
| behavior-B1 | -1.720735431 | 0.116192222 | 16/16 |
| behavior-B2 | -1.107952237 | 0.045379162 | 16/16 |
| behavior-C1 | -2.427544832 | 0.633769274 | 16/16 |
| behavior-C2 | -1.277695298 | 0.416755795 | 16/16 |
| behavior-D1 | -1.511904955 | 0.286189437 | 16/16 |
| behavior-D2 | -1.106694221 | 0.453084230 | 16/16 |
| behavior-E1 | -1.086647868 | 0.040020227 | 16/16 |
| behavior-E2 | -1.854624033 | 0.208760023 | 16/16 |
| behavior-F1 | -2.109413147 | 0.557504654 | 16/16 |
| behavior-F2 | -1.881312728 | 0.536499143 | 4/16 |

先行native生成は10/12件、164/192trial合格、unknown0。Bの方法要件は未検証であり、この表で一般のcoding正答率を主張しない。A1は負marginでnative0/16、F2は正marginでもnative4/16だった。F2の実際の失敗は開始index off-by-oneで事前foilとは異なる。LL/marginはforced-empty-thoughtのteacher forcingであり、native思考/自由生成の改善量ではない。

全12値を保持した上で、case別の共同感度・保存BF16刻み・各tensor/familyの実差分を調べる段階である。native全体の大きなnormだけを分母にして編集が小さいと判断しない。現在は候補順位、変更範囲、採否を選択していない。

生のgradient/source値npzはcanonical `analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2/work/{correct-logprob,margin}` に保持し、全file SHA/sizeをclosureへ収録した。JSON、owner receipts、stdout/stderr、monitor記録は同PRのmeasurement-v2へコピーした。[読み取りclosure操作](records/2026-11-07-native-gpu4-owner/close-measurement-v2.py)と[実行receipt](records/2026-11-07-native-gpu4-owner/close-measurement-v2-operation-receipt.json)も保存。観測後の分析担当へexact復帰・lock解放・全入力pinを引き渡した。
