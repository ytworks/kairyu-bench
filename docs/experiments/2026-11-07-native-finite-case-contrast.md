# AV native answers versus retained F02 teacher-forced margins

今回のAV通常生成では、事前に大きな有限margin変化が見られた `guard_code_0_1` の公開回答は **`7`** だった。既存correct=`7`へ生の文字列で完全一致し、foil=`3`とは一致しない。10件のguardすべてでcorrectへ完全一致した。F02のこの1件のmargin増加 +6.0287399292 は、guard10件の正の増分合計の85.840974%を占める。したがって、**guard10内の正の有限margin増大の大部分が、今回AVが既に答えられたprimitive出力に集中した**という限定した見方を支持する。

これは旧solverの全目的重みやgradient寄与率を再計算した結果ではない。guard10のliteral対照を一般の生成accuracyや公式コード得点に拡張しない。native回答はAVのものだけで、F02の通常生成前後差を測定していない。文字列不一致をコードsemantic不正解と判定していない。新しいモデル呼出し・生成コード実行・候補選択は0。

## 結合と比較の範囲

[一回の分析source](scripts/fresh-authored-native-margin-join-20261107.py) と [実行receipt](records/2026-11-07-native-finite-case-join/operation-receipt.json)、[全120行の数値・回答・採点状態](records/2026-11-07-native-finite-case-join/all120-joined.json)を保存した。入力を読み取るだけで、既存観測や採点結果を変更していない。

- 原teacher-forced120件のSHAがF02 finite planと一致し、native生成のAV treeが同finite base treeと一致する。
- purpose/fold/indexの全件長とfinite frameのIDを検査し、位置を確認してからIDへ結合した。全120行を保持し、欠落・重複・追加IDは0。
- 採点済48件ではnative公開回答のSHAが実行報告のpublic_answer SHAと一致する。全合否と試験内訳をそのまま持ち込んだ。
- 元AVとrestored AVのcase marginは全120件で厳密一致する。F02差は保存されたcandidate値−base値から今回再計算した。
- 48件の固定採点段階を使う。残72件は未採点のまま保持し、後続helper16等の別結果は混ぜない。

Teacher-forced marginは **correct回答tokenの平均logprob − foil回答tokenの平均logprob**。有限計測はforced-empty-thought frameで、通常のthinking生成とは異なる。長さの違う回答もそれぞれtoken平均を取る。nativeの公開回答・実行合否と同じ尺度の得点ではない。

## 事前指定guard10件のliteral対照

比較はPythonの文字列 `==` 相当の生Unicode文字列一致だけで、strip、fence抽出、数値変換、コード実行、semantic同値判定はしない。正解reference/foil自体が同一だった場合や欠測/errorは別状態とする。今回の10件はcorrect一致10、foil一致0、双方一致0、どちらにも一致しない0、欠測等で判定不可0。固定48の実行採点状態とは別列であり、guardの実行スコアは未採点のまま。

| ID | native公開回答 | correct | foil | literal対照 | AV margin | F02 margin | Δmargin |
|---|---|---|---|---|---:|---:|---:|
| guard_code_0_0 | `0` | `0` | `1` | correct完全一致 | 17.3101444244 | 16.8163795471 | -0.4937648773 |
| guard_code_0_1 | `7` | `7` | `3` | correct完全一致 | 0.2826290131 | 6.3113689423 | +6.0287399292 |
| guard_code_0_2 | `True` | `True` | `False` | correct完全一致 | 17.3881473541 | 17.0486392975 | -0.3395080566 |
| guard_code_0_3 | `1` | `1` | `3` | correct完全一致 | 22.7108402252 | 22.7428321838 | +0.0319919586 |
| guard_code_0_4 | `1` | `1` | `2` | correct完全一致 | 17.6202735901 | 17.4818496704 | -0.1384239197 |
| guard_code_1_0 | `0` | `0` | `1` | correct完全一致 | 23.8491764069 | 24.0799407959 | +0.2307643890 |
| guard_code_1_1 | `2` | `2` | `6` | correct完全一致 | 17.0844573975 | 17.2266616821 | +0.1422042847 |
| guard_code_1_2 | `False` | `False` | `True` | correct完全一致 | 2.7725062370 | 2.3513517380 | -0.4211544991 |
| guard_code_1_3 | `3` | `3` | `8` | correct完全一致 | 11.7187032700 | 12.3081521988 | +0.5894489288 |
| guard_code_1_4 | `4` | `4` | `5` | correct完全一致 | 21.1213531494 | 20.9814510345 | -0.1399021149 |

`guard_code_0_1` は `print([3, 5, 7][-1])` の出力を問う。AV finite margin=0.282629013061523、F02=6.311368942260742、差=6.028739929199219。AV nativeは1回、finish=stop、公開回答 `7`、1.655750701秒、入力33／出力208 tokens。隠れたthinking内容は保存していない。

正の増分の分母はguard10件の `sum(max(Δmargin, 0))=7.023149490356445`。netの分母は同10件の `sum(Δmargin)=5.490396022796631` で、0_1の比率はそれぞれ 85.840974%、109.805193%。net比率が100%を超えるのは他guardのmargin損失があるため。公式WSP codeや全solver目的に対する割合ではない。

## 固定48件の実行合否との関係

| AV native固定48の結果 | 件数 | F02 margin増 | 減 | 同値 | 平均Δmargin | 中央値Δmargin |
|---|---:|---:|---:|---:|---:|---:|
| 合格 | 46 | 31 | 15 | 0 | +0.0048291813 | +0.0021274984 |
| 失敗 | 2 | 1 | 1 | 0 | +0.0011170208 | +0.0011170208 |

46合格・2失敗、固定試験375/384合格を維持する。契約疑義3件も除外せず3/3合格、24/24試験合格として同じ48分母に含める。合格した46件にもmargin低下が15件あり、失敗2件は増加・低下が1件ずつである。この対照だけからmargin増加をnative弱点の修復、低下をnative能力悪化と読めない。48件は同じAV通常生成の実行結果であり、F02で再生成した対応実験ではない。失敗2件しかないため統計的な予測性能や因果効果を推定しない。

- `science-library-symbolic-numeric-variable-order`：AV native実行 7/8、F02 margin差 -0.0104454756（base 0.2662452459 → 0.2557997704）。失敗詳細と元回答は全120JSONの該当行に残す。
- `science-batched-linear-system`：AV native実行 0/8、F02 margin差 +0.0126795173（base 0.1978428364 → 0.2105223536）。失敗詳細と元回答は全120JSONの該当行に残す。

全48 case別の値：

| ID | Catalog | 契約疑義 | AV実行 | 試験 | AV margin | F02 margin | Δmargin |
|---|---|---|---|---:|---:|---:|---:|
| science-weighted-moments | basic | なし | 合格 | 8/8 | 0.2539279461 | 0.2505193353 | -0.0034086108 |
| science-periodic-laplacian | basic | なし | 合格 | 8/8 | 0.0644189119 | 0.0636293888 | -0.0007895231 |
| science-ridge-intercept | basic | なし | 合格 | 8/8 | 0.1077920198 | 0.1114280224 | +0.0036360025 |
| science-orthogonal-residual | basic | なし | 合格 | 8/8 | 0.5722838640 | 0.5707843304 | -0.0014995337 |
| science-library-sparse-solve-fixed-boundary | library-train-v1 | なし | 合格 | 8/8 | 0.2191759348 | 0.2091946602 | -0.0099812746 |
| science-library-spectral-contract-linear-convolution | library-train-v1 | なし | 合格 | 8/8 | 0.1240512133 | 0.1598709822 | +0.0358197689 |
| science-cumulative-trapezoid | basic | あり | 合格 | 8/8 | 0.1923300028 | 0.1922479868 | -0.0000820160 |
| science-row-softmax | basic | なし | 合格 | 8/8 | 0.2664678097 | 0.2683767080 | +0.0019088984 |
| science-markov-row-state | basic | なし | 合格 | 8/8 | 0.5273994207 | 0.5282544494 | +0.0008550286 |
| science-weighted-density | basic | なし | 合格 | 8/8 | -0.0221706629 | -0.0192527175 | +0.0029179454 |
| science-library-sparse-solve-banded-factor | library-train-v1 | なし | 合格 | 8/8 | 0.1561560631 | 0.1591801643 | +0.0030241013 |
| science-library-spectral-contract-periodic-correlation | library-train-v1 | なし | 合格 | 8/8 | 0.5226790905 | 0.5648764372 | +0.0421973467 |
| science-sample-covariance | basic | なし | 合格 | 8/8 | 0.1366347075 | 0.1485130191 | +0.0118783116 |
| science-implicit-decay | basic | なし | 合格 | 8/8 | -0.1558914185 | -0.1554495096 | +0.0004419088 |
| science-velocity-verlet | basic | なし | 合格 | 8/8 | 0.2918419838 | 0.2932338715 | +0.0013918877 |
| science-piecewise-linear | basic | なし | 合格 | 8/8 | 0.1815790534 | 0.1890279651 | +0.0074489117 |
| science-library-symbolic-numeric-variable-order | library-train-v1 | なし | 失敗 | 7/8 | 0.2662452459 | 0.2557997704 | -0.0104454756 |
| science-library-optimizer-feasibility-simplex-equality | library-train-v1 | なし | 合格 | 8/8 | 0.3138226271 | 0.3152047396 | +0.0013821125 |
| science-conservative-flux | basic | なし | 合格 | 8/8 | 0.3038215637 | 0.2956281304 | -0.0081934333 |
| science-merge-population-moments | basic | なし | 合格 | 8/8 | 0.2849801779 | 0.3304241896 | +0.0454440117 |
| science-segment-means | basic | なし | 合格 | 8/8 | 0.2211547494 | 0.2162767053 | -0.0048780441 |
| science-batched-linear-system | basic | なし | 失敗 | 0/8 | 0.1978428364 | 0.2105223536 | +0.0126795173 |
| science-library-symbolic-numeric-chain-coordinates | library-train-v1 | なし | 合格 | 8/8 | 0.2433887720 | 0.2451809645 | +0.0017921925 |
| science-library-optimizer-feasibility-weighted-hyperplane | library-train-v1 | なし | 合格 | 8/8 | 0.4477108717 | 0.4373048544 | -0.0104060173 |
| science-advanced-reset-running-stats | advanced-v2 | あり | 合格 | 8/8 | 0.5691466331 | 0.5741366148 | +0.0049899817 |
| science-advanced-capped-reservoir-chain | advanced-v2 | あり | 合格 | 8/8 | 0.3850889802 | 0.3892673850 | +0.0041784048 |
| science-library-residual-jacobian-decay-fit | library-train-v1 | なし | 合格 | 8/8 | 0.2845859528 | 0.2772049904 | -0.0073809624 |
| science-library-root-contract-implicit-sensitivity | library-train-v1 | なし | 合格 | 8/8 | 0.2646266222 | 0.2629218102 | -0.0017048120 |
| science-advanced-affine-state-history | advanced-v2 | なし | 合格 | 8/8 | 0.3520832062 | 0.3495942950 | -0.0024889112 |
| science-advanced-projected-gradient-chain | advanced-v2 | なし | 合格 | 8/8 | 0.5229636431 | 0.5265444517 | +0.0035808086 |
| science-library-residual-jacobian-weighted-affine | library-train-v1 | なし | 合格 | 8/8 | 0.1110529900 | 0.0928955078 | -0.0181574821 |
| science-library-root-contract-log-coordinate | library-train-v1 | なし | 合格 | 8/8 | 0.1894167662 | 0.2011309862 | +0.0117142200 |
| science-advanced-merge-event-multiplicity | advanced-v2 | なし | 合格 | 8/8 | 0.2968571782 | 0.3064618111 | +0.0096046329 |
| science-advanced-budget-prefix-cardinality | advanced-v2 | なし | 合格 | 8/8 | 0.4169191122 | 0.4233696461 | +0.0064505339 |
| science-advanced-lexicographic-pair-ranking | advanced-v2 | なし | 合格 | 8/8 | 0.2172504663 | 0.2184482813 | +0.0011978149 |
| science-advanced-weighted-lattice-cardinality | advanced-v2 | なし | 合格 | 8/8 | 0.1906869411 | 0.1946473122 | +0.0039603710 |
| science-advanced-gaussian-pair-energy-force | advanced-v2 | なし | 合格 | 8/8 | 0.0545735359 | 0.0521451235 | -0.0024284124 |
| science-advanced-reflecting-wall-flight | advanced-v2 | なし | 合格 | 8/8 | 0.2781717777 | 0.2644641399 | -0.0137076378 |
| science-library-quadrature-callback-piecewise-moments | library-train-v1 | なし | 合格 | 8/8 | 0.3964681625 | 0.3969725370 | +0.0005043745 |
| science-advanced-periodic-kick-drift | advanced-v2 | なし | 合格 | 8/8 | 0.2215581536 | 0.2239042521 | +0.0023460984 |
| science-advanced-inelastic-two-body | advanced-v2 | なし | 合格 | 8/8 | 0.0779025555 | 0.1064974070 | +0.0285948515 |
| science-library-quadrature-callback-domain-transform | library-train-v1 | なし | 合格 | 8/8 | -0.1251834631 | -0.1169936657 | +0.0081897974 |
| science-advanced-injected-metropolis-chain | advanced-v2 | なし | 合格 | 8/8 | 0.1279491186 | 0.1368168592 | +0.0088677406 |
| science-advanced-piecewise-linear-density-cdf | advanced-v2 | なし | 合格 | 8/8 | 0.2328135967 | 0.2316700220 | -0.0011435747 |
| science-library-distribution-interface-batched-logdensity | library-train-v1 | なし | 合格 | 8/8 | 0.6648713350 | 0.6864989996 | +0.0216276646 |
| science-advanced-categorical-without-replacement | advanced-v2 | なし | 合格 | 8/8 | 0.0782041550 | 0.0859493017 | +0.0077451468 |
| science-advanced-log-survival-event-masses | advanced-v2 | なし | 合格 | 8/8 | 0.2440704107 | 0.2633422613 | +0.0192718506 |
| science-library-distribution-interface-lognormal-moments | library-train-v1 | なし | 合格 | 8/8 | 0.3287643194 | 0.3341941833 | +0.0054298639 |

## 再現と限界

実行はCPU上の保存JSON集計1回だけ。ruff検査成功。計測/回答/採点の再実行はなく、sourceと全入力SHAを結果JSONへ記録した。下記commandは実行済みの記録で、再実行して既存出力を上書きしない。

```sh
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/worktrees/kairyu-bench-weight-tools/docs/experiments/scripts/fresh-authored-native-margin-join-20261107.py --root /mnt/nvme/kairyu/wsp/gemma4-coding-20261104 --repo /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/worktrees/kairyu-bench-weight-tools --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-native-finite-case-join-v1/all120-joined.json
```

この分析で確認できるのは、今回のAV native回答に対する既知校正caseの飽和と、過去に保存したF02有限margin変化の対応だけである。新しい目標関数・重み候補は選択していない。未見能力・DeepSWE改善・公式90点達成について追加の主張はしない。

| 保存物 | SHA256 |
|---|---|
| `all120-joined.json` | `7ff7094786ee7edec4d7ac4fa12799ac870126e3cabeea46343315cedb642df0` |
| `operation-receipt.json` | `aa7485615392ecf8d92b6a13e3ec1028c2265c5b5968f7b89017f512c801f40d` |
| `fresh-authored-native-margin-join-20261107.py` | `3247d576d0d32c84016b0097ff5d54edd441637dbe2c8e074f7d328130009261` |

[正規リモート結果とPRコピーの照合](records/2026-11-07-native-finite-case-join/canonical-copy-check.json)も全5入力でSHA一致した。
