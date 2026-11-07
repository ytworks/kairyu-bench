# 2026-11-07: F02 canonical dev10 完了・独立比較

**F02 は code 87.777778 点で AV と同点、全26コード問題も各問同点。凍結 amendment に従い採用せず、AV を維持する。** 条件付き確認 pair は発動しない。90点超は未達。overall は77.208631→77.865851（+0.657220点）だが、コードの strict gain 条件を置き換えない。F02 の元 proxy advance gate は永久に **FAILED** のまま。

両 run は canonical tier10/dev、778/778 採点、未採点0、item/response error0。全778 unique ID、source/kind/tier/area/max_tokens/samples、全29 source（うち診断3）、全14領域を current frozen manifest と独立照合した。元結果を変更せず、既存 audit_run と strict compare(partial=False) を使った。[全778対応表](evidence-2026-11-07-f02/official-v3-all-778-paired-items.json)、[比較JSON](evidence-2026-11-07-f02/official-v3-comparison.json)、[input hash・runtime・使用量](evidence-2026-11-07-f02/official-v3-summary.json) を保存。

- AV run: `gemma4-fresh-av-observed-dev10-20261107-v1`、tree `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
- F02 run: `gemma4-fresh-f02-observed-dev10-20261107-v1`、tree `46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c`.
- Manifest: `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`.

## 全14領域

各領域は source の等重み平均、overall は14領域の等重み平均。表は表示丸め、判定は JSON の全 precision。

| Area | AV 点 | F02 点 | Δ point |
|---|---:|---:|---:|
| knowledge | 88.333333 | 91.666667 | +3.333333 |
| expert | 87.500000 | 87.500000 | +0.000000 |
| facts | 75.000000 | 75.000000 | +0.000000 |
| math | 98.750000 | 95.625000 | -3.125000 |
| code | 87.777778 | 87.777778 | +0.000000 |
| instructions | 85.416667 | 95.416667 | +10.000000 |
| tools | 82.500000 | 77.500000 | -5.000000 |
| agents | 50.000000 | 50.000000 | +0.000000 |
| long-context | 66.770525 | 73.020525 | +6.250000 |
| reasoning | 76.304348 | 74.130435 | -2.173913 |
| commonsense | 63.666667 | 64.000000 | +0.333333 |
| multilingual | 82.095960 | 85.151515 | +3.055556 |
| vision | 81.250000 | 75.000000 | -6.250000 |
| safety | 55.555556 | 58.333333 | +2.777778 |

最大領域失点は vision −6.25点（5超〜10以下）、overall は非減少。しかし code delta=0 のため frozen rule の最初の条件で **no advancement**。同点を provisional improvement に読み替えない。overall の既存 stratified paired-item bootstrap 95%区間は -1.458420〜+2.805512点で、0を含む。これは item 再標本化であり、独立 run variability や因果的改善を実測した区間ではない。

能力746問の個別 score は改善16、低下11、同点719。診断32問は別の NLL 単位で全対応表に保持。HMMT問題6は1.0→0.5の部分点低下で、compare の0.5跨ぎ flips だけでは取りこぼすため、全 item delta を別集計した。

## 全 source

能力 source は0–100点、probe 3行は nat/token（低い方が良い）。probe を code 平均や overall に混ぜない。

| Source | 問数 | 単位 | AV | F02 | Δ |
|---|---:|---|---:|---:|---:|
| aime-2026 | 4 | point | 100.000000 | 100.000000 | +0.000000 |
| arc-challenge | 100 | point | 58.000000 | 57.000000 | -1.000000 |
| bbeh | 23 | point | 82.608696 | 78.260870 | -4.347826 |
| bfcl | 40 | point | 82.500000 | 77.500000 | -5.000000 |
| gpqa-diamond | 16 | point | 87.500000 | 87.500000 | +0.000000 |
| gsm8k-platinum | 16 | point | 100.000000 | 100.000000 | +0.000000 |
| hellaswag | 100 | point | 64.000000 | 65.000000 | +1.000000 |
| hmmt-feb-2026 | 4 | point | 100.000000 | 87.500000 | -12.500000 |
| humaneval-plus | 10 | point | 100.000000 | 100.000000 | +0.000000 |
| ifbench | 15 | point | 73.333333 | 93.333333 | +20.000000 |
| ifeval | 40 | point | 97.500000 | 97.500000 | +0.000000 |
| jamc-qa | 24 | point | 70.833333 | 75.000000 | +4.166667 |
| livecodebench | 6 | point | 83.333333 | 83.333333 | +0.000000 |
| math-500 | 20 | point | 95.000000 | 95.000000 | +0.000000 |
| mbpp-plus | 10 | point | 80.000000 | 80.000000 | +0.000000 |
| mgsm | 22 | point | 95.454545 | 95.454545 | +0.000000 |
| mmlu-pro | 60 | point | 88.333333 | 91.666667 | +3.333333 |
| mmlu-prox-ja | 20 | point | 80.000000 | 85.000000 | +5.000000 |
| mmmu-pro | 16 | point | 81.250000 | 75.000000 | -6.250000 |
| mrcr-v2 | 4 | point | 58.541049 | 58.541049 | +0.000000 |
| musr | 10 | point | 70.000000 | 70.000000 | +0.000000 |
| probe-code | 12 | nat/token | 4.087639 | 4.155552 | +0.067913 |
| probe-wikipedia-ja | 12 | nat/token | 6.688459 | 6.688325 | -0.000134 |
| probe-wikitext | 8 | nat/token | 7.477861 | 7.480259 | +0.002398 |
| ruler | 8 | point | 75.000000 | 87.500000 | +12.500000 |
| terminal-bench | 2 | point | 50.000000 | 50.000000 | +0.000000 |
| triviaqa | 40 | point | 75.000000 | 75.000000 | +0.000000 |
| winogrande | 100 | point | 69.000000 | 70.000000 | +1.000000 |
| xstest | 36 | point | 55.555556 | 58.333333 | +2.777778 |

コード平均は HumanEval+ 10/10、MBPP+ 8/10、LiveCodeBench 5/6 の等 source 平均。全26問の得点が一致し、失敗は両モデルとも LiveCodeBench arc191_c と MBPP410/576。MBPPの2問は base tests pass / plus tests fail。診断 probe-code NLL は4.087639→4.155552（+0.067913、悪化）で、著者 proxy の大幅 margin 増を公式コード改善と換算できないことが実測でも確認された。ただし一回の NLL 差だけで因果や一般化を断定しない。

## コード全26問

| Item ID | AV | F02 | Δ |
|---|---:|---:|---:|
| humaneval-plus:HumanEval/116 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/144 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/148 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/155 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/21 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/22 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/30 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/49 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/57 | 1.000000 | 1.000000 | +0.000000 |
| humaneval-plus:HumanEval/90 | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:abc393_d | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:abc394_b | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:abc395_f | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:abc397_b | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:abc399_c | 1.000000 | 1.000000 | +0.000000 |
| livecodebench:arc191_c | 0.000000 | 0.000000 | +0.000000 |
| mbpp-plus:Mbpp/256 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/271 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/410 | 0.000000 | 0.000000 | +0.000000 |
| mbpp-plus:Mbpp/567 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/576 | 0.000000 | 0.000000 | +0.000000 |
| mbpp-plus:Mbpp/612 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/63 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/791 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/792 | 1.000000 | 1.000000 | +0.000000 |
| mbpp-plus:Mbpp/801 | 1.000000 | 1.000000 | +0.000000 |

## 変化した全能力問題

同点も含む全778行は上記JSONに保持。以下は読解用に変化行をすべて表示する。

| Item ID | AV | F02 | Δ |
|---|---:|---:|---:|
| arc-challenge:MCAS_2004_8_23 | 0.000000 | 1.000000 | +1.000000 |
| arc-challenge:Mercury_SC_400853 | 1.000000 | 0.000000 | -1.000000 |
| arc-challenge:TIMSS_1995_8_J1 | 1.000000 | 0.000000 | -1.000000 |
| bbeh:disambiguation_qa:119 | 1.000000 | 0.000000 | -1.000000 |
| bfcl:live_irrelevance_287-69-0 | 1.000000 | 0.000000 | -1.000000 |
| bfcl:multi_turn_miss_func_111 | 1.000000 | 0.000000 | -1.000000 |
| hellaswag:03877 | 1.000000 | 0.000000 | -1.000000 |
| hellaswag:04561 | 1.000000 | 0.000000 | -1.000000 |
| hellaswag:06279 | 0.000000 | 1.000000 | +1.000000 |
| hellaswag:08609 | 0.000000 | 1.000000 | +1.000000 |
| hellaswag:08855 | 0.000000 | 1.000000 | +1.000000 |
| hmmt-feb-2026:6 | 1.000000 | 0.500000 | -0.500000 |
| ifbench:110 | 0.000000 | 1.000000 | +1.000000 |
| ifbench:182 | 0.000000 | 1.000000 | +1.000000 |
| ifbench:259 | 0.000000 | 1.000000 | +1.000000 |
| jamc-qa:jamcqa-test-geography-00039 | 0.000000 | 1.000000 | +1.000000 |
| mmlu-pro:4953 | 0.000000 | 1.000000 | +1.000000 |
| mmlu-pro:5305 | 0.000000 | 1.000000 | +1.000000 |
| mmlu-prox-ja:4820 | 0.000000 | 1.000000 | +1.000000 |
| mmmu-pro:standard:test_Agriculture_39 | 0.000000 | 1.000000 | +1.000000 |
| mmmu-pro:standard:test_Physics_30 | 1.000000 | 0.000000 | -1.000000 |
| mmmu-pro:standard:validation_Materials_27 | 1.000000 | 0.000000 | -1.000000 |
| ruler:fwe:131072:005 | 0.000000 | 1.000000 | +1.000000 |
| winogrande:0020 | 0.000000 | 1.000000 | +1.000000 |
| winogrande:0961 | 1.000000 | 0.000000 | -1.000000 |
| winogrande:1057 | 0.000000 | 1.000000 | +1.000000 |
| xstest:449 | 0.000000 | 1.000000 | +1.000000 |

## 条件、使用量、手数

actual runner は両方 `sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`。保存 actual-runner command と wrapper command を照合し、chat/logprob は同じ単一 gemma0 endpoint・各正しい checkpoint へ固定。TP2/DP1、BF16/MTP4、native thinking、chat32/logprob8/agent1。89 observer source map、dependency map、capture runtime/limits/sampling/decode offsets、モデル metadata（chat template/generation config等）が一致し、各 checkpoint は owner init の全ファイルSHA証拠と run label が一致。cache parity は package-version／official revision marker の一致で、全 package 内容の再hashではない。

| 項目 | AV | F02 |
|---|---:|---:|
| Wall 秒（load 含む） | 1800.7 | 1461.4 |
| Load 秒 | 102.8 | 105.5 |
| chat client calls | 558 | 557 |
| logprob client calls | 3062 | 3062 |
| request-scope trace 行（公式 run） | 1589 | 1588 |
| 同 trace error / retry | 0 / 0 | 0 / 0 |
| 全 capture server HTTP200 行 | 1614 | 1608 |
| source report prompt tokens | 1389577 | 1384997 |
| source report completion tokens | 1189635 | 1146032 |
| agent input tokens（別集計） | 104663 | 66654 |
| agent output tokens（別集計） | 11355 | 11959 |
| agent episodes | 19 | 14 |
| agent tool calls | 39 | 36 |

server trace は全行 HTTP200 で、4 synthetic overhead と nested agent を含み run_id を持たない。公式 client request と同じ件数だとは扱わない。各 capture の client trace にある2 trial lifecycle 行は request から分離し、4 overhead 行も除外した。agent outcome の exception は0、windows課題の reward0 はインフラ失敗ではなく公式不合格として保持。

| Agent task | AV reward / episodes / calls / 秒 | F02 reward / episodes / calls / 秒 |
|---|---:|---:|
| terminal-bench/install-windows-3.11 | 0 / 12 / 26 / 402.496 | 0 / 9 / 24 / 268.240 |
| terminal-bench/portfolio-optimization | 1 / 7 / 13 / 454.935 | 1 / 5 / 12 / 443.964 |

AV は途中で別GPU6の有限 probe と重複し、F02 はその負荷がなかった。wall 1800.7→1461.4秒を単独の重み編集による速度改善とは扱わない。コードの初期 pair は完全同点なので、事前に条件付きで定めた追加 AV/F02 confirmation pair は実行しない。元モデルとの新条件比較、全問30/60、DeepSWE、HF公開はこの一回から推論しない。

## 観測・再現・解析失敗

この比較は run/items/responses、軽量 request traces と owner receipts だけを読み、2.9GB 級 merged activation は読み込んでいない。writer/drop/未結合/発火差とサービス復旧の最終確認は observation/deployment owner の別証拠で報告される。この文書は score 比較成功から観測完全性や復旧成功を推測しない。

[既存機能を呼ぶ専用集計 recipe](scripts/fresh-f02-official-comparison.py) と [事前方法](2026-11-07-f02-official-comparison-method.md) を使用。新規モデル呼出し0。解析v1は client trace の trial 行に operation がない KeyError で止まり、[失敗記録](evidence-2026-11-07-f02/official-v1-failure.json) と source を保存。v2は request scope を分離して完了、v3は除外内訳ラベルを訂正し全server HTTP状態を加えた。両完了解析の score/判定は一致し、[v2数値](evidence-2026-11-07-f02/official-v2-summary.json) も残す。これらは同じ一回のモデル評価に対する CPU 再解析で、追加 WSP 反復ではない。v3集計2.784秒、Ruff PASS。

最終 summary SHA `021788b5e51f768337661eccd4af9f1d8a92892ab264d134147706de8f29ae2b`。PR 数値には入力／関連source SHA、全source/領域、全問題、agent手数、使用量と失敗を含む。[著者ケース監査](2026-11-07-authored-case-audit.md) の11反例・1 token への proxy 利得集中も追加解釈証拠として保持し、公式結果や元 gate を事後修正していない。
