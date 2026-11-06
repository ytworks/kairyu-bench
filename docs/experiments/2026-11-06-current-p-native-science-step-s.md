# S：採用済みPへの科学native増分と公式比較（2026-11-06）

Sはfresh Pとの公式dev90で **72.575397→74.242063、+1.666667 pp**。paired 95% CIは[−1.666667,+6.250000] pp。LCBの2問を得てMBPPの1問を失い、SciCodeは不変だった。科学校正proxyの改善を科学コード能力の改善とは読まない。Sはまだ採用せず、Pを作業親として保持し、Sの全14領域評価を実行中である。既知64問で90超、未知確認、DeepSWE/HFの達成ではない。

## 親と固定編集

親はworking-incumbent SHA 2666c54246852bf328be8b6293ff3402f011403fc27ad28113c5bcddaf923c99のP、tree 3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc。R増分撤回後の同一保存Pへ、科学numerics／contracts／stateを組にしたSを追加した。元モデルからの再開始ではない。

固定BF16 specは6,911座標／86tensor、P比ΔL2=0.00038920634195206596。45層に分散し、Δnorm²のpost-attention norm52.9131%／post-FFN norm47.0869%、pre-FFN norm2座標。input／q/k norm／layer scalar／final normは不変。小さいL2や編集位置だけで能力・因果機構を断定しない。設計、全21目的の予測と実測、編集分布は[P目的観測](2026-11-06-current-incumbent-purpose-observation.md)を参照。

CPU create203.5688秒、独立verify165.0727秒、gemma0 deployment116.8664秒を各1回。verifyは既存native editorのfresh source/candidate全file hashと元byteへの復元検査であり、410行列GEMMの再生成ではない。freeze公開時の追加payload scanなし。P1/O2/R3とembeddingのidentityは保護した。固定24自作生成を追加せず、既存finite証拠から公式比較へ進んだ。

| 保存証拠 | 値 |
|---|---|
| S tree | 60520ea1e63b25a69b10da1f39f63339221b1a0597c358d60a7cc6b0001d3be1 |
| native spec SHA256 | 168e4851418cbc209f47e50881bcf3fb235c51ce29ed50af525584a3aaf40c0d |
| manifest SHA256 | c236be7ffa3c3b465f21c6bf2e41d475f3aa81e6d561e503f5f21c0b2cf8ffe4 |
| freeze SHA256 | a978a2cb7b36c0a4ebab3e160ca7ba729e20e0d0063b18379d98ca8413db3e92 |
| completion SHA256 | 69abce68211e6bfd0430c029f9788e246e9a80301c66b238653c054a9e2a95c4 |

## 局所予測と実モデルproxy

全84線形予測制約を満たしたSでも、BF16実forwardでは18制約が局所下限を下回った。主目的correct-LLは6fold全て正、平均+.0112496、marginは5/6正、平均+.0001721。state片fold margin−.0097543、general片fold LL−.1917758、tools平均margin−.0803791の損失を保持する。instruction実平均はLL+.0900005／margin+.0369348だった。未達数は線形近似の誤差診断で、全領域無損失の採用gateではない。

比較したUは推論／指示の実proxy損失がより大きいため保留し、公式accuracy失敗とはしていない。Sのみを先行し、再scale・再solve・盲目的なS/U加算は行っていない。finiteはforced-empty-thoughtのteacher forcingで、native ON自由生成とは異なる。A/Aのmargin/NLL/sampled KL差0、96tensor／11,417 union座標のBF16 readback SHAとspecを独立照合した。KLの対象はcorrect continuation最初／最後の位置のみ。

## fresh Pとの公式dev90

両armは同一manifest、tier30/dev、HE30＋MBPP30＋LCB20＋SciCode10、direct TP2各1replica、chat concurrency32、native ON／server official sampling。archive source5e3f2dfの76files、installed source/resource82files、Python/package versions、actual image/RootFS/Image.Configが一致。HostConfig/Bindsとmountsのリスト順だけが異なり、その順序正規化後は一致した。比較は公式source等重み平均と既存paired bootstrap（2,000 resamples、seed20261104）を用いた。

| Source | 件数 | P | S | 差 pp |
|---|---:|---:|---:|---:|
| HumanEval+ | 30 | 100.000000（30/30） | 100.000000（30/30） | 0 |
| MBPP+ | 30 | 80.000000（24/30） | 76.666667（23/30） | −3.333333 |
| LiveCodeBench | 20 | 70.000000（14/20） | 80.000000（16/20） | +10.000000 |
| SciCode | 10 | 40.301587 | 40.301587 | 0 |
| **Code（4 source等重み）** | **90** | **72.575397** | **74.242063** | **+1.666667** |

得たIDはlivecodebench:3777、livecodebench:abc390_g（各0→1）。失ったIDはmbpp-plus:Mbpp/641（1→0）。他項目は不変、SciCodeの部分点も保存した。両90/90採点・未採点0・truncated/empty0。単回かつ繰返し観察したdev集合の比較なので、全般的能力改善や汎化の確定とはしない。CIは不確実性の情報であり、新たな有意差gateではない。

| 記録値 | P | S |
|---|---:|---:|
| 公式chat API calls | 119 | 119 |
| 公式prompt tokens | 93,600 | 94,512 |
| 公式completion tokens | 324,723 | 317,872 |
| report wall秒（load内数） | 357.6（8.2） | 340.9（7.9） |

endpoint process countersは両方120完了＝stop119＋length1、公式119 callsとの差はstartup probe1件に対応し、token差もinput21／output1。公式項目のtruncationではない。process startは前後不変、error/abort/repetition増分0、前後idle。ただしcounterだけで排他的利用、per-item HTTP retry、未観測reset、thought token数は証明しない。wall差を重みだけによる速度改善としない。

Run IDs:
- P: gemma4-code90-heretic-public-base-c32-dev30-code-pairs-20261106
- S: gemma4-code90-current-p-native-s-c32-dev30-code-20261106

## 全14領域確認と次方法

S full14はgemma4-code90-candidate-s-c32-dev10-full14-20261106、公式開始15:26:48.295975 UTC、tier10/dev778項目。15:28:29 UTCの台帳snapshotではrunning、reportなし、score未確定。codeは26項目でSciCodeを含まず、dev90と点数を直接比較しない。P保持確認との関連領域得失を見てrootが判断する。S採用はまだ行っていない。

方法Zは実測した少数の有限応答を使う合成と、実BF16 deltaのGram費用を組み合わせる案。匿名公開一般方法だけをOpus5.5へ1回送り、actual assistant/model usageとも5.5のみ、labelZ／tools0／fallbackなし、49.1768秒・$0.108556で検証済み。負係数・打消し・非線形性・親変更・一度のBF16丸めを別々に扱う助言を記録した。係数boxは測定endpointのconvex hullと同義でなく、Gram費用は丸め後の実normとも同義でない。追加half-scaleや生成思考context校正は任意助言で、必須化・実行していない。Zのsolve／spec／候補／GPU0。Sの結果と採用親が確定するまで進めない。

[全公式スコア台帳](2026-11-06-all-official-scores.md)へ本ペアを追加した。

## 数値証拠

以下はanalysis/code90-resume-20261106/からの相対パス。回答・問題本文・hidden reasoningはこの記録に使用していない。

| Artifact | SHA256 |
|---|---|
| incumbent-p-science-coupled-s-v1/dev90-final-summary.json | d34170ae2179b6e28b1d7a955cdafa2c12f65969d3917828b130e610f3c564e9 |
| 同dev90-safe-evidence.json | fa04a4581fe98ea3a107a3f0e4c1530d74825a94c47941959adf03b24d3ac7fb |
| 同dev90-pair-plan.json | 189d4862e19942cd875d0c44565d38e569e58b77cbc555d29cb421133f28dca2 |
| 同dev90-runner-parity.json | 8850e6f1186dd1c168f2bab02ef3fab7bea6613cd783260fc4805828b8dcd3d9 |
| 同dev90-runner-mount-order-proof.json | 6e8463f4d697e6b31787b3dd4e442ddefd7fd0f615f68333459748cb2b0a2eb2 |
| current-p-native-su-finite-v1/independent-numeric-review.json | 9729a7cd951aeffc23044f70595535977345afbbe88560acd4911ea8a72895f2 |
| candidate-s-retention-full14-v1/plan.json | 5002e29c857fceb95d294f3a88168204a1130c5bebf81aba562f7cc460fea584 |
| 同standing-root-decision.json | d8802a7886a6db7dd481276f6ccbcff6416c05aa8723276cdd5155e2eb4fcc5f |
| opus-finite-response-composition-z-review/receipt.json | 3043a6200029043e7c98c5f2e7e64b355baae0d990fc657200a81cc4c1e50c94 |
| 同method-summary.json | 9a4e9a6df9d654dc94140b7e8be4ba753262a8adbf223c18d68f99a237c3f04c |
