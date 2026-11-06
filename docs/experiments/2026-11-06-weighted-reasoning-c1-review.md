# J: weighted-reasoning donor の単一係数試行と保存重み検証（2026-11-06）

J は、repository-repair の成功履歴で学習された LoRA を係数1で元の Gemma 4 31B BF16 へ移す、固定した1試行である。短い科学関数の代理指標や math 最終答案とは異なる学習方向を調べる。**本記録は試行前の根拠・固定条件と完了した保存重み検証であり、能力改善・90超・採用を示す結果ではない。**

## 公開学習情報と移植の留保

[eexian/gemma4-31b-w4-weighted-reasoning-lora](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/f66802e3cefb6c7db4b8ff883e3c28c48d6d4093) の固定 revision は `f66802e3cefb6c7db4b8ff883e3c28c48d6d4093`。以前 main から保存した card と今回の固定版は byte hash が一致した。

- 著者の基点は `google/gemma-4-31B-it-qat-w4a16-ct@52f3f65bc7a02d555763bc923bd1d9094898219d`。INT4/group32 を frozen BF16 に復元して学習し、packed W4＋native LoRA で配信する。元の未量子化 BF16 と同一ではなく、今回の dense merge は**別基点への移植**である。
- 公開説明による raw 332 attempts / 305 tasks から、テスト成功した200 trajectoriesを利用。課題単位で180 train（2,951 decisions）と20 validation（216 decisions）へ分け、失敗・切断応答は正例に入れない。テスト成功は包括的な意味検証ではない。
- current assistant のみを教師信号とし、prompt/history/tool 出力の直接損失重み0、thinking 0.25、action/content/protocol delimiters 1。全rank・optimizer microbatchにわたる weighted CE の総和／重み総和。native Gemma template で thinking/tool channel を直列化し、tool 引数をobjectへ変換する。最長trainは19,825 tokens、準備上限24,576で黙った切断を拒否する。[固定training README](https://github.com/TheWayLost/gemmacodeagent/blob/0ab366421b93ec75697a91091f5990a0af795046/training/weighted_reasoning_lora/README.md)
- 学習loss 0.3582274は正解率ではない。zero-Bとbaseの2-prompt logit差0、trainedとの差2.5、4並列・tool経路・thinking-budgetの確認はloading効果の証拠に留まる。20 validationと別129 Kaggle tasksのrepair評価は進行中で、最終solve率やON/OFF精度比較は公表されていない。

固定dataset `d912c71174a4522199f07470dfb1b82eb17050b3` の [sources.json](https://huggingface.co/datasets/Lanzo-T-H/dpsk_untitledagent_data_lanzo/blob/d912c71174a4522199f07470dfb1b82eb17050b3/sources.json) は SWE-Gym、R2E-Gym-Subset、SWE-rebench、SWE-rebench-V2 の各revisionを列挙する。確認した説明にWSP/HE/MBPP/LCB/SciCodeの直接利用を示す記載はないが、task ID照合は未実施で、重複不在の証明にはならない。学習答案・trajectory本文は取得していない。

## 物理的な方向の確認

既存の [factor-space observer](../lora-direction-comparison.md) で、payload hash、全820 F32 factors、r16 / alpha16 / scale1、対応するtext410行列、余剰・vision factor 0、有限性を確認した。410は全60層の6 projectionと、Vを独立に持つ50層の構成に対応する。baseの意味的互換性を証明する検査ではない。

| 観測対象 | 結果 |
|---|---:|
| payload | 489,840,816 B |
| 未丸めの係数1 Δ L2 | 2.7971310164077297 |
| H（日本語agent/coder）の連続方向とのcosine | 0.0019907195081120856 |
| I（math GRPO）の連続方向とのcosine | 0.00003384561533234667 |
| MLPのΔ norm²構成比 | 76.29094849% |
| 上位10層のΔ norm²構成比 | 33.16892072% |

上位層は37、36、40、35、43、42、41、47、39、38。これらは重み空間の分布であり、機能局在・意味的独立性・組合せの加算性を表さない。表のnorm/cosineは**連続factor演算**である。保存BF16の変更数・Δ L2は下節の作成後測定で区別し、保存後の方向cosineや忠実度は未測定とする。

係数1は著者が学習したW4基点での運用点を最初の仮説に用いるもの。元BF16での最適係数とは主張せず、振幅grid、既知benchmark答案からの係数選択、追加native editは行わない。

## Opus 5.5 の意見と採用範囲

匿名の一般方法レビューは actual assistant/model usage とも `claude-opus-5-5` のみ、tools 0、53.13秒、$0.11072、return code 0。`verified_requested_model=true` / `review_usable=true`。Jをキーにしたpublic final JSONは通常のlist形式ではなく `labels_match=false` だが、単一Jのpublic-final-text経路で使用可能と確認した。raw stream・thinking blocks・環境情報は保存しない。レビューは実候補の非公開測定値や能力の承認ではない。

単一の事前固定試行は妥当との意見を受けた。W4補償成分の移植、成功例だけの学習、thinking/action重み差、長い履歴と出力cap、repairと単関数／競技コードとの分布差は留保する。normやloading検証から能力を断定せず、同条件の公式実行を判断材料にする。

次の助言はそのまま要求にしない。

- QAT基点の追加取得・全行列drift計測、別runtime logit oracle、2–3 seedの反復は任意案として時間上見送る。今回の成功要件や新しいユーザー承認待ちは追加しない。
- **historyの直接損失0は、contextを経由する勾配が無いことや、tool errorの解釈を学習できないことを意味しない。** 後続assistant損失はcontextを条件とし、そこを処理する学習可能部分に勾配を与え得る。何が学習されたかは実行結果で確認する。
- **組合せに、単体成功集合のunionを超える新成功を必須としない。** 各単体の成功を一つのモデルで回復してbest soloを上回れば、累積改善として正当である。同じ問題を二重計上せず、original・best solo・各成分と比較し、維持した成功・新成功・失敗を示す。低cosineは加算性の根拠ではない。

## 固定条件と完了した保存重み検証

metadata previewは1.3269秒でPASS。固定係数1を09:12:35 UTCにCPU作成開始し、
**416.22秒で作成、294.89秒で独立CPU verifyを完了**した。完了時刻はそれぞれ
09:19:32 / 09:24:54 UTC。数値実装は既存のF32 factor-format v2と同じkernelを再利用し、
F32 r16 / alpha16 / scale1を保って元BF16から1回だけ最終BF16丸めを行った。
af9固定image、runc、UID1003、CPU0–7、BLAS4、NumPy2.2.6、network noneのCPU処理であり、
Jのための新しいtool codeやkernel変更はない。

既存のCPU verify→freeze bridgeを1回実行し、**全410行列のBF16再生成一致、
元／候補の各12ファイル全hash、非対象span、前後inventory/stat継続性**が合格した。
変更は5,880,308,266値 / 410行列、保存後Δ L2は **2.576101233886092**。
manifest SHAは `75a985b8227e7270e52a80c6c0b5c269500aa9bda81d33157986590e8e27054d`、
candidate treeは `dbf0a54c2788a3d7b529cb925a712b037bf457d392b6346a61c7e32de57c4005`。
freeze SHAは `9e3b41c69e750c45423de6e7b0eecd38bfb980f4f4add7ec021fe106c3f3409c`。
元treeは `cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9` のまま。
新コピー後のNVMe空き321.59 GiBは、この試行の明示reserve300 GiBを維持した。

480 CPU tests（428 pass / 52 optional skip）、Ruff、mypyの成功済み証拠と同じ数値sourceを照合し、
J固有のusable Opus 5.5 receiptを検証に束縛した。dense sourceは
`58462f8f45373325b23bf845ff53af007c75eb30691fa3b3f5299bfde3fd6cfe`、
compiled kernelは `7796c76da941982bf442de539883e58436d101a64929d2488f2817f0da0af18f`。
実行例外は0。これは**保存bytesのCPU検証**であり、RAM/GPU logitsや能力の同一性ではない。
[Iでの同実装検証](2026-11-06-math-grpo-c1-review.md) と同様、新配信・smoke・WSPの成績はこの検証から主張しない。

### H・Iとの保存後の変更量

既存manifestと係数を合わせたfactor報告だけを結合した。新しい31B payload読取やGEMMは0。
共通の元tensor SHA・shapeを全410で照合し、分母は対象text410の
29,286,727,680要素・元L2 **2651.4172292228955** に限定する。

| 保存候補 | 変更要素率 | 保存Δ L2 | 対象元重みL2比 | 保存／連続norm比 | MLPの保存Δ norm²比 |
|---|---:|---:|---:|---:|---:|
| H（ja-agent、1/16） | 14.7805% | 1.8885375 | 0.0712275% | 0.885533 | 82.5373% |
| I（math、1） | 27.2576% | 3.0345490 | 0.1144501% | 0.909158 | 67.1824% |
| J（weighted-reasoning、1） | 20.0784% | 2.5761012 | 0.0971594% | 0.920980 | 77.7757% |

Jの保存Δ norm²上位10層は37、36、40、35、43、42、41、38、39、47で、全体の36.9377%。
module単位の最大相対L2はL37 gateの0.200388%、続いてL40 gate 0.185867%、L36 gate 0.181607%。
層・family別の全集計を保存しているが、局在した機能や編集対象の選択を意味しない。
**保存／連続norm比は大きさの比であり、方向保持率・忠実度・丸め誤差norm・能力保持率ではない。**
分子はFP32演算後の実BF16差分、分母はFP64 factor-Gramの連続方向であり、保存後cosineは未測定である。

## 配信後の固定評価計画

固定smokeは自作advanced-v2のtrain16＋transfer8、native thinking ON、既存official sampling/tokenizer/template、seed0、max_tokens32768。historical original-v2の23/24は構造的な破損検出の参照で、fresh pairedな因果比較ではない。転移8件はgradientに使わないが、既に既知のvalidationである。既定の停止条件はoriginal比でformat/syntax/missing/execution-unknownが2件以上増える、またはcapに達しusable finalが無い場合。semantic pass率で係数調整しない。thought channel開閉や独立reasoning token数は既存generatorで未計測。

その後の公式評価はfresh originalとのdev30 code90同時比較を準備済み。HE30 / MBPP30 / LCB20 / SciCode10、各direct TP2 endpoint、chat32 / logprob8 / agent1、固定bench source `5e3f2df0c7f455013af3fa2d7abb3719858f808f`、同manifest・native ON・公式sampling/caps/scoringを維持する。dev90はknown64の90超確認や全14領域評価と同一ではない。code各source、課題別得失、形式・cap・時間・不確実性を報告し、少量の他領域損失を自動棄却条件にしない。

先行balanced候補は公式dev90で71.3254対original72.5754となり不採用。容量確保のため、そのcheckpointディレクトリ12file（62,578,686,256 B、約58.28 GiB）のみを退役した。manifest、spec、benchmark結果は保持。cleanup receiptはmount参照0と削除後不在を記録し、空き容量は345,306,976,256→407,885,709,312 B。Jの作成・配信状態とは別の記録である。

## 追加公開候補3件の限定探索

検索一覧で見つけた3件だけを、固定revisionのAPI/card/config計9 HTTPで追跡した。payload取得0。いずれも強いadapter固有のcoding/math/agent評価根拠は確認できず、実測失敗とも判定していない。

| 候補 | 確認できた制約 |
|---|---|
| [livadies / c805fabb](https://huggingface.co/livadies/gemma-4-31B-Opus-Thinking-Adapter/tree/c805fabb3785b376f8a50b19fba821088922e3cd) | 別NF4 base、r8/alpha16、128-token切断学習。長いthinkが途切れるとの著者注意、評価成績なし |
| [win10 K1 / 712fa280](https://huggingface.co/win10/gemma-4-31B-K1/tree/712fa2808d60465fa0bde72a69fc6affcb018ffc) | heretic改変base、r32/alpha32、KIMI-K2.5/Hunter-Alpha SFT。成績・native ON比較なし |
| [cyberenchanter SC2 / 1965f8c2](https://huggingface.co/cyberenchanter/gemma-4-31b-sc2-lora/tree/1965f8c288bcc49789fc273bdaff9d42db5c999d) | unsloth31B-it、r16/alpha16、generic SFT card。課題・dataset・成績未記載、vision/text target regex。SC2をscienceともStarCraftとも断定しない |

## 証拠の位置と固定値

以下は実験artifact rootからの相対名。公開card/config・source metadataのみを引用し、学習答案や思考本文を含めない。

| 証拠 | SHA256 |
|---|---|
| `eexian-reasoning-rationale-audit-v2/summary.json` | `50130536fe92f27ca33023b316ada680e8a9f1f0538d1b63735d94c450fea949` |
| `public-weighted-reasoning-donor-audit/physical-audit-summary.json` | `063528dbda0133ff94b2da2f37318fa62fe211571024fc0a96d824c09ae937ac` |
| donor payload | `6a9414b2c6bd0b13f524ed00a6992c57efe8dc6c423d7c4b87a4ee083661c3f7` |
| donor config | `51315a9a89a50822bef43a768984515a1393ce9b74080477d32b00747101ff16` |
| `opus-weighted-reasoning-c1-review/receipt.json` | `251af5cd1f7d83bc2e7b2dbda3719b40975e51e3f8c8f4a4e22fa48b0f1c2d22` |
| review prompt | `f1930416d0af878391d21cfb8269d88ad6fb266945779a9435aeac42b3b75ab1` |
| `weighted-reasoning-c1-v1/spec.json` | `69394c3ee3bbbd856b9591d2d516c1da955ed8327fd0aacfcad4e16b7be41e54` |
| `weighted-reasoning-c1-v1/launch-plan.json` | `c493bbe467eabdc0ff9a4820c6f17077b5abae28c6b5b8d6497952eac759d9aa` |
| `weighted-reasoning-c1-v1/create-completion.json` | `63eaaaf3f7d04ea2bb900f8fbd809b03c4728dccc2d52fd2586bfec76255d731` |
| `weighted-reasoning-c1-v1/verify-completion.json` | `095d138d59b70ebdd17b4b5a29e5ea30d22e6ea717442b8f1310244fc6cf484d` |
| `weighted-reasoning-c1-v1/freeze.json` | `9e3b41c69e750c45423de6e7b0eecd38bfb980f4f4add7ec021fe106c3f3409c` |
| `weighted-reasoning-c1-v1/saved-footprint-comparison.json` | `fb580faa12a89a292ae289b96544ca529ab61ff3cde61854cf9dc5073d7e45ef` |
| `CPU CI: cpu-integration-math-factor-v2/ci-results.json` | `944f3c82e2a6cd6ccd445db8d79e97af756cd4bbed84ce568d88bb975c97cac7` |
| `J creator manifest` | `75a985b8227e7270e52a80c6c0b5c269500aa9bda81d33157986590e8e27054d` |
| `dense_lora_recipe.py` | `58462f8f45373325b23bf845ff53af007c75eb30691fa3b3f5299bfde3fd6cfe` |
| `bf16_lora_kernel.so` | `7796c76da941982bf442de539883e58436d101a64929d2488f2817f0da0af18f` |
| `weighted-reasoning-c1-v1/native-on-smoke-plan.json` | `9ac19cec06af5c164fed6dac6fdfd2734bdf49bbb4dfc1e5a1d9cc6b7fb1be3d` |
| `weighted-reasoning-c1-v1/dev90-pair-plan.json` | `57de819fb166e91a913839a2db21f538f3b6417c90f2b6cf021767f623bb9c23` |
| `cleanup-balanced-candidate-20261106.json` | `d486423fcc8476af251a0a3ec95b3987234e714fc9c58cf1fe175b85b23b118f` |
| `additional-donor-search-v2/followup-three/summary.json` | `656be477d8a6035b01667945b5f2f91eb873697159035966137c87c711291224` |

再利用toolの範囲は [inventory](reusable-tool-inventory.md) を参照。保存BF16変更数・normは本記録の完了済みCPU検証として示した。配信後のsmoke・公式結果は、これらの重み検証と区別して追記する。

## Native ON確認と公式比較開始

固定advanced-v2の24呼出しは全てstopで終了し、24/24が実行まで進んだ。train15/16＋transfer8/8=23/24で、過去original-v2と同じ反射壁課題だけが失敗した。形式失敗・unknownは0。入力2,864 token、出力49,375 token（元参照43,599）であり、独立した思考token数ではない。これは構造上の確認で、能力改善やfreshな因果比較とはしない。

`weighted-reasoning-c1-v1/smoke-numeric-summary.json` と `structural-smoke-decision.json` に数値とSHAを保存した。新しいserving-counter toolの全489件CPU CI通過後、2026-11-06 09:29:59 UTCにJ/元の開始前counterを取得し、両方running/waiting=0を確認。直後に未変更の公式dev90をdirectTP2/c32で開始した。終了後にprocess counter差分と保存sampleの数値を区別して照合する。結果が出るまでは採用・90達成・合成効果を主張しない。
