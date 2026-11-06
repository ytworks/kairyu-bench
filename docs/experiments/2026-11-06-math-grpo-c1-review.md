# Math GRPO donor I：係数1のレビュー・検証・公式dev90（2026-11-06）

Iは公開math donorを固定係数1で元BF16重みにmergeし、通常のnative thinking ONで評価する実験である。
**CPU作成・全410行列の再生成検証・自作24例の構造確認・公式dev90対照評価が完了した**。fresh original69.242063%→I70.908730%（+1.666667pp、paired95% CI [0,+4.166667]pp）で、HE151/95の2件を獲得し、他88件は同点だった。下限は0であり、安定した改善やコード90達成とは主張しない。
重み幾何や作者のmath成績を、WSP code90達成の証拠として扱わない。

## 公開入力と適用範囲

[固定revisionの公開card](https://huggingface.co/aestudio/gemma-4-31B-it-GRPO-math-lora/blob/8ec9097a26697380f9e1546fdc1e485f23bd3b61/README.md)
は、DeepMathでGRPOを行い、選別済み1,000問題で元0.629→0.758を報告している。
これは作者のthinking OFF・n=4・特定sampling/採点条件での報告であり、
全DeepMath、コード、native ONの評価ではない。今回、訓練問題や正解データは読んでいない。
baseのモデル名は一致するが、正確な訓練base revisionは不明である。

追加の公開metadata監査では、2026-04-02〜07-20の可視main履歴13 commit全てで、
2 safetensors shardのSHA256・size・blob IDが今回の固定revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` と一致した。
公開されたmain重みの違いという不確実性は、この範囲では観測されなかった。
ただし作者が実際に使ったrevision、非公開変更・別branchは証明しておらず、
tokenizer/templateは履歴中に変更されているため、全protocol同一とは言わない。
これは未認証公開metadataだけの照合で、重みの追加downloadはない。

| 固定項目 | 値 |
|---|---|
| donor | `aestudio/gemma-4-31B-it-GRPO-math-lora` |
| revision | `8ec9097a26697380f9e1546fdc1e485f23bd3b61` |
| payload SHA256 | `f6eb60e7b9d7e5a633706e335b112c7ec4da38982dcbb77688146046a28b01e7` |
| config SHA256 | `ed1722c3e2fea491ced9765978ef4ed11eb82889cade675eb7155c18317b76c0` |
| 実保存形式 | F32因子820個、rank128、alpha256、通常scale2 |
| 明示係数 | 1、text410行列だけ。vision / norm / embedding / tokenizer / templateは対象外 |
| source tree | `cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9` |
| spec SHA256 | `cc80630fb25683b9b21245dd39013b6499296a46f73f51b771a9b641208a6f4b` |

cardのBF16記述からexportのdtypeを推定せず、headerのF32を使う。
F32因子をBF16へ先に落とさず、FP32 B@A→FP32(alpha/r)→係数→元BF16をFP32へ展開して加算→BF16へ1回丸める。
作者のPEFT/unmerged実行とbit一致するとの主張ではない。

410という数はmapping漏れではない。元configは60層、
`attention_k_eq_v=true`、sliding_attention50層・full_attention10層である。
実header上、q/k/o/gate/up/downは各60、vは50。
v_projのない層5,11,17,23,29,35,41,47,53,59はfull_attention10層と完全に一致し、
configのK=V設定と整合する。従って **60×6＋50=410**。
全820因子がこの集合へ対応し、余分なfactorは0だった。

## Opus 5.5の実際の意見と採用範囲

匿名I仮説のレビューは `verified_requested_model=true`、
`review_usable=true`、`review_format=public_final_text`。
assistant modelとmodel usageはともに `claude-opus-5-5` のみ、
tools=0、returncode=0、58.523秒、記録cost $0.116276だった。
usageは入力2 / 出力5,141 / cache creation1,681 token。
raw stream・hidden thinking・環境変数は保存していない。候補の私有問題文や測定値を送っていない。

レビューは係数1を最初の単一実験として妥当とした。公開adapterの通常の適用点であり、
別donorで選んだ1/16を転用する理由はない、という助言である。
[因子空間の実観測](2026-11-06-two-donor-factor-observation.md)でも、係数1の未丸めL2は
ja-agent34.1225に対しmath3.33776、cross cosineは0.00002005だった。
差のある方向を測った根拠にはなるが、最適係数や能力の加算性を決める値ではない。

採用する手順は、元の生成設定を変えないこと、F32保持と通常LoRA scale・全mappingの検証、
固定自作問題による構造上の利用可能性の確認、公式devコードと関連分野の得失の報告である。
レビューの追加振幅0.5、runtime adapter比較、追加の巨大診断は自動的な必須条件にしない。
生産経路をFP64へ変更する助言も採用せず、既定FP32順を固定し、
小さい1項だけのfixtureでは独立したFP64 scalar積を使って向きとscaleを検証した。
利用者の既存承認を、reviewerの文言による新しい承認待ちへ置き換えない。

## OFF→ONの限界と固定smoke

作者は学習・評価ともthinking OFFで、thought tokenをsamplerから抑制したと記載している。
LoRA更新がnative ONのthought文脈で有効かは分からない。
finalの欠落、打切り、format、反復や長さの変化を確認する必要があるが、
成績を合わせるためthinking OFF、thought抑制、template変更、cap増加は行わない。
「モデル本体がthoughtを学習したことがない」という意味ではなく、
このadapterがON条件で訓練・検証されていないという制限である。

事前固定した `math-factor-format-v2/native-on-smoke-plan.json` は、
既存advanced-v2のtrain16＋transfer8、合計24例を変更せず一度生成する計画である。
native ON、公式sampling、seed0、cap32768を維持し、既存generatorとCPU sandboxを使う。
比較参照は既存original-v2の15/16＋8/8=23/24で、**新しく同時生成した対照ではない**。
差が出てもこの小集合だけから重みの因果的改善を確定しない。
transfer8はgradientから除外されたままだが、既に結果を見た確認集合であり未見holdoutとは呼ばない。

追加のformat / syntax / missing / execution-unknownが元参照より2件以上、
またはcap到達かつusable final無しなら構造上のstop対象とする。
単なる自作数値testの合格率をWSPの代わりや係数調整の目標にしない。
既存generatorは**thought channelのopen/closeや独立したreasoning token数を測らない**ため、
その検証済みとは記載しない。公開finalだけをsandboxへ渡し、
担当者は既存の数値reportを読む。smoke合格でも能力向上の証拠にはならない。

## 実装と検証の状態

新しい `factor_format` v2はstandalone dense create/verifyへの明示opt-inで、
既定rank16/BF16経路とmodule/native/joint wrappersの仮定を広げない。
rank1〜256、BF16/F32、全extent/shape/hashを検証し、
DoRA/rsLoRA、追加保存module、非一様rank/alpha等は拒否する。
既存merge kernelとoperation wrapperは不変。
[recipe説明](../../scripts/experiments/gemma4_lora/README.md#explicit-factor-format-v2-standalone-dense-only)と
[CPU freeze bridge](../wsp-dense-checkpoint-freeze.md)を参照する。

変更前24sourceとcompiled kernelをreference commit `260fa28` のsnapshotとして保存した。
過去manifestのimplementation hashを書き換えず、旧証拠は旧sourceで再現する。
新source SHAは `58462f8f45373325b23bf845ff53af007c75eb30691fa3b3f5299bfde3fd6cfe`。
順序付きbridge→joint→dense→moduleの19契約testと独立read-onlyレビューが通過した。
全CPU CIは **480件中428成功・52 optional skip、失敗/error0**、
Ruff・mypy73 source files成功、実行前後source変更0である。

実際のmetadata previewは既存imageで4.89秒、成功。
CPU私有コピー作成は2026-11-06 08:23:28 UTCに開始し、08:30:54に完了した。
外側の作成時間446.372秒、manifest内creator時間442.086秒で、計時区間を分けて記録する。

## 保存後の実測とCPU検証

| 指標 | 完成したIの実測 |
|---|---:|
| 変更BF16要素 / tensor | 7,982,864,156 / 410 |
| 実 ΔL2 | 3.034548978874273 |
| 未丸め学習方向のL2 | 3.337757948937157 |
| 実ノルム / 未丸めノルム | 0.9091578914 |
| candidate tree | `9acfbe674fe18fb64eef47d4ece93e72b8f28a5d6ae9ee2bdcf9af54f33ead25` |

0.9091578914は**ノルム比だけ**であり、cosineや方向忠実度、実行能力の保持率ではない。
freshな実方向cosineは今回測っていない。

CPU freeze bridgeは全410行列を元の因子から1回再生成し、保存BF16 byteを照合した。
元と候補の各12 fileをfull hashし、全未編集span、spec / code / kernel、
前後inventory/statも確認した。外側320.670秒、bridge内320.144秒、終了08:37:56 UTC。
`cpu_recipe_verify_calls=1`、GPU/モデルAPI/検証中checkpoint書込み0、
`ram_comparison_performed=false`であり、GPU/RAM比較による証明とは呼ばない。
manifest SHAは `8b44843b8e30f4008552471fb4e78dde4a913512043ff83208ba277fe817ad41`、
freeze SHAは `d14cc8b77bb31e95ba8b8d7ce983d82f9b585016796deb4df54a9a11b9f0d991`。

最初の検証waiterはCI mapに存在しないキーを参照して、実検証起動前にKeyErrorとなった。
その失敗receiptを保持し、runtime側だけを修正した。この失敗ではGEMMは0回で、
数値sourceの変更や追加の410行列再計算はなかった。
その後の上記1回が成功した。PR commit `26a2769` の4 GitHub Actions checksも成功している。

## 生成・公式評価の状態

検証済みfreezeの後に配信・固定24例smoke・公式dev90を完了した。保存重みの証明、構造smoke、公式精度は異なる証拠であり、以下に分けて記録する。Iの全14領域・known64はこの記録では未測定である。

## 証拠

artifact相対rootは `analysis/code90-resume-20261106/`。

| 証拠 | SHA256 |
|---|---|
| `opus-math-donor-c1-review/receipt.json` | `aad51a015b9fa162e008f142ae9953ee68190eb4238b976a197aceb9dda93b24` |
| 同review prompt | `bf8b67179b54f48a2cb351ce40c1942a96623c8f0510c6b4fb96f024357dec9c` |
| cached固定card | `29a6f91aad3c0f7b61af7336bdfe90e487afac6a7d7aec770832182fa85efcd0` |
| `public-math-donor-audit/header-compatibility.json` | `7a2d35ead378b3d0654d4218f5d47a587d1dd70a17e9ddd81dd1a805cdb27f5e` |
| `math-base-revision-audit/weight-revision-comparison.json` | `3b7b38e733ebdc907126e71a1d006678000f543a403b63d2df16275d565567da` |
| `math-factor-format-v2/native-on-smoke-plan.json` | `847fd0e1edd5540ef2c64f20808796c42d03d857fbcc0cee71ddeeb33aa8fee4` |
| `math-factor-format-v2/create-completion.json` | `c9d578643aa700e594b51f3e9ea5d4a74672d46f758e9ee80b46268ab3265bc0` |
| `math-factor-format-v2/verify-completion.json` | `a8d21e96410b7b150da238233c8a9447479ae3c3e0e040a3f75b969cca995833` |
| `math-factor-format-v2/legacy-source.json` | `5027f18ba44e7b2f8ac2f3fb13f46b3bf9b8e63cbfea01895c7b7cc542b78f43` |
| `cpu-integration-math-factor-v2/ci-results.json` | `944f3c82e2a6cd6ccd445db8d79e97af756cd4bbed84ce568d88bb975c97cac7` |

元config SHA256は `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1`。
この文書の作成では新しいモデル呼出し、benchmark本文・正解読取、数値source変更を行っていない。

## 保存後の変更分布とnative ONの実行確認

同じ410 text行列（29,286,727,680要素）でHとIの保存済み変更量を比較した。
元tensor SHAとmappingは一致し、下表の相対L2の分母はこのtext410だけである。
embedding・norm・visionを含むモデル全体の分母と混ぜない。

| 保存後の指標 | H：ja-agent係数1/16 | I：math係数1 |
|---|---:|---:|
| 変更要素率 | 14.78% | 27.26% |
| ΔL2 | 1.88854 | 3.03455 |
| 対象text410に対する相対L2 | 0.07123% | 0.11445% |
| MLPのΔnorm²比率 | 82.54% | 67.18% |
| 上位10層にあるΔnorm² | 35.55% | 19.16% |

Hの上位層は後半、Iは中間付近に分布し、両者の上位10層に重複はなかった。
既存manifestと因子比較JSONだけを集計し、31B payload再読取・GEMM・モデル呼出しは0。
これは局在した能力や機能上の補完性の証拠ではなく、今後の編集量を比較する材料である。
保存後の二方向のcosineは測っていない。
証拠は `math-factor-format-v2/saved-footprint-comparison.json`
（SHA256 `a7c6c569f66f7f11504177a1fe46859a5f2ba56a9b43aa329563c870ca82e5cb`）。

Iをgemma-3へ配置し、固定したadvanced-v2をnative ONで24回生成・実行した。
24/24が形式検査を通って実行され、finish=stopが24、unknown=0。
合格はtrain15/16＋transfer8/8=23/24で、過去のoriginal-v2参照と同数だった。
元で失敗した反射壁の課題は成功へ転じた一方、非復元抽出の課題が失敗へ転じた。
この小集合から能力改善とは判断せず、構造上のstop条件に該当しないことだけを確認した。

| 記録 | 過去のoriginal-v2 | I |
|---|---:|---:|
| 呼出し | 24 | 24 |
| 入力token | 2,864 | 2,864 |
| 出力token | 43,599 | 56,044 |
| 合格 | 23/24 | 23/24 |

出力tokenはAPIの合計であり、独立した思考token数ではない。
`math-factor-format-v2/smoke-numeric-summary.json` と
`structural-smoke-decision.json` に数値と入力SHAを保存した。
2026-11-06 08:42:34 UTCにIとfresh originalのdev90をdirect TP2/c32で開始した。
以下の公式結果までを記録する。採用判断は実験ownerが行い、90達成・合成効果を主張しない。

## 完了した公式dev90：fresh originalとの主比較

両方とも同一90 ID、tier30/dev、manifest SHA
`a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`、
model IDでcompleted・90採点・未採点0・応答error0となった。
固定archive5e3f2dfの公式scorerを使い、問題・生成設定・採点の変更や再実行は行っていない。

- I: `gemma4-code90-math-grpo-c1-c32-dev30-code-20261106`
- fresh original: `gemma4-code90-original-base-c32-dev30-code-pairi-20261106`

| Source | 問数 | fresh original | I |
|---|---:|---:|---:|
| HumanEval+ | 30 | 28/30 | 30/30 |
| MBPP+ | 30 | 22/30 | 22/30 |
| LiveCodeBench | 20 | 14/20 | 14/20 |
| SciCode | 10 | 40.301587% | 40.301587% |
| **公式code（4 source等重み平均）** | **90** | **69.242063%** | **70.908730%** |

差は **+1.666667pp**、公式paired bootstrap95% CIは **[0,+4.166667]pp**
（2,000 resamples / seed20261104）。下限は0であり、strict positiveな区間とは呼ばない。
gain2 / loss0で、点が変わったのは **HumanEval/151、HumanEval/95の0→1**だけだった。
残るLCB・MBPP・SciCodeはsource平均だけでなく各項目も同点である。

両HE項目は今回・過去2回の元モデルでいずれも0で、Iは1だった。
一方、先行Hでも同じ2件を獲得しており、Hに対して独立したコード利益を積み上げた証拠ではない。
過去originalのcode72.575397%、pairHの71.325397%を主対照へ置き換えず、
今回のfresh69.242063%を維持する。Iはその2つの歴史値より低いが、
条件時点の異なる比較だけから因果的な退行とも断定しない。

| 記録量 | fresh original | I |
|---|---:|---:|
| 保存sampleの呼出し | 119 | 119 |
| 入力token | 97,013 | 98,839 |
| 出力token | 332,235 | 385,026 |
| stop / length終了 | 119 / 0 | 118 / 1 |
| wall秒（run開始→完了） | 372.750 | 757.438 |

呼出し119はHE30・MBPP30・LCB20・SciCode39で、問題数90とは異なる。
agent tool手数や独立したHTTP再試行総数ではない。入力token差には前段生成を次stepへ渡す
SciCodeの差が含まれるが、生成内容の診断は行っていない。
両runは08:42:34 UTCに開始し、元08:48:46.858、I08:55:11.687に終了した。
wallはload・生成・採点を含み、別工程を足して二重計上しない。

Iの最後の保存LCB sampleは **abc390_g、出力32,768 token、finish=length**。
今回は元0・I0なので、fresh比較の追加失点には数えない。
過去2回の元モデルは1であり、元モデルの反復変動も存在する。
保存済みmetadataだけから、usable finalの有無、構文やアルゴリズムの原因、
hidden thoughtの長さ、HTTP再試行の回数は断定しない。
途中に読まれたサーバーのcumulative length counterは事前baselineがないため、
特定のbenchmark sampleへ独立に対応付けた証拠ではない。
公式clientのlength終了は自動continuationとは扱わず、保存sampleとサーバー累積値を区別する。

Iはgemma-3、元はgemma-2の直接TP2 endpointで、chat32 / logprob8 / agent1。
CLIはendpoint・label・run ID以外で一致し、mixed gatewayは使っていない。
両実runnerを稼働中に捕捉し、OCI image
`sha256:1467c6e8f2ef28cfbf74a83886af8f9b5880036f5917c4cfd0acca2699596d25`、
Config SHA、16層RootFS、User、host設定、mount集合が一致した。
runner CIDはI `a5ccfaf9a7b2…` / 元 `96c031669f9f…`。
過去Hとはimage IDが違うが、Config/実layer listは一致している。
image ID・layer list hash・checkpoint treeは別の識別子として記録する。

起動前にfreezeの全file既存hash＋fresh stat、元inventory/stat、CPU CI、
Opus review、配信receipt、rootの数値smoke判定、実CID・RO mount・tree・health・idle・IP、
新規run IDを照合した。ここで重みpayloadを再hashした、または追加GEMMを行ったとは言わない。
ランチャーの最初の型チェックはCIの「変更file一覧=[]」をFalseと比較する誤りがあり、
**実起動前**に空list契約へ修正し、実receipt schemaで照合した。最終runtime scriptの
独立read-only reviewも通過した。既存CIを新規model実行の証明として扱わない。

この段階では、Iがfresh対照のHE2件を改善した事実、科学計算・難コードの残差を改善しなかった事実、
出力とwallが増えた事実を併記する。採用・次段階はownerが判断し、この記録から
全14領域のoverall、known64のcode>90、二donor合成の成功を推定しない。
問題本文・生成コード・隠れた推論・gold/非公開test・新しい診断fieldは読んでいない。

証拠は `math-factor-format-v2/dev90-{evidence,final-summary,progress}.json` と
`dev90-runner-evidence-{i,base}.json`、`dev90-pair-launch.json`。

- evidence SHA256: `16e33ab99ca644b933f7a841d6bf827da87debabbf458ee3339fd17f7034b5db`
- final-summary SHA256: `346a94e9fb2f3d355d95b66841e908dc69cb78ebf558800139e70945bd0d9de7`
