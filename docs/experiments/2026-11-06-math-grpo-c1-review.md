# Math GRPO donor I：係数1の事前レビューと検証条件（2026-11-06）

Iは公開math donorを固定係数1で元BF16重みにmergeし、通常のnative thinking ONで評価する実験である。
**本記録時点ではCPU作成が開始された段階で、Iの保存後変更量・生成結果・公式コード成績は未確定**。
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
CPU私有コピー作成は2026-11-06 08:23:28 UTCに開始した。
この時点の記録は作成の成功・保存後ノルム・配信・smoke完了を含まない。
完了後は実変更数、実ΔL2と未丸め方向に対する比、source/target全file・未編集領域の検証を別に記録する。

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
| `math-factor-format-v2/legacy-source.json` | `5027f18ba44e7b2f8ac2f3fb13f46b3bf9b8e63cbfea01895c7b7cc542b78f43` |
| `cpu-integration-math-factor-v2/ci-results.json` | `944f3c82e2a6cd6ccd445db8d79e97af756cd4bbed84ce568d88bb975c97cac7` |

元config SHA256は `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1`。
この文書の作成では新しいモデル呼出し、benchmark本文・正解読取、数値source変更を行っていない。
