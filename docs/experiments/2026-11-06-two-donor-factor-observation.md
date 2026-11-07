# 二つの公開LoRA方向の因子空間比較（2026-11-06）

既存のja-agent donorと新しいmath donorを、共通のtext410行列へ明示対応させてCPUで比較した。
**未丸めの学習方向はほぼ直交したが、科学能力の補完性、生成正答、実BF16保存後の効果は未測定**。
これは次の候補を検討するための重み観測であり、候補の採用やWSP code90達成ではない。

## 固定入力と方法

| 入力 | revision | 保存因子 | alpha / rank |
|---|---|---|---:|
| [Tsunamayo7/gemma4-31b-ja-agent-coder](https://huggingface.co/Tsunamayo7/gemma4-31b-ja-agent-coder/tree/b07b0f2a87a0ff83802e6dc6b7228bab5af55aea) | `b07b0f2a87a0ff83802e6dc6b7228bab5af55aea` | BF16、rank16 | 32 / 16 = 2 |
| [aestudio/gemma-4-31B-it-GRPO-math-lora](https://huggingface.co/aestudio/gemma-4-31B-it-GRPO-math-lora/tree/8ec9097a26697380f9e1546fdc1e485f23bd3b61) | `8ec9097a26697380f9e1546fdc1e485f23bd3b61` | F32、rank128 | 256 / 128 = 2 |

両方のcoefficientを1として、通常LoRA更新 `2BA` の全因子をFP64で処理した。
異rankでも、`(B1.T @ B2)` と `(A1 @ A2.T)` の小さいrank空間積から内積を得られる。
[再利用可能なobserver](../lora-direction-comparison.md)が全体・層・familyごとにnorm²とdotを加算し、
その後cosineを計算する。module cosineの平均や標本座標からの外挿ではない。

元checkpointのheader形状・inventory/statと既存source receiptを照合したが、
元31Bのpayloadは読んでいない。adapterは全file SHAを前後に検証し、選択した因子の非有限値を拒否した。
既存donorのvision378因子は明示除外、新math donorは820個のF32因子全てがtext410に対応した。
ここでのshape対応は両者の訓練base revisionや訓練・評価重複の一致を証明しない。

## 実測

| 指標 | ja-agent | math |
|---|---:|---:|
| 未丸め方向のFrobenius L2 | 34.122502301 | 3.337757949 |
| MLPのnorm²比率 | 79.89924% | 68.33240% |
| attentionのnorm²比率 | 20.10076% | 31.66760% |

global dotは **0.00228387657527**、cosineは **0.00002005288760**。
零norm moduleは0。module cosineの正216 / 負194、範囲−0.00258155〜+0.00271084だった。
60層の集約cosineは−0.00015401〜+0.00046847である。

| Family | ja-agent L2 | math L2 | cross cosine |
|---|---:|---:|---:|
| q | 10.552445 | 1.228678 | +0.0000407830 |
| k | 6.016483 | 0.773054 | +0.0000102262 |
| v | 5.248709 | 0.748454 | +0.0001437958 |
| o | 7.677310 | 0.927645 | +0.0001052489 |
| gate | 23.027437 | 1.834301 | −0.0000290831 |
| up | 17.032949 | 1.838660 | +0.0000482434 |
| down | 10.484214 | 0.931304 | +0.0000113335 |

計算時間15.232秒、外側process時間15.385秒、peak RSSは80,904KiB（79.01MiB）。
4 BLAS threads、GPU/API/モデル呼出し0、dense更新作成0。
全410moduleが有限で、normの丸め補正・cosine clipは0、入力SHA/statの前後継続性が一致した。
値はFP64計算であり、厳密な有理数演算やFP32/BF16 mergeと同じ数値経路ではない。

## 次の判断に使える範囲

mathの未丸めL2はja-agentの約9.78%である。同じ係数でも変更量は同じにならない。
例えばmathへ `1/16` を掛けた場合の**未丸め数学値**はL2 0.208609872になるが、
この値はその係数のspec作成、重み保存、BF16変更数・方向の実測、公式評価を行った結果ではない。
既存Hの係数を機械的に新donorへ移す理由にはしない。

低cosineは、短いproxyに合わせた同種方向の微調整とは異なる方向を観測した根拠になる。
一方、特定層・MLP/attentionに数学能力が局在する、二つを加えると能力が足し算になる、
新donorで科学コードが改善するとは言えない。実BF16と通常生成の公式成績で別途判断する。

## 再現証拠

artifact相対rootは `analysis/code90-resume-20261106/lora-factor-comparison-v1/`。

- `plan.json` SHA256: `6cc7aa6b1b0d62801151cbfc815d6c8c3d8349a96d9ae5556a6534d184aa7188`
- `report.json` SHA256: `d2cc91b57d78245e396ccf199ec306410ee05ebc086f9208e66f205e72233931`
- `launch.json`、`status.json`、`preview.json`、`prepared.json`: 実argv、時間、見積り、終了0。
- ja-agent payload SHA256: `4709c0b3604c6dec88a13215c614c6d68ac81ce6a451887f63adc3ff8f65cc06`
- math payload SHA256: `f6eb60e7b9d7e5a633706e335b112c7ec4da38982dcbb77688146046a28b01e7`
- observer source SHA256: `18e1bbe20e5ae5d9f62a65463b6f54f63915c76c80ad9230f73d152b42a6bf27`

合成3契約fixture、Ruff、mypy 1 file、独立read-onlyレビューを通してから実観測した。
同一実pathへ異なるSHAを宣言したときのpin上書きはレビューで発見し、拒否する修正と回帰条件を加えた。
実観測中にsourceは変更していない。
