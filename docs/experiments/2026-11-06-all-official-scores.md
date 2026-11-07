# Gemma4公式スコア全実行台帳（2026-11-06）

snapshot: 2026-11-07T01:16:12.826409+00:00。WSP **145 runs**（completed140、partial4、running1＝旧保存記録のみ、現在の稼働を意味しない）。
今回のexploratory AD空overlay/AX/AV/AW完了4件を追記。旧141 row objectsの全fieldは完全一致。logical identityを保存checkpoint treeと混同しない。AVのみfresh対照比+0.357143ppで保存/full14確認対象、まだ不採用。working AD7d65と積み上げP→S→AC→ADを維持。ADは全分野改善とは呼ばない。
新規採点・モデル実行なし。指定したNVMe resultsとlegacy resultsの直下run.json/report.jsonだけを採録した。
5つのDeepSWE runはWSP台帳から分離し、完了・取消結果を代表表へ併記する。元R1/R2はincompleteで、この台帳から完成した点を捏造しない。
公式reportから値を転記し、欠測は「—」。表は小数6桁表示、sourceの未丸め値はcatalog JSONに残す。
partialのoverallは公開表で空欄にし、reportの元値はcatalogのreported_overall_rawだけへ保持する。
旧running記録は実プロセス稼働を意味しない。probe sourceのraw scoreはNLL等であり、一律に100倍しない。

## 代表比較

スコアは100点換算。これは代表比較で、最高値だけを選んだランキングではない。
全runの正式値・source別分母・manifest SHAは別の全件台帳を参照する。
字母は実験周期で再利用されたため、正式候補名と実行窓を併記する。
旧A〜Gと、後期B/F/G/H以降を同じ候補として結合しない。

### 公式dev90：HE30＋MBPP30＋LCB20＋SciCode10

4 source等重み平均。単回の生成差を含む。全14領域overallではなく、目標の既知64問とも別集合。

| 候補・実行窓 | Code | 比較対象 | 対照Code | 差 pp | 比較条件 |
|---|---:|---|---:|---:|---|
| 0397・最初の4候補対照 | 70.4921 | — | — | — | B/F/Gの同窓対照 |
| B native-contracts・初回 | 73.4087 | 0397 | 70.4921 | +2.9167 | original比較ではない |
| F matrix-balanced・初回 | 67.5754 | 0397 | 70.4921 | −2.9167 | 同窓 |
| G native＋matrix・初回 | 70.7698 | 0397 | 70.4921 | +0.2778 | 同窓 |
| 元モデル・追加対照 | 72.5754 | — | — | — | 上記B/F/Gより後の窓 |
| new-frame balanced donor | 71.3254 | 元モデル・追加対照 | 72.5754 | −1.2500 | 同一項目、別時点の参照 |
| new-frame native＋donor JOINT | 70.4921 | 元モデル・追加対照 | 72.5754 | −2.0833 | 同一項目、別時点の参照 |
| H coherent donor 1/16・初回 | 72.1587 | fresh original | 71.3254 | +0.8333 | 同時ペア |
| I math GRPO c1 | 70.9087 | fresh original | 69.2421 | +1.6667 | 同時ペア |
| J weighted-reasoning c1 | 72.1587 | fresh original | 69.1032 | +3.0556 | 同時ペア |
| B・Kの4条件窓 | 70.0754 | 同窓original | 70.7698 | −0.6944 | 再測定 |
| H・Kの4条件窓 | 70.7698 | 同窓original | 70.7698 | 0.0000 | 再測定 |
| K＝B＋H | 70.9087 | 同窓original | 70.7698 | +0.1389 | 同時4条件 |
| L＝original＋K1 text c1・初回 | 67.1825 | fresh original | 70.7004 | −3.5179 | 同時ペア |
| O・Mの4条件窓 | 71.1865 | — | — | — | 新しい同時対照 |
| L・Mの4条件窓 | 68.0754 | O | 71.1865 | −3.1111 | 再測定 |
| P＝公開改変base・元protocol | 73.8254 | O | 71.1865 | +2.6389 | 初回working採用、R撤回後に復帰、現在Sの親 |
| M＝P＋同じK1 text c1 | 67.9087 | P | 73.8254 | −5.9167 | 積み上げでは悪化、Pを保持 |
| P・Nの同時対照 | 74.2421 | — | — | — | 同じPのfresh再測定、重み更新ではない |
| N＝P＋K1 text1/16 | 71.6032 | fresh P | 74.2421 | −2.6389 | 同時ペア、Nを採らずPを保持 |
| P・Rの同時対照 | 73.2698 | — | — | — | 同じPのfresh再測定 |
| R＝P＋math MLP全層c1 | 73.6270 | fresh P | 73.2698 | +0.3571 | 一時昇格、後続full14退行で撤回 |
| P・Sの同時対照 | 72.5754 | — | — | — | 同じPのfresh再測定 |
| S＝P＋科学native連成 | 74.2421 | fresh P | 72.5754 | +1.6667 | LCB2 gain／MBPP1 loss、Sci不変、full14得失を受けworking採用 |
| S・ACの同時対照 | 73.2698 | — | — | — | 同じSのfresh再測定、重み変更ではない |
| AC＝S＋code優先matrix小変化 | 73.8254 | fresh S | 73.2698 | +0.5556 | MBPP1 gain／LCB1 loss／Sci部分得失、full14 net gain確認後working採用 |
| AC・AD/AEの同時対照 | 71.4087 | — | — | — | 同じACのfresh再測定、旧73.8254を代用しない |
| AD＝AC＋native依存セット | 73.8254 | fresh AC | 71.4087 | +2.4167 | full14退行とRULER cap2件を明示してworking採用 |
| AE＝AC＋repair matrix | 72.1587 | fresh AC | 71.4087 | +0.7500 | 未採用 |
| AD・AF/AGの同時対照 | 74.2421 | — | — | — | 同じADのfresh再測定、旧73.8254を代用しない |
| AF＝AD＋科学nativeセットB | 72.6587 | fresh AD | 74.2421 | −1.5833 | Sci +2pp、LCB/MBPP各1 loss、不採用 |
| AG＝AD＋科学連成セットH | 73.4087 | fresh AD | 74.2421 | −0.8333 | HE1 loss、他3source同点、不採用 |
| AD・保存AN/ARの同時対照 | 74.7421 | — | — | — | fresh ordinary保存重み対照 |
| AN＝AD＋functional science/code | 72.4365 | fresh AD | 74.7421 | −2.3056 | LCB1 loss＋Sci部分点2 loss、不採用・full14なし |
| AR＝AD＋functional repair | 73.0754 | fresh AD | 74.7421 | −1.6667 | HE1 gain／LCB2 loss、不採用・full14なし |

M−Oは−3.2778pp。Pのworking採用はユーザーの「改善後から積み上げる」方針による。
最終code>90達成、未知集合への汎化、統計的確定とは別である。
各窓のoriginalは69.10〜72.58など変動しており、異なる窓の差を同時比較として扱わない。
M窓のM−O差は−3.2778pp、CI[−9.4167,+2.5000]。M−P差は−5.9167pp、CI[−12.5000,+0.2500]。CIをworking採用の追加gateにしない。
N−fresh Pは−2.6389pp、CI[−7.0833,+1.1111]。Pの73.8254→74.2421は同じ保存重みの再生成窓の差であり、追加の重み改善ではない。
R−fresh Pは+0.3571429pp、CI[−2.142857,+2.857143]。HE95 loss、MBPP641 gain、SciCode80は2/7→3/7。ユーザーの積み上げ方針で一時Rへ更新したが、後続full14のnet lossでR増分を退け、同一保存Pへ復帰した。微小差・不確実性を能力改善の確定としない。R full14は後続snapshotで778/778採点完了し、overall76.3823193／code84.4444444。

### 既知64問：HE20＋MBPP20＋LCB14＋SciCode10

| 実行窓・候補 | Code | 留保 |
|---|---:|---|
| 凍結確認の元モデル | 77.4286 | 固定1491のうちコード64は全件採点 |
| 凍結確認0397 | 82.5476 | 元比+5.1190、CI[+0.8333,+10.6548]、TP2×4構成 |
| 後の0397再測定 | 77.0119 | direct TP2×1、c32 |
| 同窓B native-contracts | 76.4762 | 対0397−0.5357、CI[−8.5119,+7.3810] |
| その後の元モデル追加参照 | 78.2619 | 同一64項目だが別時点 |

この集合で90超は未達。dev10で90超の行があっても同一目標の達成ではない。
固定確認集合は既に成績を見ており、未見testとは呼ばない。

### 3-source codeのみの旧tier10と全14領域

tier10のcodeはHE10＋MBPP10＋LCB6、SciCodeなし。dev/testと実行窓も分ける。

| 窓 | 元Code → 候補Code | 全14領域overall 元 → 候補 |
|---|---|---|
| 旧finalnorm＋1% full dev10 | 84.4444 → 87.7778 | 81.2237 → 81.3005 |
| 旧finalnorm＋1% test10 | 91.1111 → 91.1111 | 85.1305 → 82.3601 |
| B vs fresh original full dev10 | 87.7778 → 84.4444 | 80.4638 → 79.2920 |
| H vs fresh original full dev10 | 87.7778 → 84.4444 | 80.1642 → 81.9725 |
| P保持確認 vs 歴史original | 87.7778 → 87.7778 | 80.1642 → 78.4751 |
| R保持確認 vs 歴史P | 87.7778 → 84.4444 | 78.4751 → 76.3823 |
| S保持確認 vs 歴史P | 87.7778 → 84.4444 | 78.4751 → 78.1294 |
| AC保持確認 vs 歴史S | 84.4444 → 87.7778 | 78.1294 → 78.9349 |
| AD保持確認 vs 歴史AC | 87.7778 → 87.7778 | 78.9349 → 77.5764 |

P保持確認は778/778採点。対originalは歴史比較で、overall差−1.6891pp、CI[−4.2426,+0.9258]。
safetyはXSTest36件の31/36→21/36（−27.7778pp、11 loss／1 gain）であり、他分野の利得とともに退行を明示する。
同時対照ではなく、過去runnerのinstalled package inventoryも未保存のため完全runtime再現や因果差は主張しない。

95-item screenはHE＋MBPPのみのcode、hard30はLCB＋SciCodeのみ。
それらのcode平均、単source再測定、full60も独立のprotocol欄へ分離する。

### DeepSWE：113課題、各1試行、32並列

| モデル | 公式成功／採点 | 未採点 | agent steps | agent記録API calls |
|---|---:|---:|---:|---:|
| 元R3 | 1/113 | 0 | 8,274 | 8,306 |
| 凍結0397 | 1/113 | 0 | 8,071 | 8,098 |
| 旧finalnorm＋1%・ユーザー停止 | 0/109 | 4/113 | 7,844 | 未集計 |

元と0397は双方pass@1=0.88495575%。新成功1・旧成功1の喪失で同点。
旧停止runを0/113完走と表示しない。API列はagentの記録counterで、独立HTTP telemetryではない。
0397ではF2P625→473、P2P226,371→78,067の退行もあり、同点のみをもって改善としない。


## v2までの112 runの候補・protocol・code・overall

### calibration_subset (8 runs)

校正subset。chat/agent/SWEは別source集合、partialあり。通常tierのfull14へ合算しない。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| wsp-calib-dev-chat-20261104 | 原モデル calibration | dev/60 | partial | 6265/6266 | 71.073716 | — |
| wsp-calib-dev-agents-20261104 | 原モデル calibration | dev/60 | completed | 30/30 | — | — |
| wsp-calib-test-agents-20261104 | 原モデル calibration | test/60 | completed | 30/30 | — | — |
| wsp-calib-test-chat-20261104 | 原モデル calibration | test/60 | partial | 6264/6266 | 74.629671 | — |
| wsp-calib-dev-swe-20261104 | 原モデル calibration | dev/60 | completed | 12/12 | — | — |
| wsp-calib-test-swe-20261104 | 原モデル calibration | test/60 | completed | 12/12 | — | — |
| wsp-calib-dev-agents2-20261104 | 原モデル calibration | dev/60 | completed | 30/30 | — | — |
| wsp-calib-test-agents2-20261104 | 原モデル calibration | test/60 | partial | 29/30 | — | — |

### full14_tier10 (20 runs)

全14領域tier10、各778項目。codeはHE10+MBPP10+LCB6の3source（SciCodeなし）。dev/testを別記。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| wsp-base-10-dev-1-20261104 | O 原モデル | dev/10 | completed | 778/778 | 84.444444 | 81.223722 |
| wsp-base-10-dev-2-20261104 | O 原モデル | dev/10 | running | —/778 | — | — |
| wsp-base-10-test-1-20261104 | O 原モデル | test/10 | completed | 778/778 | 91.111111 | 85.130499 |
| gemma4-joint-purpose-balanced-small-dev10-full-20261105 | joint-purpose-balanced-small | dev/10 | completed | 778/778 | 87.777778 | 81.490386 |
| gemma4-joint-purpose-programming-set-dev10-full-20261105 | joint-purpose-programming-set | dev/10 | completed | 778/778 | 87.777778 | 80.705754 |
| gemma4-joint-purpose-chat-set-dev10-full-20261105 | joint-purpose-chat-set | dev/10 | completed | 778/778 | 87.777778 | 80.476941 |
| gemma4-finalnorm-plus1pct-dev10-full-20261105 | finalnorm-plus1pct | dev/10 | completed | 778/778 | 87.777778 | 81.300496 |
| gemma4-donor-text-c1-dev10-full-20261105 | donor-text-c1 | dev/10 | completed | 778/778 | 74.444444 | 76.589114 |
| gemma4-joint-programming-finalnorm-plus1pct-dev10-full-20261105 | joint-programming-finalnorm-plus1pct | dev/10 | completed | 778/778 | 84.444444 | 80.897718 |
| gemma4-finalnorm-plus0p5pct-dev10-full-20261105 | finalnorm-plus0p5pct | dev/10 | completed | 778/778 | 75.555556 | 79.121497 |
| gemma4-donor-text-c0p25-dev10-full-20261105 | donor-text-c0p25 | dev/10 | completed | 778/778 | 84.444444 | 81.194800 |
| gemma4-frozen-c5be8132c454-test10-20261105 | finalnorm-plus1pct | test/10 | completed | 778/778 | 91.111111 | 82.360074 |
| gemma4-base-observed-dev10-20261106 | O 原モデル・観測ON | dev/10 | completed | 778/778 | 87.777778 | 80.214802 |
| gemma4-resume-h0397-dev10-full-20261106 | 0397 | dev/10 | completed | 778/778 | 87.777778 | 81.919633 |
| gemma4-resume-h0485-dev10-full-20261106 | 0485 | dev/10 | completed | 778/778 | 84.444444 | 82.163967 |
| gemma4-code90-original-base-c32-dev10-full14-20261106 | O 原モデル | dev/10 | completed | 778/778 | 87.777778 | 80.463802 |
| gemma4-code90-native-contracts-c32-dev10-full14-20261106 | B native-contracts | dev/10 | completed | 778/778 | 84.444444 | 79.291973 |
| gemma4-code90-coherent-c00625-c32-dev10-full14-20261106 | H coherent donor1/16 | dev/10 | completed | 778/778 | 84.444444 | 81.972528 |
| gemma4-code90-original-base-c32-dev10-full14-pairh-20261106 | O 原モデル | dev/10 | completed | 778/778 | 87.777778 | 80.164156 |
| gemma4-code90-incumbent-p-c32-dev10-full14-20261106 | P 公開改変base・保持確認 | dev/10 | completed | 778/778 | 87.777778 | 78.475099 |

### full14_tier30 (2 runs)

全14領域tier30、各2338項目。dev90のcode集合と同じcode sourcesでも全体測定であり、窓は異なる。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| wsp-base-30-dev-1-20261104 | O 原モデル | dev/30 | completed | 2338/2338 | 69.658730 | 78.768381 |
| wsp-base-30-test-1-20261104 | O 原モデル | test/30 | completed | 2338/2338 | 78.083333 | 78.020991 |

### full14_tier60 (2 runs)

全14領域tier60、各4841項目。dev/testを別集合として扱う。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| wsp-base-60-dev-1-20261104 | O 原モデル | dev/60 | completed | 4841/4841 | 70.421769 | 76.604069 |
| wsp-base-60-test-1-20261104 | O 原モデル | test/60 | completed | 4841/4841 | 75.952381 | 77.417560 |

### screen95_dev10 (32 runs)

95項目screen。codeはHE10+MBPP10の2source。BFCL/IFBench/probe等を含むがfull14 overallなし。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-base-dev10-screen-20261104 | O 原モデル | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l28-minus4-dev10-screen-20261104 | scalar-l28-minus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l28-plus4-dev10-screen-20261104 | scalar-l28-plus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l23-minus4-dev10-screen-20261104 | scalar-l23-minus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l23-plus4-dev10-screen-20261104 | scalar-l23-plus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l41-minus4-dev10-screen-20261104 | scalar-l41-minus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-scalar-l41-plus4-dev10-screen-20261104 | scalar-l41-plus4 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-sparse-code-margin3-dev10-screen-20261105 | sparse-code-margin3 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-sparse-action-margin3-dev10-screen-20261105 | sparse-action-margin3 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-balanced-small-dev10-screen-20261105 | joint-purpose-balanced-small | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-joint-purpose-balanced-small-dev10-screen-20261105-r2 | joint-purpose-balanced-small | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-programming-set-dev10-screen-20261105 | joint-purpose-programming-set | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-programming-set-dev10-screen-20261105-r2 | joint-purpose-programming-set | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-algorithm-set-dev10-screen-20261105 | joint-purpose-algorithm-set | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-workflow-language-dev10-screen-20261105 | joint-purpose-workflow-language | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-sparse-code-action6-dev10-screen-20261105 | sparse-code-action6 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-balanced-medium-dev10-screen-20261105 | joint-purpose-balanced-medium | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-joint-purpose-balanced-medium-dev10-screen-20261105-r2 | joint-purpose-balanced-medium | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-joint-purpose-chat-set-dev10-screen-20261105 | joint-purpose-chat-set | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-finalnorm-minus1pct-dev10-screen-20261105 | finalnorm-minus1pct | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-finalnorm-plus1pct-dev10-screen-20261105 | finalnorm-plus1pct | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-finalnorm-plus1pct-dev10-screen-20261105-r2 | finalnorm-plus1pct | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-finalnorm-minus1pct-dev10-screen-20261105-r2 | finalnorm-minus1pct | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-donor-text-c1-dev10-screen-20261105 | donor-text-c1 | dev/10 | completed | 95/95 | 70.000000 | — |
| gemma4-joint-programming-finalnorm-plus1pct-dev10-screen-20261105 | joint-programming-finalnorm-plus1pct | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-finalnorm-plus0p5pct-dev10-screen-20261105 | finalnorm-plus0p5pct | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-donor-text-c0p25-dev10-screen-20261105 | donor-text-c0p25 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-resume-hs0020-dev10-screen-20261106 | hs0020 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-resume-h0397-dev10-screen-20261106 | 0397 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-resume-h0485-dev10-screen-20261106 | 0485 | dev/10 | completed | 95/95 | 85.000000 | — |
| gemma4-resume-localgc-dev10-screen-20261106 | localG-C | dev/10 | completed | 95/95 | 90.000000 | — |
| gemma4-resume-composed-dev10-screen-20261106 | 0397＋localG-C | dev/10 | completed | 95/95 | 85.000000 | — |

### hard_code30_dev30 (13 runs)

LCB20+SciCode10の2source平均。4source code90へ置き換えない。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-joint-purpose-balanced-small-dev30-hard-code-20261105 | joint-purpose-balanced-small | dev/30 | completed | 30/30 | 50.150794 | — |
| gemma4-joint-purpose-programming-set-dev30-hard-code-20261105 | joint-purpose-programming-set | dev/30 | completed | 30/30 | 55.150794 | — |
| gemma4-joint-purpose-algorithm-set-dev30-hard-code-20261105 | joint-purpose-algorithm-set | dev/30 | completed | 30/30 | 53.365079 | — |
| gemma4-joint-purpose-chat-set-dev30-hard-code-20261105 | joint-purpose-chat-set | dev/30 | completed | 30/30 | 57.650794 | — |
| gemma4-finalnorm-plus1pct-dev30-hard-code-20261105 | finalnorm-plus1pct | dev/30 | completed | 30/30 | 58.206349 | — |
| gemma4-donor-text-c1-dev30-hard-code-20261105 | donor-text-c1 | dev/30 | completed | 30/30 | 55.984127 | — |
| gemma4-joint-programming-finalnorm-plus1pct-dev30-hard-code-20261105 | joint-programming-finalnorm-plus1pct | dev/30 | completed | 30/30 | 57.650794 | — |
| gemma4-finalnorm-plus0p5pct-dev30-hard-code-20261105 | finalnorm-plus0p5pct | dev/30 | completed | 30/30 | 55.150794 | — |
| gemma4-donor-text-c0p25-dev30-hard-code-20261105 | donor-text-c0p25 | dev/30 | completed | 30/30 | 50.150794 | — |
| gemma4-resume-h0397-dev30-hard-code-20261106 | 0397 | dev/30 | completed | 30/30 | 57.650794 | — |
| gemma4-resume-h0485-dev30-hard-code-20261106 | 0485 | dev/30 | completed | 30/30 | 60.150794 | — |
| gemma4-resume-localgc-dev30-hard-code-20261106 | localG-C | dev/30 | completed | 30/30 | 55.150794 | — |
| gemma4-resume-composed-dev30-hard-code-20261106 | 0397＋localG-C | dev/30 | completed | 30/30 | 60.150794 | — |

### selected_subset_162 (1 runs)

観測損失の選択162項目。コード集合ではなく、全体overallなし。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-joint-purpose-balanced-small-dev10-observed-losses-repeat-20261105 | joint-purpose-balanced-small | dev/10 | completed | 162/162 | — | — |

### selected_subset_107 (1 runs)

core+longの選択107項目。codeはHE+MBPPのみ。全体overallなし。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-joint-donor-quarter-finalnorm-plus1pct-dev10-core-long-20261105 | joint-donor-quarter-finalnorm-plus1pct | dev/10 | completed | 107/107 | 85.000000 | — |

### livecodebench20_dev30 (2 runs)

LCB20のみの再測定。単source scoreであり4source codeではない。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-resume-h0397-dev30-lcb-repeat-20261106 | 0397 | dev/30 | completed | 20/20 | 75.000000 | — |
| gemma4-resume-base-dev30-lcb-matched-20261106 | O 原モデル | dev/30 | completed | 20/20 | 70.000000 | — |

### confirmation_reserve_1491 (1 runs)

固定確認1491。元モデルはmteb1項目取消、1490採点。コード64は全採点。未採点を0にしない。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-confirmation-base-1491-20261106 | O 原モデル | test/30 | partial | 1490/1491 | 77.428571 | — |

### confirmation_reserve_1490 (1 runs)

同じ固定manifestからmtebを明示除外した1490。コード64は全採点。元1491との共通1490比較を使用。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-confirmation-h0397-1490-20261106 | 0397 | test/30 | completed | 1490/1490 | 82.547619 | 80.311317 |

### code90_dev30 (26 runs)

dev/tier30のコード4source：HE30/MBPP30/LCB20/SciCode10、各90項目。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-code90-0397-dev30-code-20261106 | 0397 | dev/30 | completed | 90/90 | 70.075397 | — |
| gemma4-code90-h0397-control-c32-dev30-code-20261106 | 0397 | dev/30 | completed | 90/90 | 70.492063 | — |
| gemma4-code90-native-contracts-c32-dev30-code-20261106 | B native-contracts | dev/30 | completed | 90/90 | 73.408730 | — |
| gemma4-code90-joint-c32-dev30-code-20261106 | G native＋matrix | dev/30 | completed | 90/90 | 70.769841 | — |
| gemma4-code90-matrix-balanced-c32-dev30-code-20261106 | F matrix-balanced | dev/30 | completed | 90/90 | 67.575397 | — |
| gemma4-code90-original-base-c32-dev30-code-20261106 | O 原モデル | dev/30 | completed | 90/90 | 72.575397 | — |
| gemma4-code90-public-final-balanced-c32-dev30-code-20261106 | correctLL balanced donor | dev/30 | completed | 90/90 | 71.325397 | — |
| gemma4-code90-public-final-joint-c32-dev30-code-20261106 | correctLL JOINT | dev/30 | completed | 90/90 | 70.492063 | — |
| gemma4-code90-original-base-c32-dev30-code-pairh-20261106 | O 原モデル | dev/30 | completed | 90/90 | 71.325397 | — |
| gemma4-code90-coherent-c00625-c32-dev30-code-20261106 | H coherent donor1/16 | dev/30 | completed | 90/90 | 72.158730 | — |
| gemma4-code90-original-base-c32-dev30-code-pairi-20261106 | O 原モデル | dev/30 | completed | 90/90 | 69.242063 | — |
| gemma4-code90-math-grpo-c1-c32-dev30-code-20261106 | I math GRPO c1 | dev/30 | completed | 90/90 | 70.908730 | — |
| gemma4-code90-original-base-c32-dev30-code-pairj-20261106 | O 原モデル | dev/30 | completed | 90/90 | 69.103175 | — |
| gemma4-code90-weighted-reasoning-c1-c32-dev30-code-20261106 | J weighted-reasoning c1 | dev/30 | completed | 90/90 | 72.158730 | — |
| gemma4-code90-native-b-plus-coherent-h-c32-dev30-code-pairk-20261106 | K＝B＋H | dev/30 | completed | 90/90 | 70.908730 | — |
| gemma4-code90-original-base-c32-dev30-code-pairk-20261106 | O 原モデル | dev/30 | completed | 90/90 | 70.769841 | — |
| gemma4-code90-native-b-c32-dev30-code-pairk-20261106 | B native-contracts | dev/30 | completed | 90/90 | 70.075397 | — |
| gemma4-code90-coherent-h-c32-dev30-code-pairk-20261106 | H coherent donor1/16 | dev/30 | completed | 90/90 | 70.769841 | — |
| gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairl-20261106 | L＝O＋K1 | dev/30 | completed | 90/90 | 67.182540 | — |
| gemma4-code90-original-base-c32-dev30-code-pairl-20261106 | O 原モデル | dev/30 | completed | 90/90 | 70.700397 | — |
| gemma4-code90-heretic-public-base-c32-dev30-code-pairm-20261106 | P 公開base・元protocol | dev/30 | completed | 90/90 | 73.825397 | — |
| gemma4-code90-heretic-public-base-k1-c1-c32-dev30-code-pairm-20261106 | M＝P＋K1 | dev/30 | completed | 90/90 | 67.908730 | — |
| gemma4-code90-original-base-c32-dev30-code-pairm-20261106 | O 原モデル | dev/30 | completed | 90/90 | 71.186508 | — |
| gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairm-20261106 | L＝O＋K1 | dev/30 | completed | 90/90 | 68.075397 | — |
| gemma4-code90-heretic-public-base-k1-c00625-c32-dev30-code-20261106 | N＝P＋K1 text1/16 | dev/30 | completed | 90/90 | 71.603175 | — |
| gemma4-code90-heretic-public-base-c32-dev30-code-pairn-20261106 | P・Nの同時対照 | dev/30 | completed | 90/90 | 74.242063 | — |

### known64_test30 (3 runs)

test/tier30の固定確認subset：HE20/MBPP20/LCB14/SciCode10。既に成績を見た64項目。

| Run ID | 候補 | split/tier | 状態 | 採点/予定 | Code % | Overall % |
|---|---|---|---|---:|---:|---:|
| gemma4-code90-h0397-control-c32-known64-r1-20261106 | 0397 | test/30 | completed | 64/64 | 77.011905 | — |
| gemma4-code90-native-contracts-c32-known64-r1-20261106 | B native-contracts | test/30 | completed | 64/64 | 76.476190 | — |
| gemma4-code90-original-base-c32-known64-r1-20261106 | O 原モデル | test/30 | completed | 64/64 | 78.261905 | — |

## 候補ラベルと完全性の確認

初期22候補・45 completed dev runは保存済みclosure監査の正式candidate/run対応を使用し、全45がこの112-run台帳に存在することを確認した。
旧joint-purposeのA/B/C/D/E/Gと後期scienceのB/F/G、H以降の公開donor試行は別cycleである。方法案・CPU予測・finite proxyだけの案を新たなWSPスコアとして追加しない。
全runの分母は以下のsource items/scoredで再確認できる。同じmanifest SHAだけではsubset・split・実行窓・topologyの同一性を保証しない。
B73.4087の同時対照は0397であり、後日original72.5754を同時対照へ差替えない。
観測ONの元モデル、TP2×4からdirect TP2×1への変更、過去baselineのCID記録欠測は各個別実験記録の留保を維持する。

## 全runの領域点・source元単位・証拠hash

領域点は100点換算。source scoreは公式reportの元単位（通常成功割合0–1、probeはNLL等）。SciCodeは各問題のstep率の平均を保持し、全step一括率に置換しない。

<details><summary>wsp-calib-dev-chat-20261104</summary>

候補: 原モデル calibration。partial、dev/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 85.740741 |
| expert | 79.591837 |
| facts | 70.833333 |
| math | 92.144097 |
| code | 71.073716 |
| instructions | 88.301735 |
| tools | 80.000000 |
| agents | — |
| long-context | 77.956647 |
| reasoning | 80.434783 |
| commonsense | 62.781815 |
| multilingual | 79.629630 |
| vision | 72.777778 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.983333 | 15/15 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.577778 | 585/585 | — | — |
| bbeh | reasoning | 0.775362 | 138/138 | 0.000000 | 0.000000 |
| bfcl | tools | 0.800000 | 450/450 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.795918 | 98/98 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 150/150 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.631111 | 900/900 | — | — |
| hmmt-feb-2026 | math | 0.796875 | 16/16 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.950617 | 81/81 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.810811 | 148/148 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.955224 | 268/268 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.700000 | 240/240 | 0.000000 | 0.000000 |
| livecodebench | code | 0.773333 | 75/75 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.750000 | 44/45 | 0.000000 | 0.000000 |
| math-500 | math | 0.905556 | 180/180 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.811111 | 90/90 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.866667 | 165/165 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.857407 | 540/540 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.822222 | 180/180 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.727778 | 180/180 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.602588 | 48/48 | 0.000000 | 0.000000 |
| musr | reasoning | 0.833333 | 90/90 | 0.000000 | 0.000000 |
| probe-code | None | 4.048434 | 120/120 | — | — |
| probe-wikipedia-ja | None | 7.372237 | 120/120 | — | — |
| probe-wikitext | None | 8.580089 | 31/31 | — | — |
| ruler | long-context | 0.986111 | 72/72 | 0.013889 | 0.013889 |
| scicode | code | 0.307887 | 32/32 | 0.000000 | 0.000000 |
| triviaqa | facts | 0.708333 | 360/360 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.674566 | 633/633 | — | — |
| xstest | safety | 0.888889 | 216/216 | 0.000000 | 0.000000 |

開始: 2026-11-04T03:00:45.581428Z。終了: 2026-11-04T05:18:56.417267Z。
- run.json SHA256: abd015e36d67bd9b0044fae8fb67c627244b309eeed8d1fff166ee5b91ab443d
- report.json SHA256: 39cadf0f754bb42a92fb48d71f366ac37372f74ecd6c26598e2e61c357c61797

</details>

<details><summary>wsp-calib-dev-agents-20261104</summary>

候補: 原モデル calibration。completed、dev/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 30.555556 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.000000 | 12/12 | — | — |
| terminal-bench | agents | 0.611111 | 18/18 | — | — |

開始: 2026-11-04T03:52:56.384268Z。終了: 2026-11-04T05:07:22.624067Z。
- run.json SHA256: c885ea5e16bbe8633271b5c834f7a5d0bb06e46761f094bab611769478274630
- report.json SHA256: da4f2c72b3fcb78c44824dade557e86ec4c37a0ac1fafcb345cec1161dcbc969

</details>

<details><summary>wsp-calib-test-agents-20261104</summary>

候補: 原モデル calibration。completed、test/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 13.888889 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.000000 | 12/12 | — | — |
| terminal-bench | agents | 0.277778 | 18/18 | — | — |

開始: 2026-11-04T03:52:56.385796Z。終了: 2026-11-04T05:09:41.841028Z。
- run.json SHA256: 7b4b049ac4ebbf081b509e7e5bc1e6c27e6a657ac385b07253046c3eeb8b495b
- report.json SHA256: 30d4114a9bcc6eb6760768df9052765a14d1302080e1a076600587bde362b788

</details>

<details><summary>wsp-calib-test-chat-20261104</summary>

候補: 原モデル calibration。partial、test/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 85.925926 |
| expert | 89.795918 |
| facts | 76.111111 |
| math | 87.274306 |
| code | 74.629671 |
| instructions | 89.058088 |
| tools | 80.000000 |
| agents | — |
| long-context | 74.271109 |
| reasoning | 79.685990 |
| commonsense | 62.462106 |
| multilingual | 80.206229 |
| vision | 70.555556 |
| safety | 90.740741 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.883333 | 15/15 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.579487 | 585/585 | — | — |
| bbeh | reasoning | 0.782609 | 138/138 | 0.000000 | 0.000000 |
| bfcl | tools | 0.800000 | 450/450 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.897959 | 98/98 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 150/150 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.646667 | 900/900 | — | — |
| hmmt-feb-2026 | math | 0.718750 | 16/16 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.925926 | 81/81 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.844595 | 148/148 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.936567 | 268/268 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.687500 | 240/240 | 0.000000 | 0.000000 |
| livecodebench | code | 0.853333 | 75/75 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.558140 | 43/45 | 0.000000 | 0.000000 |
| math-500 | math | 0.888889 | 180/180 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.788889 | 90/90 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.890909 | 165/165 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.859259 | 540/540 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.827778 | 180/180 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.705556 | 180/180 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.683883 | 48/48 | 0.000000 | 0.000000 |
| musr | reasoning | 0.811111 | 90/90 | 0.000000 | 0.000000 |
| probe-code | None | 4.098405 | 120/120 | — | — |
| probe-wikipedia-ja | None | 7.398139 | 120/120 | — | — |
| probe-wikitext | None | 8.563951 | 31/31 | — | — |
| ruler | long-context | 0.986111 | 72/72 | 0.013889 | 0.013889 |
| scicode | code | 0.417039 | 32/32 | 0.000000 | 0.000000 |
| triviaqa | facts | 0.761111 | 360/360 | 0.002778 | 0.002778 |
| winogrande | commonsense | 0.647709 | 633/633 | — | — |
| xstest | safety | 0.907407 | 216/216 | 0.000000 | 0.000000 |

開始: 2026-11-04T03:53:10.943638Z。終了: 2026-11-04T05:21:19.193792Z。
- run.json SHA256: 300894da3f7d847e3268fd87af9b07092ad5e86c3e3e95d9e0330b94c6d51ee1
- report.json SHA256: 75cb941a6c2396fa6a32ad8a3cc9584cb87a5904f4e9d464ef90909bb2bd5eb5

</details>

<details><summary>wsp-calib-dev-swe-20261104</summary>

候補: 原モデル calibration。completed、dev/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 16.666667 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.166667 | 12/12 | — | — |

開始: 2026-11-04T04:30:19.022653Z。終了: 2026-11-04T05:11:28.599425Z。
- run.json SHA256: 8d1cfe240875f505e80c5326369d7cc1c8ea86644c0390549b861fff5d333dca
- report.json SHA256: a32bb0aec97d868dd948a0616659a385a7045230a971ddd3ba9f39908532d842

</details>

<details><summary>wsp-calib-test-swe-20261104</summary>

候補: 原モデル calibration。completed、test/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 16.666667 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.166667 | 12/12 | — | — |

開始: 2026-11-04T04:30:21.392785Z。終了: 2026-11-04T05:12:47.505566Z。
- run.json SHA256: c0b1fd1d865bf482c1a1c2a97299be2d5e7511ceafe0627b130040b79f7d95b0
- report.json SHA256: 9b7e5b6fec4243fd26c269bb8fe904f692b5ec2a39b988c535be81301f52a883

</details>

<details><summary>wsp-calib-dev-agents2-20261104</summary>

候補: 原モデル calibration。completed、dev/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 44.444444 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.166667 | 12/12 | — | — |
| terminal-bench | agents | 0.722222 | 18/18 | — | — |

開始: 2026-11-04T05:27:03.463210Z。終了: 2026-11-04T05:47:17.867225Z。
- run.json SHA256: dd9521497547dcabfc758bf6a146b0a71800075e2829c0aa997a53f8606a7c6d
- report.json SHA256: 2d212dc12b66279a296fbc242f14d067d8217e34bb4df6d9024a68996c49c909

</details>

<details><summary>wsp-calib-test-agents2-20261104</summary>

候補: 原モデル calibration。partial、test/tier60、manifest SHA256: 50552147f6dbe74ae83ab36b484d6e35480864c6bdec400ae5e568000ad0bac0。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | 36.029412 |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| swe-bench-pro | agents | 0.250000 | 12/12 | — | — |
| terminal-bench | agents | 0.470588 | 17/18 | — | — |

開始: 2026-11-04T05:27:03.575165Z。終了: 2026-11-04T07:35:00.912842Z。
- run.json SHA256: 2157be2502312b2265a7adbd1e9c2dac27706ebb58349ef09833e1e132677b89
- report.json SHA256: 73894a1e5817bff880784c9791f548a28615196e136e11167188d0f8e1f0ec3f

</details>

<details><summary>wsp-base-10-dev-1-20261104</summary>

候補: O 原モデル。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 88.333333 |
| expert | 87.500000 |
| facts | 65.000000 |
| math | 97.500000 |
| code | 84.444444 |
| instructions | 95.416667 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 79.130435 |
| commonsense | 66.333333 |
| multilingual | 80.984848 |
| vision | 87.500000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.782609 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.625000 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.883333 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.136125 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.130973 | 12/12 | — | — |
| probe-wikitext | None | 8.608087 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.650000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-04T07:37:57.494650Z。終了: 2026-11-04T07:48:31.873157Z。
- run.json SHA256: 2d33965fd6da69a222bb6f0f6d0b4bd3e1b634c912938a870d262b6d20be5c50
- report.json SHA256: c5c57858230272720e8d7d781cf9be7c73bfea79a0a30e2ac8313e772ab0c763

</details>

<details><summary>wsp-base-10-dev-2-20261104</summary>

候補: O 原モデル。running、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| 記録なし | — | — | — | — | — |

開始: 2026-11-04T07:50:20.705907Z。終了: 未記録。
- run.json SHA256: e08c7eb41454eb84498878019d9f6adf65e0ce16c9e0d87daaa732e6f4f10f73

</details>

<details><summary>wsp-base-10-test-1-20261104</summary>

候補: O 原モデル。completed、test/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 93.333333 |
| expert | 87.500000 |
| facts | 65.000000 |
| math | 98.750000 |
| code | 91.111111 |
| instructions | 82.083333 |
| tools | 77.500000 |
| agents | 100.000000 |
| long-context | 89.447212 |
| reasoning | 83.478261 |
| commonsense | 66.666667 |
| multilingual | 84.040404 |
| vision | 81.250000 |
| safety | 91.666667 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.710000 | 100/100 | — | — |
| bbeh | reasoning | 0.869565 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.775000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.620000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.666667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.900000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.933333 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.788944 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 3.066865 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.147582 | 12/12 | — | — |
| probe-wikitext | None | 8.206123 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 1.000000 | 2/2 | — | — |
| triviaqa | facts | 0.650000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.670000 | 100/100 | — | — |
| xstest | safety | 0.916667 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-04T07:51:02.100174Z。終了: 2026-11-04T07:59:26.319139Z。
- run.json SHA256: 9e02d061aabbbb75f438602e5ebbe84ef976dfb8160b23a9ca2cdfed15424f65
- report.json SHA256: 8f47fe06857b99a94eee996c97391318489f0ab4e392cb50b14a463a344e6e5b

</details>

<details><summary>wsp-base-30-dev-1-20261104</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 83.888889 |
| expert | 85.416667 |
| facts | 69.166667 |
| math | 90.905449 |
| code | 69.658730 |
| instructions | 87.500000 |
| tools | 85.833333 |
| agents | 54.166667 |
| long-context | 82.080138 |
| reasoning | 85.217391 |
| commonsense | 64.222222 |
| multilingual | 76.414141 |
| vision | 81.250000 |
| safety | 87.037037 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.937500 | 12/12 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.600000 | 300/300 | — | — |
| bbeh | reasoning | 0.804348 | 46/46 | 0.000000 | 0.000000 |
| bfcl | tools | 0.858333 | 120/120 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.854167 | 48/48 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 50/50 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.643333 | 300/300 | — | — |
| hmmt-feb-2026 | math | 0.865385 | 13/13 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 50/50 | 0.020000 | 0.020000 |
| ifeval | instructions | 0.950000 | 120/120 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.583333 | 72/72 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.900000 | 10/10 | 0.000000 | 0.000000 |
| math-500 | math | 0.833333 | 60/60 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 55/55 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.838889 | 180/180 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 60/60 | 0.016667 | 0.016667 |
| mmmu-pro | vision | 0.812500 | 48/48 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.645737 | 16/16 | 0.000000 | 0.000000 |
| musr | reasoning | 0.900000 | 30/30 | 0.000000 | 0.000000 |
| probe-code | None | 3.969614 | 40/40 | — | — |
| probe-wikipedia-ja | None | 7.116688 | 40/40 | — | — |
| probe-wikitext | None | 8.558045 | 16/16 | — | — |
| ruler | long-context | 0.916667 | 24/24 | 0.083333 | 0.083333 |
| scicode | code | 0.303016 | 10/10 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 4/4 | — | — |
| terminal-bench | agents | 0.833333 | 6/6 | — | — |
| triviaqa | facts | 0.691667 | 120/120 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.683333 | 300/300 | — | — |
| xstest | safety | 0.870370 | 108/108 | 0.000000 | 0.000000 |

開始: 2026-11-04T07:59:30.595245Z。終了: 2026-11-04T08:33:30.012122Z。
- run.json SHA256: 75dcf95b9641240c975f51dc14a86adaf5675425231788711de12dbdbecd93da
- report.json SHA256: a65fd199ecbc11c74c50e0489bd67c0f19daac27747b7a460e331b4b2feab855

</details>

<details><summary>wsp-base-30-test-1-20261104</summary>

候補: O 原モデル。completed、test/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 88.333333 |
| expert | 85.416667 |
| facts | 72.500000 |
| math | 89.126603 |
| code | 78.083333 |
| instructions | 89.750000 |
| tools | 82.500000 |
| agents | 29.166667 |
| long-context | 76.232784 |
| reasoning | 83.478261 |
| commonsense | 63.888889 |
| multilingual | 79.974747 |
| vision | 81.250000 |
| safety | 92.592593 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.895833 | 12/12 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.600000 | 300/300 | — | — |
| bbeh | reasoning | 0.869565 | 46/46 | 0.000000 | 0.000000 |
| bfcl | tools | 0.825000 | 120/120 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.854167 | 48/48 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 50/50 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.646667 | 300/300 | — | — |
| hmmt-feb-2026 | math | 0.769231 | 13/13 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.820000 | 50/50 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 120/120 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.708333 | 72/72 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.890909 | 55/55 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.883333 | 180/180 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 60/60 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 48/48 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.711984 | 16/16 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| probe-code | None | 3.489751 | 40/40 | — | — |
| probe-wikipedia-ja | None | 7.312945 | 40/40 | — | — |
| probe-wikitext | None | 8.505503 | 16/16 | — | — |
| ruler | long-context | 0.875000 | 24/24 | 0.125000 | 0.125000 |
| scicode | code | 0.523333 | 10/10 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 4/4 | — | — |
| terminal-bench | agents | 0.333333 | 6/6 | — | — |
| triviaqa | facts | 0.725000 | 120/120 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.670000 | 300/300 | — | — |
| xstest | safety | 0.925926 | 108/108 | 0.000000 | 0.000000 |

開始: 2026-11-04T08:33:36.526208Z。終了: 2026-11-04T09:30:15.113896Z。
- run.json SHA256: a91d1b661d3fceba13cee4d222cca9a8db36b9858c6d3c4140097447943ddbc0
- report.json SHA256: 0177de9c46529d4d24bcb3ab388181aa7e677b7b51bf6a8bd9bc113f674bc2e9

</details>

<details><summary>wsp-base-60-dev-1-20261104</summary>

候補: O 原モデル。completed、dev/tier60、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 88.055556 |
| expert | 83.673469 |
| facts | 70.416667 |
| math | 89.973958 |
| code | 70.421769 |
| instructions | 85.936870 |
| tools | 81.333333 |
| agents | 45.833333 |
| long-context | 76.624464 |
| reasoning | 76.159420 |
| commonsense | 63.030651 |
| multilingual | 79.608586 |
| vision | 72.500000 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.916667 | 15/15 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.577586 | 580/580 | — | — |
| bbeh | reasoning | 0.706522 | 92/92 | 0.000000 | 0.000000 |
| bfcl | tools | 0.813333 | 300/300 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.836735 | 98/98 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 100/100 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.645000 | 600/600 | — | — |
| hmmt-feb-2026 | math | 0.765625 | 16/16 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.983333 | 60/60 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.763514 | 148/148 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.955224 | 268/268 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.637500 | 160/160 | 0.000000 | 0.000000 |
| livecodebench | code | 0.740000 | 50/50 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| math-500 | math | 0.916667 | 120/120 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 60/60 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 110/110 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.880556 | 360/360 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.841667 | 120/120 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.725000 | 120/120 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.582067 | 32/32 | 0.000000 | 0.000000 |
| musr | reasoning | 0.816667 | 60/60 | 0.000000 | 0.000000 |
| probe-code | None | 4.114659 | 80/80 | — | — |
| probe-wikipedia-ja | None | 7.284737 | 80/80 | — | — |
| probe-wikitext | None | 8.538738 | 30/30 | — | — |
| ruler | long-context | 0.916667 | 48/48 | 0.083333 | 0.083333 |
| scicode | code | 0.326871 | 28/28 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 8/8 | — | — |
| terminal-bench | agents | 0.666667 | 12/12 | — | — |
| triviaqa | facts | 0.704167 | 240/240 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.668333 | 600/600 | — | — |
| xstest | safety | 0.888889 | 216/216 | 0.000000 | 0.000000 |

開始: 2026-11-04T09:30:21.451667Z。終了: 2026-11-04T10:30:09.952237Z。
- run.json SHA256: d7884e8b3e8d44715f41d34557ff8738c8467c8431b1cbe7f840fbc7ef40ea7f
- report.json SHA256: da86e8f3b9b8bf46b3b283b4bd199a0100f870cd7659d17afac22037903e6e42

</details>

<details><summary>wsp-base-60-test-1-20261104</summary>

候補: O 原モデル。completed、test/tier60、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 86.388889 |
| expert | 87.755102 |
| facts | 77.083333 |
| math | 87.942708 |
| code | 75.952381 |
| instructions | 87.520169 |
| tools | 82.666667 |
| agents | 45.833333 |
| long-context | 71.826987 |
| reasoning | 77.500000 |
| commonsense | 62.704981 |
| multilingual | 80.763889 |
| vision | 69.166667 |
| safety | 90.740741 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.900000 | 15/15 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.584483 | 580/580 | — | — |
| bbeh | reasoning | 0.750000 | 92/92 | 0.000000 | 0.000000 |
| bfcl | tools | 0.826667 | 300/300 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.877551 | 98/98 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 100/100 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.638333 | 600/600 | — | — |
| hmmt-feb-2026 | math | 0.734375 | 16/16 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.950000 | 60/60 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.817568 | 148/148 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.932836 | 268/268 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.706250 | 160/160 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 50/50 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.466667 | 30/30 | 0.000000 | 0.000000 |
| math-500 | math | 0.883333 | 120/120 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 60/60 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.900000 | 110/110 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.863889 | 360/360 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.816667 | 120/120 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.691667 | 120/120 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.729810 | 32/32 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 60/60 | 0.000000 | 0.000000 |
| probe-code | None | 4.014010 | 80/80 | — | — |
| probe-wikipedia-ja | None | 7.274277 | 80/80 | — | — |
| probe-wikitext | None | 8.573560 | 30/30 | — | — |
| ruler | long-context | 0.958333 | 48/48 | 0.041667 | 0.041667 |
| scicode | code | 0.488095 | 28/28 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 8/8 | — | — |
| terminal-bench | agents | 0.666667 | 12/12 | — | — |
| triviaqa | facts | 0.770833 | 240/240 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.658333 | 600/600 | — | — |
| xstest | safety | 0.907407 | 216/216 | 0.000000 | 0.000000 |

開始: 2026-11-04T10:30:17.570154Z。終了: 2026-11-04T11:34:49.316172Z。
- run.json SHA256: 7b0ab0f4ed6f5f9c1bd4976744f868b6ce92153cb6a92f34b2d47d99733617a2
- report.json SHA256: 96d75203eb2dc9135f5585868a24a72c6ab45254cac5d652af93dd56477590a0

</details>

<details><summary>gemma4-base-dev10-screen-20261104</summary>

候補: O 原モデル。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.125331 | 12/12 | — | — |
| probe-wikitext | None | 8.567775 | 8/8 | — | — |

開始: 2026-11-04T15:58:56.779029Z。終了: 2026-11-04T16:00:11.196540Z。
- run.json SHA256: 7b6f4fe3116d5c031b2185028f4daeaaf8ae9032ddad2a5b107d3b28d2135626
- report.json SHA256: cbd397699dba53f1b8905a2f6f85a307415dc2d76b39774b59471fedcb3def91

</details>

<details><summary>gemma4-scalar-l28-minus4-dev10-screen-20261104</summary>

候補: scalar-l28-minus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.139366 | 12/12 | — | — |
| probe-wikitext | None | 8.654247 | 8/8 | — | — |

開始: 2026-11-05T02:08:38.242037Z。終了: 2026-11-05T02:09:47.551980Z。
- run.json SHA256: 9df243b3c2364712b54a48736b34588bbe5f1acc84923ec1b87acf8a6b2b732e
- report.json SHA256: 315a8caa0147af57e2509ca8dedd8a6cb9a99de5b452a92fa2508fe1d1e2157e

</details>

<details><summary>gemma4-scalar-l28-plus4-dev10-screen-20261104</summary>

候補: scalar-l28-plus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.124057 | 12/12 | — | — |
| probe-wikitext | None | 8.501516 | 8/8 | — | — |

開始: 2026-11-05T02:13:54.569060Z。終了: 2026-11-05T02:15:06.554584Z。
- run.json SHA256: ccb86c8c3c5254d28bf9b119eeb382b75284b02dd4f3c8627b14d002490295e4
- report.json SHA256: c711a14759a9ca49cfc4b3908c151219c8cf132ab631175155b60f93506bd03d

</details>

<details><summary>gemma4-scalar-l23-minus4-dev10-screen-20261104</summary>

候補: scalar-l23-minus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 82.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.153842 | 12/12 | — | — |
| probe-wikitext | None | 8.714065 | 8/8 | — | — |

開始: 2026-11-05T02:19:00.331195Z。終了: 2026-11-05T02:21:19.208428Z。
- run.json SHA256: 232f044f4c23ded015ba20d3178a474ff852bd25fbb5ea736975dbc06c6b9456
- report.json SHA256: aabced1e0b88fa9df089a3defd549e8db8567efb9f301195c8f58e187920ceae

</details>

<details><summary>gemma4-scalar-l23-plus4-dev10-screen-20261104</summary>

候補: scalar-l23-plus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.109485 | 12/12 | — | — |
| probe-wikitext | None | 8.446254 | 8/8 | — | — |

開始: 2026-11-05T02:26:05.903513Z。終了: 2026-11-05T02:27:50.259229Z。
- run.json SHA256: 1a2169a58a7801d975b99e3e98e77f5d1e26ed44478a5cc8d6577db506fdeb94
- report.json SHA256: 3770a4c180181c74f70050c324637f633121763e30f49b986b3456eb346e0309

</details>

<details><summary>gemma4-scalar-l41-minus4-dev10-screen-20261104</summary>

候補: scalar-l41-minus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.147252 | 12/12 | — | — |
| probe-wikitext | None | 8.583271 | 8/8 | — | — |

開始: 2026-11-05T02:32:15.142279Z。終了: 2026-11-05T02:33:27.836586Z。
- run.json SHA256: 5e6ae8153f13cb6b2b2f04efefed1ff2e443901db8b170eacf22a91350fc3caf
- report.json SHA256: 0d92aac6cf36691c79edfdaea79c00cf07080034a1134dee1e546951bacc4ba8

</details>

<details><summary>gemma4-scalar-l41-plus4-dev10-screen-20261104</summary>

候補: scalar-l41-plus4。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.123217 | 12/12 | — | — |
| probe-wikitext | None | 8.565935 | 8/8 | — | — |

開始: 2026-11-05T02:37:41.289536Z。終了: 2026-11-05T02:39:02.205163Z。
- run.json SHA256: 757f72b8882f6ed00ccef65f736a19e6b2f2c6d352a3fc2970e559cd7dec0c0d
- report.json SHA256: 99fe4a3115f5004f62a289c46434f4e9d3041a7191fca5ffd9db00eeff3f8781

</details>

<details><summary>gemma4-sparse-code-margin3-dev10-screen-20261105</summary>

候補: sparse-code-margin3。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.127461 | 12/12 | — | — |
| probe-wikitext | None | 8.594061 | 8/8 | — | — |

開始: 2026-11-05T02:43:51.360331Z。終了: 2026-11-05T02:44:55.112359Z。
- run.json SHA256: a8def4bfda1aadc97370f2ca34c6b259feed88cb04886769adc188a000101806
- report.json SHA256: 73fb95a73d18e1b1f602f709fb2572b2f9612f3c8eb518076146f8ea99ed409d

</details>

<details><summary>gemma4-sparse-action-margin3-dev10-screen-20261105</summary>

候補: sparse-action-margin3。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.141939 | 12/12 | — | — |
| probe-wikitext | None | 8.560785 | 8/8 | — | — |

開始: 2026-11-05T02:49:12.838944Z。終了: 2026-11-05T02:50:23.361619Z。
- run.json SHA256: 9c086105da5da651dddde5d57ae6987eafe6aa11389a32fb437c221906b8ea32
- report.json SHA256: b1fa91cf111e0c630103cfd307569686d92d0a4eb277e7162d949ad18033e68f

</details>

<details><summary>gemma4-joint-purpose-balanced-small-dev10-screen-20261105</summary>

候補: joint-purpose-balanced-small。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 82.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.130500 | 12/12 | — | — |
| probe-wikitext | None | 8.604809 | 8/8 | — | — |

開始: 2026-11-05T02:55:09.183202Z。終了: 2026-11-05T02:56:15.132949Z。
- run.json SHA256: aca353fe39a1e41c4b821325428c32b337e576e7dff31e722c8b08acd8e25d1c
- report.json SHA256: 3049df86bdf0ecfcb4384dc2ee8558d7fcf0c94ef4c4788d832871b16cf8bbfd

</details>

<details><summary>gemma4-joint-purpose-balanced-small-dev10-full-20261105</summary>

候補: joint-purpose-balanced-small。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 67.500000 |
| math | 95.937500 |
| code | 87.777778 |
| instructions | 92.083333 |
| tools | 90.000000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 76.956522 |
| commonsense | 66.000000 |
| multilingual | 82.121212 |
| vision | 93.750000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.690000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.750000 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.863636 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.937500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.133655 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.144694 | 12/12 | — | — |
| probe-wikitext | None | 8.624612 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.650000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T02:57:52.718706Z。終了: 2026-11-05T03:08:48.346192Z。
- run.json SHA256: ada03807ff2de69163396857853b4431fc08955e147250f8854ee1fdf8d2af25
- report.json SHA256: 54c13ba6518d8c3e66eb2fc8edb93ce6777fa361e30cc043e063676b1923a708

</details>

<details><summary>gemma4-joint-purpose-balanced-small-dev10-screen-20261105-r2</summary>

候補: joint-purpose-balanced-small。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.127267 | 12/12 | — | — |
| probe-wikitext | None | 8.660205 | 8/8 | — | — |

開始: 2026-11-05T03:10:27.895208Z。終了: 2026-11-05T03:11:56.543612Z。
- run.json SHA256: fb73b9d05736b56db0e0d9ed81afd4d5face86591833b234254ebb7c29ca5b9f
- report.json SHA256: 6dab384ae28da0fe722edbf826f31c50fb9b1f9e7baa263a193d2a3eff85c61d

</details>

<details><summary>gemma4-joint-purpose-balanced-small-dev30-hard-code-20261105</summary>

候補: joint-purpose-balanced-small。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 50.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.600000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T03:13:33.023990Z。終了: 2026-11-05T03:17:56.238282Z。
- run.json SHA256: 730c03a6ce6f4c2fb4a8a2c66b7e25aaede623c864b2ff36204e1847254af894
- report.json SHA256: ff678ab755233cc9a290a637099ec5eb1339e1158e10c355cb137335144d81eb

</details>

<details><summary>gemma4-joint-purpose-balanced-small-dev10-observed-losses-repeat-20261105</summary>

候補: joint-purpose-balanced-small。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | 67.500000 |
| math | — |
| code | — |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | 65.000000 |
| multilingual | 95.454545 |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.650000 | 100/100 | — | — |

開始: 2026-11-05T03:19:33.010859Z。終了: 2026-11-05T03:20:21.623053Z。
- run.json SHA256: 7d210f89670ef407de784a6189519dcf8dd2f8769c2841a30acd7539b7ec2e91
- report.json SHA256: d13b0ac67d6f50eecd20703fa47e1f3c165bb93ef2bf70ea2e434551bfe18cb3

</details>

<details><summary>gemma4-joint-purpose-programming-set-dev10-screen-20261105</summary>

候補: joint-purpose-programming-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 73.333333 |
| tools | 90.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.733333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.146464 | 12/12 | — | — |
| probe-wikitext | None | 8.617423 | 8/8 | — | — |

開始: 2026-11-05T03:25:01.147379Z。終了: 2026-11-05T03:26:12.210607Z。
- run.json SHA256: b0ad7622805307bf23edf600f5a6a6522f7e7bf59c166d193175d067ed56b921
- report.json SHA256: 74bb17c1c34e3a2a541e212a7dc40e4d08fd75533ca1693d59c72ab0bf9137aa

</details>

<details><summary>gemma4-joint-purpose-programming-set-dev10-full-20261105</summary>

候補: joint-purpose-programming-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 91.666667 |
| expert | 81.250000 |
| facts | 65.000000 |
| math | 97.187500 |
| code | 87.777778 |
| instructions | 92.083333 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 76.956522 |
| commonsense | 66.000000 |
| multilingual | 79.469697 |
| vision | 87.500000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.625000 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.916667 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.133101 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.154607 | 12/12 | — | — |
| probe-wikitext | None | 8.621810 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.650000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.650000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T03:27:49.696749Z。終了: 2026-11-05T03:38:21.662160Z。
- run.json SHA256: 803a38664035f9d74b7c06ce4f2ec3b9326de35ee1ede710c70147166394512a
- report.json SHA256: 02cdfd8c0525ff4e38fa16939a80c5b47b66951288bcfd1be054346111ffdad3

</details>

<details><summary>gemma4-joint-purpose-programming-set-dev10-screen-20261105-r2</summary>

候補: joint-purpose-programming-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 73.333333 |
| tools | 90.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.733333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.143484 | 12/12 | — | — |
| probe-wikitext | None | 8.615194 | 8/8 | — | — |

開始: 2026-11-05T03:40:09.552293Z。終了: 2026-11-05T03:41:20.294406Z。
- run.json SHA256: 7d35f5d7bbdbf0b83be11f7b309a933c8a560650d490ed8d7f71e361e1a752ef
- report.json SHA256: 87d4f68bb80fc085052e55db9af9e4e6bb49b6074725b1c57d292cbed84fb901

</details>

<details><summary>gemma4-joint-purpose-programming-set-dev30-hard-code-20261105</summary>

候補: joint-purpose-programming-set。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 55.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T03:43:01.542527Z。終了: 2026-11-05T03:47:45.349030Z。
- run.json SHA256: 21ddb29ac6544cf7c091c82603dac15685bffaea7fb1370dea9e1719b251bb06
- report.json SHA256: a84af7fe972fcb40a3880ba5572535c2b111d03a676ba2536b8525191e974e4f

</details>

<details><summary>gemma4-joint-purpose-algorithm-set-dev10-screen-20261105</summary>

候補: joint-purpose-algorithm-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.144520 | 12/12 | — | — |
| probe-wikitext | None | 8.629085 | 8/8 | — | — |

開始: 2026-11-05T03:52:24.327537Z。終了: 2026-11-05T03:53:33.557965Z。
- run.json SHA256: 6d7769c496ed9b141d073d1bdd848986525853c088c7d0a852dbfb3e543b75b1
- report.json SHA256: 1c038507bb1f21b9590c1d308fc5105c11ad0cae96d865c8fb4fdfbf768630e1

</details>

<details><summary>gemma4-joint-purpose-workflow-language-dev10-screen-20261105</summary>

候補: joint-purpose-workflow-language。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 82.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.123375 | 12/12 | — | — |
| probe-wikitext | None | 8.579147 | 8/8 | — | — |

開始: 2026-11-05T03:58:32.568426Z。終了: 2026-11-05T03:59:47.970832Z。
- run.json SHA256: 95df0af8a76bfa6b5a43441c40f87b821787cb2ee765a0ac776324ea22efb142
- report.json SHA256: 3c9f3babd61c283877cf88ddc595c4d93f986ba48eab737bc735d40e239b350f

</details>

<details><summary>gemma4-joint-purpose-algorithm-set-dev30-hard-code-20261105</summary>

候補: joint-purpose-algorithm-set。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 53.365079 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.650000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.417302 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T04:03:38.178008Z。終了: 2026-11-05T04:08:59.868185Z。
- run.json SHA256: 60ded72ff342eeb7228cb858c6b3b3b200be3e1eea2178c969aec69549486e8f
- report.json SHA256: 189fcadfb45afd5353edd1ae95da7a72dbb10b3675a336598b4f4b1bf10eeba9

</details>

<details><summary>gemma4-sparse-code-action6-dev10-screen-20261105</summary>

候補: sparse-code-action6。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 82.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.134008 | 12/12 | — | — |
| probe-wikitext | None | 8.589391 | 8/8 | — | — |

開始: 2026-11-05T04:13:42.317863Z。終了: 2026-11-05T04:15:02.691206Z。
- run.json SHA256: b5bd10e00906bb54b6f9473e60992429b58e7958bf190c04342a3614aa3cddee
- report.json SHA256: 92a8c03a98e12dfee9317d618d806b3fba81e10d5d04ca38f2f0151a83675dc6

</details>

<details><summary>gemma4-joint-purpose-balanced-medium-dev10-screen-20261105</summary>

候補: joint-purpose-balanced-medium。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.140795 | 12/12 | — | — |
| probe-wikitext | None | 8.632858 | 8/8 | — | — |

開始: 2026-11-05T04:20:04.376388Z。終了: 2026-11-05T04:21:12.963360Z。
- run.json SHA256: 14f29c18d3bf716dcb448ee44db5167cd479e38d1f2351de72752a90acf38a8f
- report.json SHA256: e599104fadc27c0607a2b74983a9ae66fad6e9258bab9f315b69b8f471eca4ac

</details>

<details><summary>gemma4-joint-purpose-balanced-medium-dev10-screen-20261105-r2</summary>

候補: joint-purpose-balanced-medium。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 73.333333 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.733333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.140840 | 12/12 | — | — |
| probe-wikitext | None | 8.636447 | 8/8 | — | — |

開始: 2026-11-05T04:22:53.635549Z。終了: 2026-11-05T04:24:07.961799Z。
- run.json SHA256: a7456b47e95508e17e6f6f3be258ac434eaaaf4e9a8edf5399431794ef845ef4
- report.json SHA256: 209dd3d8b67ff4a5551d49c7337d3a0968baea4987776079f2d0b88cfd18527d

</details>

<details><summary>gemma4-joint-purpose-chat-set-dev10-screen-20261105</summary>

候補: joint-purpose-chat-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.166410 | 12/12 | — | — |
| probe-wikitext | None | 8.807983 | 8/8 | — | — |

開始: 2026-11-05T04:34:17.118688Z。終了: 2026-11-05T04:35:29.011209Z。
- run.json SHA256: 63ea2ef75cab1177bbdd6e6370901770a561620be12d32d6d26681b3d95ce960
- report.json SHA256: 5b63cd5155080b6b9f0c750eb69920f48418ad6e7f466d30c99a8ad171112539

</details>

<details><summary>gemma4-joint-purpose-chat-set-dev10-full-20261105</summary>

候補: joint-purpose-chat-set。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 75.000000 |
| facts | 67.500000 |
| math | 97.187500 |
| code | 87.777778 |
| instructions | 88.750000 |
| tools | 85.000000 |
| agents | 50.000000 |
| long-context | 87.500000 |
| reasoning | 69.782609 |
| commonsense | 67.333333 |
| multilingual | 80.984848 |
| vision | 93.750000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.695652 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.750000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.625000 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.937500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.172591 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.295084 | 12/12 | — | — |
| probe-wikitext | None | 8.863381 | 8/8 | — | — |
| ruler | long-context | 0.750000 | 8/8 | 0.250000 | 0.250000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.680000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T04:37:09.818019Z。終了: 2026-11-05T04:47:26.749315Z。
- run.json SHA256: a44bf32baaebf79f3847edaa9455bbacc3ba58e8d8f9852fe230fddfc49079da
- report.json SHA256: e89d5d64d0eda67ce38df0a15260c518314cd9d4212b910a8e9ae87b32b61d0a

</details>

<details><summary>gemma4-joint-purpose-chat-set-dev30-hard-code-20261105</summary>

候補: joint-purpose-chat-set。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 57.650794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T04:49:07.508509Z。終了: 2026-11-05T04:54:10.665014Z。
- run.json SHA256: b4594792ee237e03da7bce9d34ad19c7aa3ca9e0711e39f990231cad7809907c
- report.json SHA256: ab9bff0093ce756122c24b7bc64db5bd04420bdc44447ccb6cc715f903aa2db6

</details>

<details><summary>gemma4-finalnorm-minus1pct-dev10-screen-20261105</summary>

候補: finalnorm-minus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 90.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.129108 | 12/12 | — | — |
| probe-wikitext | None | 8.530616 | 8/8 | — | — |

開始: 2026-11-05T05:34:47.906314Z。終了: 2026-11-05T05:36:02.928808Z。
- run.json SHA256: 325a7276a0e9b2777886fcbb5cb96a5c449b2b7879af3af0262cda75a5fec48f
- report.json SHA256: 56cd36d3c23d1938ecac555cdc2af439dc31c4e9f0d2e4a3413e7bc59c6390b1

</details>

<details><summary>gemma4-finalnorm-plus1pct-dev10-screen-20261105</summary>

候補: finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 90.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.147231 | 12/12 | — | — |
| probe-wikitext | None | 8.636700 | 8/8 | — | — |

開始: 2026-11-05T05:40:39.505329Z。終了: 2026-11-05T05:41:57.771503Z。
- run.json SHA256: 7c7e1e9cbe29d96c5ed3605e91e39b6f5efabc983bc300a7c22baa3a4b2ab8cb
- report.json SHA256: 0286ad2d6d266c4b8ba9d7707b000a038476b1f33ec1f89cbcc2b5e4f96bfc09

</details>

<details><summary>gemma4-finalnorm-plus1pct-dev10-screen-20261105-r2</summary>

候補: finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.142809 | 12/12 | — | — |
| probe-wikitext | None | 8.614988 | 8/8 | — | — |

開始: 2026-11-05T05:43:33.844473Z。終了: 2026-11-05T05:44:37.739233Z。
- run.json SHA256: 707eb7f3241e6ec6170324852bd3c810bccc5a59e0d83826481747ff8bf7457b
- report.json SHA256: eb879a54257a596dfc679452527b5886ff10a594fdb8f33b2bb000c5b52b46a9

</details>

<details><summary>gemma4-finalnorm-minus1pct-dev10-screen-20261105-r2</summary>

候補: finalnorm-minus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 93.333333 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.125676 | 12/12 | — | — |
| probe-wikitext | None | 8.524182 | 8/8 | — | — |

開始: 2026-11-05T05:48:24.950318Z。終了: 2026-11-05T05:49:42.094765Z。
- run.json SHA256: 7a79e2df3f37678ea9663e88e222e7d3301da41c12256903fce68012ae9a2e3b
- report.json SHA256: 6b265f663ebf9c0ffbd93cb6204f3fc3f35cbb4eac241ba05ed6668b869c010e

</details>

<details><summary>gemma4-finalnorm-plus1pct-dev10-full-20261105</summary>

候補: finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 70.000000 |
| math | 95.625000 |
| code | 87.777778 |
| instructions | 95.416667 |
| tools | 90.000000 |
| agents | 50.000000 |
| long-context | 75.127943 |
| reasoning | 76.956522 |
| commonsense | 67.000000 |
| multilingual | 79.191919 |
| vision | 93.750000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.937500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.145546 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.204176 | 12/12 | — | — |
| probe-wikitext | None | 8.672178 | 8/8 | — | — |
| ruler | long-context | 0.750000 | 8/8 | 0.250000 | 0.250000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.700000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.680000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T05:53:42.013483Z。終了: 2026-11-05T06:05:08.946029Z。
- run.json SHA256: f90f6d7be67488a7d71d4d283dce29776ccf6f37412db17f54fa546169f16b32
- report.json SHA256: c2d6b041146035da5444095fd6f9eea2390d4170fdfb7eca7d2f786095074ca3

</details>

<details><summary>gemma4-finalnorm-plus1pct-dev30-hard-code-20261105</summary>

候補: finalnorm-plus1pct。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 58.206349 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.414127 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T06:10:39.806898Z。終了: 2026-11-05T06:15:19.359625Z。
- run.json SHA256: a7e5df263cfebc7547df6fc7a85b0287a09bf72faaf9d0af7883e6dc34766538
- report.json SHA256: 213dda28f6fde0b8931d646375e26d2d3245f3c387e1ccec952330738e4f874a

</details>

<details><summary>gemma4-donor-text-c1-dev10-screen-20261105</summary>

候補: donor-text-c1。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.000000 |
| instructions | 93.333333 |
| tools | 80.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.800000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.900000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.500000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 2.989850 | 12/12 | — | — |
| probe-wikitext | None | 4.012250 | 8/8 | — | — |

開始: 2026-11-05T06:26:08.514349Z。終了: 2026-11-05T06:27:21.532383Z。
- run.json SHA256: e6fe22ca17727236eb18b7705577e1feabc2c82fab5f06126b99564f2a2c8d07
- report.json SHA256: 9e5b0bc4627522845c5d050d1c552762e1a8d4d0392fb690da1fbc3bf5a0a709

</details>

<details><summary>gemma4-donor-text-c1-dev30-hard-code-20261105</summary>

候補: donor-text-c1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 55.984127 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.369683 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T06:34:03.855894Z。終了: 2026-11-05T06:37:15.803777Z。
- run.json SHA256: a46c6bad418080141291ebd8af51647b36e6e21a85d7dc6295a5830c5e4e325a
- report.json SHA256: cc4e3a514f7e681e6be73bab1d5980554c52c62d0838251aff39396478fc6b21

</details>

<details><summary>gemma4-donor-text-c1-dev10-full-20261105</summary>

候補: donor-text-c1。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 67.500000 |
| math | 97.500000 |
| code | 74.444444 |
| instructions | 88.750000 |
| tools | 82.500000 |
| agents | 0.000000 |
| long-context | 70.732668 |
| reasoning | 74.782609 |
| commonsense | 72.666667 |
| multilingual | 77.676768 |
| vision | 100.000000 |
| safety | 94.444444 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.690000 | 100/100 | — | — |
| bbeh | reasoning | 0.695652 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.760000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.400000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.863636 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.414653 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 2.978367 | 12/12 | — | — |
| probe-wikipedia-ja | None | 3.935821 | 12/12 | — | — |
| probe-wikitext | None | 4.009537 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 0.000000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.730000 | 100/100 | — | — |
| xstest | safety | 0.944444 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T06:38:52.231379Z。終了: 2026-11-05T06:47:17.422152Z。
- run.json SHA256: 411290d4a7f39c8c5d7db6602db795da12e6dcf1faf14430a5da530e8cc49e88
- report.json SHA256: 493577602b6b75818dd4995d73315c02178f494dfa530da1cfee56c358331380

</details>

<details><summary>gemma4-joint-programming-finalnorm-plus1pct-dev10-screen-20261105</summary>

候補: joint-programming-finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.141089 | 12/12 | — | — |
| probe-wikitext | None | 8.620399 | 8/8 | — | — |

開始: 2026-11-05T06:57:59.548050Z。終了: 2026-11-05T06:59:18.617835Z。
- run.json SHA256: c52bab237e61b7b0eac4061074d11df897fc9ad9f9278c9d649060e789a5d1c1
- report.json SHA256: 4e9c7dd036cc37ffe1754ae03750920a7e83cec926e542e4ad81c0badd90d726

</details>

<details><summary>gemma4-joint-programming-finalnorm-plus1pct-dev30-hard-code-20261105</summary>

候補: joint-programming-finalnorm-plus1pct。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 57.650794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T07:00:56.374441Z。終了: 2026-11-05T07:05:38.611261Z。
- run.json SHA256: 342d86d9083ca6106be0a0e7428e873749f67f357451870f95147ce29a63cb3b
- report.json SHA256: d25e617935c4c6f4a9a4f2c9269bd45702aa5dff5c8f87fbc0f45745214d3b7b

</details>

<details><summary>gemma4-joint-programming-finalnorm-plus1pct-dev10-full-20261105</summary>

候補: joint-programming-finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 70.000000 |
| math | 95.625000 |
| code | 84.444444 |
| instructions | 88.750000 |
| tools | 85.000000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 76.956522 |
| commonsense | 67.333333 |
| multilingual | 79.191919 |
| vision | 93.750000 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.875000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.041667 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.937500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.136806 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.219972 | 12/12 | — | — |
| probe-wikitext | None | 8.587383 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.700000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.670000 | 100/100 | — | — |
| xstest | safety | 0.888889 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T07:07:12.191000Z。終了: 2026-11-05T07:17:48.356618Z。
- run.json SHA256: a2fa54e01202f117e66a83768f3a054ce02b0c13d62646153401177fec42cfb1
- report.json SHA256: 33535d710a7164f3d03ef9429b40ab82fa440a30b83a2e00ce1e0f0fa3ffc81a

</details>

<details><summary>gemma4-finalnorm-plus0p5pct-dev10-screen-20261105</summary>

候補: finalnorm-plus0p5pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.131553 | 12/12 | — | — |
| probe-wikitext | None | 8.584090 | 8/8 | — | — |

開始: 2026-11-05T07:48:30.127744Z。終了: 2026-11-05T07:49:49.695703Z。
- run.json SHA256: faf67516c0e2b6a8589d6ff218e6aaea3e088e32574353e8cecbf2f4349d022d
- report.json SHA256: 20d6577e0ca7588691aee0d8e0985ce9a07ac51f3adc82c2475a46aba5899e15

</details>

<details><summary>gemma4-finalnorm-plus0p5pct-dev30-hard-code-20261105</summary>

候補: finalnorm-plus0p5pct。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 55.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T07:51:26.094822Z。終了: 2026-11-05T07:56:23.185798Z。
- run.json SHA256: 8d0afb4f5d258c59d46327aebc2c0ce181bb9fda69e47e15ab473bd79edaf72e
- report.json SHA256: 4b4cc8fef78398a6fdadacaa2779d9e2be9ec3c7b8bde023af22e16fa67e054e

</details>

<details><summary>gemma4-finalnorm-plus0p5pct-dev10-full-20261105</summary>

候補: finalnorm-plus0p5pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 67.500000 |
| math | 98.750000 |
| code | 75.555556 |
| instructions | 92.083333 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 64.782609 |
| commonsense | 66.666667 |
| multilingual | 82.095960 |
| vision | 81.250000 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.695652 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.900000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.708333 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.666667 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.600000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.139274 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.153421 | 12/12 | — | — |
| probe-wikitext | None | 8.614346 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.888889 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T07:57:56.321706Z。終了: 2026-11-05T08:08:31.942737Z。
- run.json SHA256: f98be3b41cc527397d2107052d5bc2cfb7f0f6c32ede52927e5df8a993ef3627
- report.json SHA256: 5acff2ae154c93c00ac6b6505d87c7cf947571effcc4d0afb2fab4b825d5ec34

</details>

<details><summary>gemma4-donor-text-c0p25-dev10-screen-20261105</summary>

候補: donor-text-c0p25。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 73.333333 |
| tools | 90.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.733333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 3.652861 | 12/12 | — | — |
| probe-wikitext | None | 7.206249 | 8/8 | — | — |

開始: 2026-11-05T08:23:41.878984Z。終了: 2026-11-05T08:24:53.909923Z。
- run.json SHA256: 146a220f7d7d0b77f38f347ae81dd3d207edde70f4a6eaeba215147605a5534d
- report.json SHA256: e1360fee7cb13407e0b727cd4d34b7e88de6145a7ce18414a25f6d3b33664123

</details>

<details><summary>gemma4-donor-text-c0p25-dev30-hard-code-20261105</summary>

候補: donor-text-c0p25。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 50.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.650000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.353016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T08:26:32.511964Z。終了: 2026-11-05T08:30:33.935298Z。
- run.json SHA256: 09572ae59628286278eaf59b0e47bbe3abfa312cfe267583e8ab0eb7300348b6
- report.json SHA256: a3ee34514d0f097cce3a82ffcdafc857eee08b11bcf975ed03a1652bb9059fc4

</details>

<details><summary>gemma4-donor-text-c0p25-dev10-full-20261105</summary>

候補: donor-text-c0p25。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 86.666667 |
| expert | 87.500000 |
| facts | 65.000000 |
| math | 98.750000 |
| code | 84.444444 |
| instructions | 92.083333 |
| tools | 85.000000 |
| agents | 50.000000 |
| long-context | 91.371681 |
| reasoning | 76.956522 |
| commonsense | 66.000000 |
| multilingual | 79.343434 |
| vision | 87.500000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.650000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.863636 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.866667 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.827434 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 3.648467 | 12/12 | — | — |
| probe-wikipedia-ja | None | 6.122187 | 12/12 | — | — |
| probe-wikitext | None | 7.190865 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.650000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.690000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T08:32:06.368345Z。終了: 2026-11-05T08:40:51.731707Z。
- run.json SHA256: 12c4a6dad070aa549c6370445b1e78f4bc1b767a142d8d725346937afaee689b
- report.json SHA256: 5fce40c1fcbada02a5372561284d07e56f56b4799556e2480b9bc9e0f76df14b

</details>

<details><summary>gemma4-joint-donor-quarter-finalnorm-plus1pct-dev10-core-long-20261105</summary>

候補: joint-donor-quarter-finalnorm-plus1pct。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 86.666667 |
| tools | 87.500000 |
| agents | — |
| long-context | 85.121681 |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.827434 | 4/4 | 0.000000 | 0.000000 |
| probe-code | None | 3.644979 | 12/12 | — | — |
| probe-wikitext | None | 7.224283 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |

開始: 2026-11-05T09:21:49.468996Z。終了: 2026-11-05T09:27:41.956290Z。
- run.json SHA256: 7a4ff8878e024ab05531f1823222937175e77f6eafd336b0eb66f51e4a960cb0
- report.json SHA256: db4563dce4b72316dca5e414f1bad0cf759bcb44a5594866297c0e27d947ac49

</details>

<details><summary>gemma4-frozen-c5be8132c454-test10-20261105</summary>

候補: finalnorm-plus1pct。completed、test/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 93.333333 |
| expert | 87.500000 |
| facts | 65.000000 |
| math | 97.187500 |
| code | 91.111111 |
| instructions | 82.083333 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 89.445983 |
| reasoning | 83.478261 |
| commonsense | 66.666667 |
| multilingual | 86.818182 |
| vision | 81.250000 |
| safety | 91.666667 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.720000 | 100/100 | — | — |
| bbeh | reasoning | 0.869565 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.620000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.666667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.750000 | 24/24 | 0.041667 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.900000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.933333 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.788920 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 3.053613 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.181534 | 12/12 | — | — |
| probe-wikitext | None | 8.175101 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.650000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.916667 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T09:40:00.743969Z。終了: 2026-11-05T09:47:42.917927Z。
- run.json SHA256: 44b4bd438c851d7aa0a10c85313f95f34054532fbd4062fe82381e4cf483bde5
- report.json SHA256: 599c0693127ee81755511f7263568c018e92c02905b6843ced93952e8cf86dcb

</details>

<details><summary>gemma4-base-observed-dev10-20261106</summary>

候補: O 原モデル・観測ON。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 70.000000 |
| math | 98.750000 |
| code | 87.777778 |
| instructions | 88.750000 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 81.343633 |
| reasoning | 71.956522 |
| commonsense | 67.333333 |
| multilingual | 80.984848 |
| vision | 81.250000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.650000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.625000 | 24/24 | 0.041667 | 0.041667 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.751873 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.137872 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.139389 | 12/12 | — | — |
| probe-wikitext | None | 8.601027 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.700000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T16:05:18.901106Z。終了: 2026-11-05T16:15:33.029372Z。
- run.json SHA256: a65a8fb643205d30813d56c30a134a0da1e54f46b26b223a5954acc322c0527e
- report.json SHA256: 2a8604a7ea07061359e17ee8f488d29421a40fc6d04623b361c525de45eff970

</details>

<details><summary>gemma4-resume-hs0020-dev10-screen-20261106</summary>

候補: hs0020。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.132883 | 12/12 | — | — |
| probe-wikitext | None | 8.593194 | 8/8 | — | — |

開始: 2026-11-05T16:38:53.996360Z。終了: 2026-11-05T16:40:26.269657Z。
- run.json SHA256: 1b144021bef0a1bfe9a38e1bc7eb4d023a69b7e86dec0bb4aff07eaf510c794b
- report.json SHA256: 8f5e792891296101293c00500eaad6ebbb3c616a4b65df15a54a2c181938acd3

</details>

<details><summary>gemma4-resume-h0397-dev10-screen-20261106</summary>

候補: 0397。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 80.000000 |
| tools | 87.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.152029 | 12/12 | — | — |
| probe-wikitext | None | 8.719063 | 8/8 | — | — |

開始: 2026-11-05T16:59:38.497404Z。終了: 2026-11-05T17:00:40.878599Z。
- run.json SHA256: 05c62d5c93263e03c3c14f326e56f7a1fb00766fc1adc4bac1acb63e0851ad51
- report.json SHA256: 90dade94cce67475617ae1766c9024865011045c6031dbaf46427d76f7eb5d65

</details>

<details><summary>gemma4-resume-h0397-dev30-hard-code-20261106</summary>

候補: 0397。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 57.650794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.353016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:02:25.356045Z。終了: 2026-11-05T17:08:42.417085Z。
- run.json SHA256: 0427e32d8312e469290963840a837761076a874b9af95fef0337fae18803c242
- report.json SHA256: 78dde5a5dc92dbdbb0c2f2de8f4349d978d35bc4cf805e4fa41aa6c12b7080b8

</details>

<details><summary>gemma4-resume-h0397-dev10-full-20261106</summary>

候補: 0397。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 88.333333 |
| expert | 87.500000 |
| facts | 67.500000 |
| math | 98.750000 |
| code | 87.777778 |
| instructions | 90.833333 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 76.956522 |
| commonsense | 67.000000 |
| multilingual | 80.707071 |
| vision | 93.750000 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.950000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.883333 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.937500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.146247 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.239116 | 12/12 | — | — |
| probe-wikitext | None | 8.728829 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.888889 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:15:04.517812Z。終了: 2026-11-05T17:25:41.107777Z。
- run.json SHA256: bf00b8b4d2a650b3bb350d7f4f78e6f95fffebbe1c56c548c841573a3f339f6e
- report.json SHA256: e612ed12f6995836f6572b2198706232d7f83d504021c248a8c399d44e6055c8

</details>

<details><summary>gemma4-resume-h0397-dev30-lcb-repeat-20261106</summary>

候補: 0397。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 75.000000 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:27:04.609006Z。終了: 2026-11-05T17:29:30.474816Z。
- run.json SHA256: f485107515facf99bb1cd3f590bed4cef6afc8fa6726d7357ca8927f93a766e9
- report.json SHA256: fb30187681af93299308ad027788ee1cd40c9bf6863dde99af9c34421b3a1b2e

</details>

<details><summary>gemma4-resume-base-dev30-lcb-matched-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.000000 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:35:28.521347Z。終了: 2026-11-05T17:38:02.477535Z。
- run.json SHA256: af85522a7b70973e26152e3fca9af343c233e58533d81b4d9d15c8cc15f4a02f
- report.json SHA256: ef1b15358e5ab23b66f9298f8a306f22f6c0fad743983d4938e91525635ea1d3

</details>

<details><summary>gemma4-resume-h0485-dev10-screen-20261106</summary>

候補: 0485。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.161173 | 12/12 | — | — |
| probe-wikitext | None | 8.761959 | 8/8 | — | — |

開始: 2026-11-05T17:44:25.389735Z。終了: 2026-11-05T17:45:33.242738Z。
- run.json SHA256: df5da90fdfea00ebb6d55689f9925dc5e788d54b62c173ff1d1567f7750d4d64
- report.json SHA256: cfff09c4a3bc547a37c6147b6b4815551176f85373b459c20e90c29c1757e043

</details>

<details><summary>gemma4-resume-h0485-dev30-hard-code-20261106</summary>

候補: 0485。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 60.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:48:17.931895Z。終了: 2026-11-05T17:53:26.705133Z。
- run.json SHA256: 8bef8ec7502f050419a4d5bf4c55631f3dcb1eb025d9d1a182495a9becd8b0b7
- report.json SHA256: 251da96c5b951cf026edec90362b6dcd342dcee6e79fb89c8f861cc42f14181b

</details>

<details><summary>gemma4-resume-h0485-dev10-full-20261106</summary>

候補: 0485。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 67.500000 |
| math | 95.937500 |
| code | 84.444444 |
| instructions | 92.083333 |
| tools | 87.500000 |
| agents | 50.000000 |
| long-context | 93.750000 |
| reasoning | 71.956522 |
| commonsense | 66.000000 |
| multilingual | 83.762626 |
| vision | 100.000000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.875000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.708333 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.149892 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.226455 | 12/12 | — | — |
| probe-wikitext | None | 8.775992 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.640000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-05T17:57:50.307291Z。終了: 2026-11-05T18:08:10.536073Z。
- run.json SHA256: e52902c59888124c9c0162d69c4f84646dcc6e0b1b98f2403c21d9d6c4762974
- report.json SHA256: 67a1757deab4ec89ba23524bb9ad07e5bee90a9e63732f9cbe883446d7fe30f3

</details>

<details><summary>gemma4-resume-localgc-dev10-screen-20261106</summary>

候補: localG-C。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 90.000000 |
| instructions | 86.666667 |
| tools | 85.000000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.137327 | 12/12 | — | — |
| probe-wikitext | None | 8.621577 | 8/8 | — | — |

開始: 2026-11-05T18:30:58.814983Z。終了: 2026-11-05T18:32:10.717861Z。
- run.json SHA256: a202c6aa57ea159af9620f19aa66dba2fe51068c2f1e556c44a647aba8e541d8
- report.json SHA256: fd70dc61e72d11f43eb45a3dc071598a3f9348d1b7f03ca8acd16b038d728ca6

</details>

<details><summary>gemma4-resume-localgc-dev30-hard-code-20261106</summary>

候補: localG-C。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 55.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T18:32:14.433003Z。終了: 2026-11-05T18:36:40.209967Z。
- run.json SHA256: 26f4901fc5d4fb1f070fbafe68cd31471c57d08add2f1ebe7fce37ab856782a4
- report.json SHA256: 0da9ed7e84d04b77054b2c02f58b750b6dbe4ec8f0a6fb2b77d3363057a0677b

</details>

<details><summary>gemma4-resume-composed-dev30-hard-code-20261106</summary>

候補: 0397＋localG-C。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 60.150794 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-05T18:54:08.764879Z。終了: 2026-11-05T18:58:51.863001Z。
- run.json SHA256: 2cfb380f5135a17da20fa6e9efecc645802fc7fa6e185294dd0e94a04530e0d3
- report.json SHA256: 4279a102620051e8cf83f9957f5f3fe780faf17de2d6c6eb880c025605f51635

</details>

<details><summary>gemma4-resume-composed-dev10-screen-20261106</summary>

候補: 0397＋localG-C。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 85.000000 |
| instructions | 93.333333 |
| tools | 82.500000 |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.933333 | 15/15 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.167710 | 12/12 | — | — |
| probe-wikitext | None | 8.823593 | 8/8 | — | — |

開始: 2026-11-05T18:58:55.095928Z。終了: 2026-11-05T19:00:02.074773Z。
- run.json SHA256: 7f4149a4bd6d7cbee32473563086db1f6070df111068a0283086fefc2115e83b
- report.json SHA256: e141d64aafecfc902b90d09570def6a9791b67a8d2f2bbfb781cb0d4294d8538

</details>

<details><summary>gemma4-confirmation-base-1491-20261106</summary>

候補: O 原モデル。partial、test/tier30、manifest SHA256: 52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 84.873950 |
| expert | 90.625000 |
| facts | 78.750000 |
| math | 77.222222 |
| code | 77.428571 |
| instructions | 93.035714 |
| tools | 82.500000 |
| agents | 62.500000 |
| long-context | 79.915784 |
| reasoning | 79.130435 |
| commonsense | 63.500000 |
| multilingual | 78.525641 |
| vision | 78.125000 |
| safety | 93.055556 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.750000 | 8/8 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.555000 | 200/200 | — | — |
| bbeh | reasoning | 0.782609 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.825000 | 80/80 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.906250 | 32/32 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.665000 | 200/200 | — | — |
| hmmt-feb-2026 | math | 0.666667 | 9/9 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.885714 | 35/35 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 80/80 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.750000 | 48/48 | 0.000000 | 0.000000 |
| livecodebench | code | 0.857143 | 14/14 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.848739 | 119/119 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.820513 | 39/39 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.781250 | 32/32 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.697474 | 12/12 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| probe-code | None | 3.660257 | 28/28 | — | — |
| probe-wikipedia-ja | None | 7.414620 | 28/28 | — | — |
| probe-wikitext | None | 8.900103 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| scicode | code | 0.540000 | 10/10 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 4/4 | — | — |
| terminal-bench | agents | 1.000000 | 3/4 | — | — |
| triviaqa | facts | 0.787500 | 80/80 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.685000 | 200/200 | — | — |
| xstest | safety | 0.930556 | 72/72 | 0.000000 | 0.000000 |

開始: 2026-11-05T19:20:13.730302Z。終了: 2026-11-05T20:00:35.074834Z。
- run.json SHA256: bd43fef4461d22b35e93b83786d2c202d9ea7ce4d8c82d8cd704237eff883169
- report.json SHA256: 4263492b15f698f651c4d279eef300c0bb3267e797b3876555b99f2651f0c228

</details>

<details><summary>gemma4-confirmation-h0397-1490-20261106</summary>

候補: 0397。completed、test/tier30、manifest SHA256: 52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 87.394958 |
| expert | 90.625000 |
| facts | 77.500000 |
| math | 80.000000 |
| code | 82.547619 |
| instructions | 95.267857 |
| tools | 81.250000 |
| agents | 62.500000 |
| long-context | 78.475667 |
| reasoning | 79.456522 |
| commonsense | 63.000000 |
| multilingual | 78.285256 |
| vision | 75.000000 |
| safety | 93.055556 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 0.750000 | 8/8 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.555000 | 200/200 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.812500 | 80/80 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.906250 | 32/32 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.655000 | 200/200 | — | — |
| hmmt-feb-2026 | math | 0.750000 | 9/9 | 0.000000 | 0.000000 |
| humaneval-plus | code | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.942857 | 35/35 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.962500 | 80/80 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.770833 | 48/48 | 0.000000 | 0.000000 |
| livecodebench | code | 0.928571 | 14/14 | 0.000000 | 0.000000 |
| longbench-v2 | long-context | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| math-500 | math | 0.900000 | 40/40 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.873950 | 119/119 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.794872 | 39/39 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.750000 | 32/32 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.616770 | 12/12 | 0.000000 | 0.000000 |
| musr | reasoning | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| probe-code | None | 3.691530 | 28/28 | — | — |
| probe-wikipedia-ja | None | 7.471774 | 28/28 | — | — |
| probe-wikitext | None | 9.084036 | 8/8 | — | — |
| ruler | long-context | 0.937500 | 16/16 | 0.062500 | 0.062500 |
| scicode | code | 0.573333 | 10/10 | 0.000000 | 0.000000 |
| swe-bench-pro | agents | 0.250000 | 4/4 | — | — |
| terminal-bench | agents | 1.000000 | 3/3 | — | — |
| triviaqa | facts | 0.775000 | 80/80 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.680000 | 200/200 | — | — |
| xstest | safety | 0.930556 | 72/72 | 0.000000 | 0.000000 |

開始: 2026-11-05T20:14:55.005025Z。終了: 2026-11-05T20:35:20.644781Z。
- run.json SHA256: 252913840440b86ac5cdbeeb386f885139c46176b3063ed01b25ea6f6f2bcbfb
- report.json SHA256: 6397a72dd2b3702e98412f2163a5c4a769bdde9b68c4f21bff484794f178607b

</details>

<details><summary>gemma4-code90-0397-dev30-code-20261106</summary>

候補: 0397。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.075397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T02:09:46.710151Z。終了: 2026-11-06T02:14:40.538510Z。
- run.json SHA256: 424ef3615d4ffc06ac53b65e7024d82a178394dbc693cf005a60337f19e4d6a8
- report.json SHA256: 9bdf4c40fe85ab44a15d28b3e8daaaab6f997d52b4b31f39d9c04fc9cb889081

</details>

<details><summary>gemma4-code90-h0397-control-c32-dev30-code-20261106</summary>

候補: 0397。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.492063 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.369683 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:34:18.210102Z。終了: 2026-11-06T03:40:13.540063Z。
- run.json SHA256: 654bfe1af790f0ae1c2d70574248b1b1c5522defd22c3f89d1df222fbfe45a41
- report.json SHA256: bee53c62a97344b6576422867a9a7b6c39283019bd2e901930e4296d6893a63f

</details>

<details><summary>gemma4-code90-native-contracts-c32-dev30-code-20261106</summary>

候補: B native-contracts。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 73.408730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.369683 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:34:18.404270Z。終了: 2026-11-06T03:40:21.921936Z。
- run.json SHA256: b5f2485b7088720a05ac5d80af1646997877fdec06f203daa48a0975eac7e79e
- report.json SHA256: 3cc2a707b06a316a02284785a3f648910b5fcfca7c90fcb6b7615cad886fc1d3

</details>

<details><summary>gemma4-code90-joint-c32-dev30-code-20261106</summary>

候補: G native＋matrix。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.769841 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.380794 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:34:18.411984Z。終了: 2026-11-06T03:40:09.816114Z。
- run.json SHA256: ea5f5241e7110bfbbc41799638d951aeddd03d5a8bb473f509f2c0ab9f0da944
- report.json SHA256: c07db3d608dbc42d8d31d585af6ca5cff7d6d21fe262ec0c086ed41cb04b325d

</details>

<details><summary>gemma4-code90-matrix-balanced-c32-dev30-code-20261106</summary>

候補: F matrix-balanced。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 67.575397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.600000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:34:18.417403Z。終了: 2026-11-06T03:40:11.045159Z。
- run.json SHA256: f1763b976bed299c4b68421359ec294cefcc9540ae8379751f3298d8e043c40f
- report.json SHA256: bc6dd054e338acfac4cb4ac2ddb56f7582ba2ac8e774b70af38c134ebb0ab37c

</details>

<details><summary>gemma4-code90-h0397-control-c32-known64-r1-20261106</summary>

候補: 0397。completed、test/tier30、manifest SHA256: 52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 77.011905 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.900000 | 20/20 | 0.000000 | 0.000000 |
| livecodebench | code | 0.857143 | 14/14 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.523333 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:44:49.857293Z。終了: 2026-11-06T03:54:10.709350Z。
- run.json SHA256: 855b9c5d689658a5f615d6b5dcc766b2766b96048676ff959c1e3695170ef98e
- report.json SHA256: 613b909c5075b8a0670188b9332f57755a95d2eae1ea7690d8e82329e5fb1c3b

</details>

<details><summary>gemma4-code90-native-contracts-c32-known64-r1-20261106</summary>

候補: B native-contracts。completed、test/tier30、manifest SHA256: 52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 76.476190 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| livecodebench | code | 0.785714 | 14/14 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.523333 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T03:44:49.882988Z。終了: 2026-11-06T03:54:07.769360Z。
- run.json SHA256: d3590449fc23fa0d0759661347d6e7a41f443e9df93891e38cea89ac872c5701
- report.json SHA256: cb6bf765a09a82c7bee1810d4029f6d8962e5b9c444cf1b9aa950f5e64b02afa

</details>

<details><summary>gemma4-code90-original-base-c32-dev10-full14-20261106</summary>

候補: O 原モデル。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 88.333333 |
| expert | 81.250000 |
| facts | 70.000000 |
| math | 98.750000 |
| code | 87.777778 |
| instructions | 88.750000 |
| tools | 85.000000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 79.130435 |
| commonsense | 66.666667 |
| multilingual | 82.095960 |
| vision | 81.250000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.782609 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.800000 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.708333 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.883333 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.130139 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.152233 | 12/12 | — | — |
| probe-wikitext | None | 8.495862 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.700000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-06T04:07:38.583047Z。終了: 2026-11-06T04:33:32.386819Z。
- run.json SHA256: d5ed89a5d8a86f1398389ce758c29e2abc617739c4993de44e5da3adf4cb9f1d
- report.json SHA256: f321f4cff446b378decf43d6b0107f1e4dabde90ec51e2151fb3c97ea28aa0f2

</details>

<details><summary>gemma4-code90-native-contracts-c32-dev10-full14-20261106</summary>

候補: B native-contracts。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 67.500000 |
| math | 97.187500 |
| code | 84.444444 |
| instructions | 92.083333 |
| tools | 85.000000 |
| agents | 50.000000 |
| long-context | 68.877943 |
| reasoning | 71.956522 |
| commonsense | 66.333333 |
| multilingual | 81.843434 |
| vision | 87.500000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.850000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.690000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.791667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.863636 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.158645 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.285199 | 12/12 | — | — |
| probe-wikitext | None | 8.703654 | 8/8 | — | — |
| ruler | long-context | 0.625000 | 8/8 | 0.375000 | 0.375000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.675000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-06T04:07:38.604696Z。終了: 2026-11-06T04:35:53.284962Z。
- run.json SHA256: ff5a8b2b0b2b74d8d0698fc4ac0904e79c34391f19556f1ca9806f9739a100d6
- report.json SHA256: ebc449eb221dbe5de5324b8cbd8c40860d330088c86cbec001aac15b0806f168

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 72.575397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T04:36:28.977895Z。終了: 2026-11-06T04:43:03.078030Z。
- run.json SHA256: 976b916b38f2a813d66a6f55455b7c80aa99d768f05b58f8de8c739d6b1a5d17
- report.json SHA256: 7d903fdda296a19810e5adb425f06545267e4182c1ff3cd7872d946007ebd96a

</details>

<details><summary>gemma4-code90-original-base-c32-known64-r1-20261106</summary>

候補: O 原モデル。completed、test/tier30、manifest SHA256: 52a76039574df920138e7b88247955859af89865b2aa565fb11b00a0682177a6。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 78.261905 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| livecodebench | code | 0.857143 | 14/14 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| scicode | code | 0.573333 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T05:19:54.683960Z。終了: 2026-11-06T05:29:36.297592Z。
- run.json SHA256: 0efb01627fc7ad264f7c8c5b9d612c3aa271f610bdf666c426d527e7b0b35f1b
- report.json SHA256: b890e80a8660f56ed9d100a68b609d81a0ce751dd7633136d7a37841eed1e06f

</details>

<details><summary>gemma4-code90-public-final-balanced-c32-dev30-code-20261106</summary>

候補: correctLL balanced donor。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 71.325397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T06:13:42.219476Z。終了: 2026-11-06T06:24:23.043175Z。
- run.json SHA256: c924d8e739744742bd4e15a93f02b1688062ca2402683725d19dd162076f140b
- report.json SHA256: 7c4fcf2c5825ab88c277be717fda3d6e5f7d6532f4fe601044ee781e11c9d6dc

</details>

<details><summary>gemma4-code90-public-final-joint-c32-dev30-code-20261106</summary>

候補: correctLL JOINT。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.492063 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T07:08:29.853228Z。終了: 2026-11-06T07:14:10.006343Z。
- run.json SHA256: 1ed7c325f64f596dc1f5586da75611514dfe2340e0b02dc17da3664503ea3d63
- report.json SHA256: a860fe223b54b0c4cf87e059f4fbffbad91944af9b0ba9e69d2162dac39e4007

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairh-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 71.325397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.353016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T07:38:18.753984Z。終了: 2026-11-06T07:43:18.637051Z。
- run.json SHA256: 97636377628978ccc754fc5cef45032adec42bc0df1451ce9953c5d38da93b55
- report.json SHA256: de3e4d8cf2c8d03b7e4f953bdddae6ac5e961a6ba1c1ab29273481388426a278

</details>

<details><summary>gemma4-code90-coherent-c00625-c32-dev30-code-20261106</summary>

候補: H coherent donor1/16。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 72.158730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T07:38:18.863444Z。終了: 2026-11-06T07:44:28.506645Z。
- run.json SHA256: 6688ae421b321eb5416aef9221ff0f2e94e441e127b09d926081b07fc3e24c18
- report.json SHA256: 67c991781bfc25f247f90a81373711567aa1c4577b697c5d572f13a9ed7a2532

</details>

<details><summary>gemma4-code90-coherent-c00625-c32-dev10-full14-20261106</summary>

候補: H coherent donor1/16。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 81.250000 |
| facts | 70.000000 |
| math | 97.187500 |
| code | 84.444444 |
| instructions | 92.083333 |
| tools | 82.500000 |
| agents | 50.000000 |
| long-context | 87.627943 |
| reasoning | 76.956522 |
| commonsense | 67.333333 |
| multilingual | 79.343434 |
| vision | 100.000000 |
| safety | 88.888889 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.640000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.812500 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.710000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.937500 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.700000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.863636 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.030322 | 12/12 | — | — |
| probe-wikipedia-ja | None | 6.992556 | 12/12 | — | — |
| probe-wikitext | None | 8.327093 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.700000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.670000 | 100/100 | — | — |
| xstest | safety | 0.888889 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-06T07:52:34.621486Z。終了: 2026-11-06T08:15:04.437913Z。
- run.json SHA256: 59678ec34a45ef18ebc398230b3e2d80b22fe58ab157e710b95cf0662027ea37
- report.json SHA256: b089129d176904ef73586b96ee3138a26402e3b38fbc862e8e2239b8d8a94916

</details>

<details><summary>gemma4-code90-original-base-c32-dev10-full14-pairh-20261106</summary>

候補: O 原モデル。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 75.000000 |
| facts | 72.500000 |
| math | 95.625000 |
| code | 87.777778 |
| instructions | 92.083333 |
| tools | 82.500000 |
| agents | 50.000000 |
| long-context | 81.377943 |
| reasoning | 74.782609 |
| commonsense | 66.333333 |
| multilingual | 80.707071 |
| vision | 87.500000 |
| safety | 86.111111 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.630000 | 100/100 | — | — |
| bbeh | reasoning | 0.695652 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.825000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.750000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.700000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.875000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.666667 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.954545 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.800000 | 20/20 | 0.050000 | 0.050000 |
| mmmu-pro | vision | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.143497 | 12/12 | — | — |
| probe-wikipedia-ja | None | 7.120654 | 12/12 | — | — |
| probe-wikitext | None | 8.631947 | 8/8 | — | — |
| ruler | long-context | 0.875000 | 8/8 | 0.125000 | 0.125000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.725000 | 40/40 | 0.025000 | 0.025000 |
| winogrande | commonsense | 0.660000 | 100/100 | — | — |
| xstest | safety | 0.861111 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-06T07:52:34.704277Z。終了: 2026-11-06T08:20:32.130919Z。
- run.json SHA256: 19872bafe5d284065268e9e76bc9f67348a401e0d518bf8f10cfca379a540098
- report.json SHA256: deff52391c5a4953046dcb310c8a88aebe33a13bf17403adedc266e01590faf1

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairi-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 69.242063 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T08:42:34.107851Z。終了: 2026-11-06T08:48:46.858013Z。
- run.json SHA256: 3d1b1b12332b0e329a6fda899499fa9136ef413b92ae0682d5c759a7e5515206
- report.json SHA256: 56ae5084c17535cee83b480875085438348beb333cba38eb80b5a223cdd38d56

</details>

<details><summary>gemma4-code90-math-grpo-c1-c32-dev30-code-20261106</summary>

候補: I math GRPO c1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.908730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.050000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T08:42:34.248988Z。終了: 2026-11-06T08:55:11.686619Z。
- run.json SHA256: a1de0e534402b09622eedf90c5966c10916857108b72d681530fd085ddddbde6
- report.json SHA256: 46eedd3b1c0354663900697973ecd92f7d44ff5a389860cdbbf6878028b44c2d

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairj-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 69.103175 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.364127 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T09:30:04.539371Z。終了: 2026-11-06T09:36:36.649575Z。
- run.json SHA256: 4ccd67db0098d1c2a8e273b59fcb9d77b018ba80bb4e9fb553307758d4659c02
- report.json SHA256: 38feecd48d15aeba0f667fa4f9a60aee286bcc1ddb43a56751a84ae2224c8528

</details>

<details><summary>gemma4-code90-weighted-reasoning-c1-c32-dev30-code-20261106</summary>

候補: J weighted-reasoning c1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 72.158730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T09:30:04.652131Z。終了: 2026-11-06T09:35:55.370989Z。
- run.json SHA256: eb92ebc3701df4b250c80d898e3062afefc7805447a4617a885742e69cecd745
- report.json SHA256: ebc835f1050a896f0a27a5d37e8ab57b1b76f819388b71e0bbbb7902a30d2a3f

</details>

<details><summary>gemma4-code90-native-b-plus-coherent-h-c32-dev30-code-pairk-20261106</summary>

候補: K＝B＋H。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.908730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T10:33:27.466625Z。終了: 2026-11-06T10:39:32.321129Z。
- run.json SHA256: adcdce5bbae9106a7a20026ad1061c73db2a1abd290b88a118b1cd529cc4f32a
- report.json SHA256: 8147c5576a6c037a8671cd3ea9c181c2fa9ebd9ed81e38ddfbadd71514178e8a

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairk-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.769841 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.380794 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T10:33:27.648481Z。終了: 2026-11-06T10:40:59.193746Z。
- run.json SHA256: 0c808a93c8fe5df62ef21e5bf213e8ad7e361bf197076e29d8d2faeffefbf917
- report.json SHA256: 96c344f454bc5dc6d51f66890b1a721d771f9adab68b153566c71aa46bd90738

</details>

<details><summary>gemma4-code90-native-b-c32-dev30-code-pairk-20261106</summary>

候補: B native-contracts。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.075397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T10:33:27.693057Z。終了: 2026-11-06T10:40:27.181324Z。
- run.json SHA256: 4874a422a964730f9387a5e0abdaa50592e1d3f30af73764a531b148243271af
- report.json SHA256: 4706561625e7316a1947e3f87d25ca89172ef1e83c17e480782ddd6d628d51b6

</details>

<details><summary>gemma4-code90-coherent-h-c32-dev30-code-pairk-20261106</summary>

候補: H coherent donor1/16。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.769841 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.380794 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T10:33:27.714544Z。終了: 2026-11-06T10:39:41.981593Z。
- run.json SHA256: 2fd6820bf308f10e1cc8e1566d609610f2ea42d44351fb4d6675cae983479f77
- report.json SHA256: c91d25afa92824143cb69612058ae66b194b0da79858d6bd066a188e9e8a2c53

</details>

<details><summary>gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairl-20261106</summary>

候補: L＝O＋K1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 67.182540 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.237302 | 10/10 | 0.100000 | 0.025641 |

開始: 2026-11-06T11:03:26.607817Z。終了: 2026-11-06T11:17:40.920228Z。
- run.json SHA256: 3787215a01dbc159d4e4181a41db03582c8b1695ea98296d9b3b83a7794b8cc0
- report.json SHA256: b2cfb4ff8dd0e2ec4acb3a1ff079c435f54d470e012fb20dcfcdfecf0c3a0b1c

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairl-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 70.700397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.766667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.344683 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T11:03:26.608068Z。終了: 2026-11-06T11:09:48.665598Z。
- run.json SHA256: 80cd15de696cc36d1bd3b8f4f59fe613e46cbd16d65a218d9171d622b2f1426f
- report.json SHA256: abbcc61679101146d559aed11af2e9064c67f102acc016b09d0c13bfb9a2b2b8

</details>

<details><summary>gemma4-code90-heretic-public-base-c32-dev30-code-pairm-20261106</summary>

候補: P 公開base・元protocol。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 73.825397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 1.000000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T12:13:24.939856Z。終了: 2026-11-06T12:18:49.547439Z。
- run.json SHA256: 0135ae37dcac59dfb8315456fe285afdb9c9a3050d6a07f833c9d36ddabd7047
- report.json SHA256: 30e7d43837c2b3bf310c15dd931eedb9ce120c20bccdb29f55a76726d02381e3

</details>

<details><summary>gemma4-code90-heretic-public-base-k1-c1-c32-dev30-code-pairm-20261106</summary>

候補: M＝P＋K1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 67.908730 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.900000 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.650000 | 20/20 | 0.050000 | 0.050000 |
| mbpp-plus | code | 0.866667 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.299683 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T12:13:25.010508Z。終了: 2026-11-06T12:27:26.463233Z。
- run.json SHA256: 3ffc6c45487ac29130cc5a048a091efac701725b44e203e305af920ee7c38923
- report.json SHA256: dee83592ddaee3029002ced9abababc7a5648ac87cefcb9faffd0e052bec5c02

</details>

<details><summary>gemma4-code90-original-base-c32-dev30-code-pairm-20261106</summary>

候補: O 原モデル。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 71.186508 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.933333 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.380794 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T12:13:25.019571Z。終了: 2026-11-06T12:19:01.490872Z。
- run.json SHA256: 862a79ebb4ea18595690e8893e70799a2ab84531e5ab8d687a660f19b2f4202f
- report.json SHA256: 560d6b1ae6b011c213ba22506ec1523cf5e9355b08b9779347839b935b2947f2

</details>

<details><summary>gemma4-code90-k1-exported-text-c1-c32-dev30-code-pairm-20261106</summary>

候補: L＝O＋K1。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 68.075397 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.700000 | 20/20 | 0.050000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.256349 | 10/10 | 0.200000 | 0.051282 |

開始: 2026-11-06T12:13:25.046707Z。終了: 2026-11-06T12:27:16.454799Z。
- run.json SHA256: 9f12667978007452856a6987737f293a9b000fcf1d07b60a463be678b6584d4e
- report.json SHA256: bc6f80499c678a9fb456d018a5e9e0fdd9491284d458f0b8ae4683e10ac3bd06

</details>

<details><summary>gemma4-code90-incumbent-p-c32-dev10-full14-20261106</summary>

候補: P 公開改変base・保持確認。completed、dev/tier10、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | 90.000000 |
| expert | 87.500000 |
| facts | 72.500000 |
| math | 95.625000 |
| code | 87.777778 |
| instructions | 92.083333 |
| tools | 80.000000 |
| agents | 50.000000 |
| long-context | 87.627943 |
| reasoning | 76.956522 |
| commonsense | 63.000000 |
| multilingual | 82.247475 |
| vision | 75.000000 |
| safety | 58.333333 |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| aime-2026 | math | 1.000000 | 4/4 | 0.000000 | 0.000000 |
| arc-challenge | commonsense | 0.610000 | 100/100 | — | — |
| bbeh | reasoning | 0.739130 | 23/23 | 0.000000 | 0.000000 |
| bfcl | tools | 0.800000 | 40/40 | 0.000000 | 0.000000 |
| gpqa-diamond | expert | 0.875000 | 16/16 | 0.000000 | 0.000000 |
| gsm8k-platinum | math | 1.000000 | 16/16 | 0.000000 | 0.000000 |
| hellaswag | commonsense | 0.650000 | 100/100 | — | — |
| hmmt-feb-2026 | math | 0.875000 | 4/4 | 0.000000 | 0.000000 |
| humaneval-plus | code | 1.000000 | 10/10 | 0.000000 | 0.000000 |
| ifbench | instructions | 0.866667 | 15/15 | 0.000000 | 0.000000 |
| ifeval | instructions | 0.975000 | 40/40 | 0.000000 | 0.000000 |
| jamc-qa | multilingual | 0.708333 | 24/24 | 0.000000 | 0.000000 |
| livecodebench | code | 0.833333 | 6/6 | 0.000000 | 0.000000 |
| math-500 | math | 0.950000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| mgsm | multilingual | 0.909091 | 22/22 | 0.000000 | 0.000000 |
| mmlu-pro | knowledge | 0.900000 | 60/60 | 0.000000 | 0.000000 |
| mmlu-prox-ja | multilingual | 0.850000 | 20/20 | 0.000000 | 0.000000 |
| mmmu-pro | vision | 0.750000 | 16/16 | 0.000000 | 0.000000 |
| mrcr-v2 | long-context | 0.752559 | 4/4 | 0.000000 | 0.000000 |
| musr | reasoning | 0.800000 | 10/10 | 0.000000 | 0.000000 |
| probe-code | None | 4.090560 | 12/12 | — | — |
| probe-wikipedia-ja | None | 6.857430 | 12/12 | — | — |
| probe-wikitext | None | 7.878542 | 8/8 | — | — |
| ruler | long-context | 1.000000 | 8/8 | 0.000000 | 0.000000 |
| terminal-bench | agents | 0.500000 | 2/2 | — | — |
| triviaqa | facts | 0.725000 | 40/40 | 0.000000 | 0.000000 |
| winogrande | commonsense | 0.630000 | 100/100 | — | — |
| xstest | safety | 0.583333 | 36/36 | 0.000000 | 0.000000 |

開始: 2026-11-06T12:42:02.587775Z。終了: 2026-11-06T13:03:38.098375Z。

- run.json SHA256: 2a6f7b214bf76346d55df358d642ecbec705f3121debe665e268a1a72856c71f
- report.json SHA256: f72d4567cb926c373896705428f3524699d9dd9bb39d9650f9c72f843ce53703

</details>

<details><summary>gemma4-code90-heretic-public-base-k1-c00625-c32-dev30-code-20261106</summary>

候補: N＝P＋K1 text1/16。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 71.603175 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.750000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.733333 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.414127 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T13:06:59.857310Z。終了: 2026-11-06T13:12:45.247995Z。

- run.json SHA256: ae276489a7b7fa750abbbfbc9efe7e07a8b7afc689885bb7f7e54b855e572549
- report.json SHA256: 8aaab65b01ac336ce741ec98a3c459bb8036f3ec3c3b9ac47126a509a0ffb63c

</details>

<details><summary>gemma4-code90-heretic-public-base-c32-dev30-code-pairn-20261106</summary>

候補: P・Nの同時対照。completed、dev/tier30、manifest SHA256: a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b。

| 領域 | 点 /100 |
|---|---:|
| knowledge | — |
| expert | — |
| facts | — |
| math | — |
| code | 74.242063 |
| instructions | — |
| tools | — |
| agents | — |
| long-context | — |
| reasoning | — |
| commonsense | — |
| multilingual | — |
| vision | — |
| safety | — |

| Source | Area | raw score | 採点/予定 | truncated rate | empty rate |
|---|---|---:|---:|---:|---:|
| humaneval-plus | code | 0.966667 | 30/30 | 0.000000 | 0.000000 |
| livecodebench | code | 0.800000 | 20/20 | 0.000000 | 0.000000 |
| mbpp-plus | code | 0.800000 | 30/30 | 0.000000 | 0.000000 |
| scicode | code | 0.403016 | 10/10 | 0.000000 | 0.000000 |

開始: 2026-11-06T13:07:00.025016Z。終了: 2026-11-06T13:12:51.230144Z。

- run.json SHA256: ff831e6277c3454889172c176584def2f5022cdf48fdbe643719ed373f80668f
- report.json SHA256: bcbde7aa399ac3145a8a2cd9a2bd2a975996b0f6a75ae2db8f2f2a14c42e5a4a

</details>

## 保存証拠

初回109件の凍結artifactはanalysis/code90-resume-20261106/all-scores-20261106-v1/。

| Artifact | SHA256 |
|---|---|
| catalog.json | 1f413b7fa4047e04cdb37b4cb980ee73831d960924279a75acc355a00d6524f4 |
| coverage.json | bc350ad5b74578d428eee22fb1e6ff06e98bdca9667d4e2068f26f9fa4592392 |
| runs.csv | 104ef13cccf4d828df9dcf4c915dedd881f022460fb12e686ad0c80c6d822ab6 |
| source-scores.csv | f7a2cb77b3e38520f2a344de419e100b9e867c9166c143804e020a2b214c8c1f |
| candidate-documentation-map.json | 2a1faa5a83cf0d337a0aaed9835b6db6c0cccd2dd5bdff7584037134e11a7629 |
| main-candidate-scores-final.md | 87ff80a4a14ead988f83b5d376588a514e25e3ad09af8229cec68a399b329f16 |

更新112件は同analysis配下all-scores-20261106-v2/。既存exportの出力先だけを変更してrun/reportを再採録し、旧109行が完全一致、新3行だけの追加、読取エラー・metadata不一致0を確認した。

| 更新artifact | SHA256 |
|---|---|
| catalog.json | 790eb18d552bb47eb90aff6c575528c0fc91a76e5fb2f85bea703ef5fa8c8384 |
| coverage.json | ed9713b77cac9ba5136e18bcf6726ab8025f2c8776b1982695e799eb67ef6dfa |
| runs.csv | 83ca8434adbdca69c133deddb22b750642caca4006c46dd7b7ba25da879077c5 |
| source-scores.csv | cbe01ff4fe001131857d3e23f3c3386f6fda9d44907712dee0b79b72374daea0 |
| refresh-validation.json | 5c99a6a06067d419cd9cc2459f192c5b0dab6898654f18cb91a243916cd19319 |

これはsnapshotであり、後続試行は別の更新として記録する。originalは不変の参照として残す。MとNではPを保持し、Rのdev90改善で一時更新したが、full14のnet lossにより現working incumbentを同一保存Pへ戻した。originalへ戻していない。[Nの記録](2026-11-06-incumbent-coherent-step-n.md)と[Rの記録](2026-11-06-cumulative-mlp-math-step-r.md)を参照。

## v3追加3実行と保存証跡

旧112行は全fieldが完全一致し、別artifact all-scores-20261106-v3/へ3行だけ追加した。v1/v2は不変。読取エラー／metadata不一致0。初回v3ではR full14がrunningだった。終了後は別v3-finalへ再採録し、他114行の全field一致・Rの1行だけcompletedへの遷移を検証した。旧v3原本も保持する。

| Run ID | 状態 | protocol | 項目 / 採点 / 未採点 | code % | overall % |
|---|---|---|---:|---:|---:|
| gemma4-code90-heretic-public-base-math-mlp-c1-c32-dev30-code-20261106 | completed | code90_dev30 | 90 / 90 / 0 | 73.626984 | — |
| gemma4-code90-heretic-public-base-c32-dev30-code-pairr-20261106 | completed | code90_dev30 | 90 / 90 / 0 | 73.269841 | — |
| gemma4-code90-incumbent-r-c32-dev10-full14-20261106 | completed | full14_tier10 | 778 / 778 / 0 | 84.444444 | 76.382319 |

| v3 Artifact | SHA256 |
|---|---|
| catalog.json | c7cd64071ed4d7e119b14e6ab06b4b10a46d3d0f3b478b8b4a79169c80710e5c |
| coverage.json | 14f6b6353ac454f9d69e37663775b97ae0e0be1f5346707dfd4c611ea165edf7 |
| runs.csv | 47d3054c4d590b6c6fb4f1ce6816e2d31658c07dd5efc8b3fe4b04cbb5e6d961 |
| source-scores.csv | e9f6acd34c61f0dcb8053deba24dcb1ed7000328685215749231617f6f27449b |
| append-validation.json | 5e9b737803eae1c0359911480058e219c7d2cb3b3eb4051833600dfb829affd4 |

一時R採用時のstate SHA c0e271c4811d2e60b856e58dd6c7940fef10b6593454ef8554efcb2932a89d49は履歴として保持。S/Uのlocal-gradient予測は公式scoreへ追加しない。[Rの目的別観測](2026-11-06-current-r-purpose-observation.md)は別種の診断として記録する。

終了確認版 all-scores-20261106-v3-final/。R full14の終了は14:33:58.879312 UTC、report SHA c0a78d4488a5df88afb4108153bd7adb42decf795e5893485614639e8ae32ccc。R/Pは異時点の固定cohort比較であり、重みだけの因果差・速度差とはしない。tier10 code26にはSciCodeがなく、dev90や既知64とは比較しない。

| v3-final Artifact | SHA256 |
|---|---|
| catalog.json | 38f47cffd5e36ec3794854ffe5e99c3ffc0fd035a284ef2d5c7d99f8f2d184bf |
| coverage.json | 4ce5b2ba5707ef43bc546cb844bf651b10d19f35336ccecde55e2914a6e21fab |
| runs.csv | 17e54be4701b94bf5bf4c3e961fb3f34e9e1d842ebf5625b4016874011f05bb4 |
| source-scores.csv | e2f43ec7a18aacdbd48c50cf2175c800655bad62fc62acf0e22f0c748f3855d7 |
| terminal-validation.json | ba1bc55bbe39a6a7e8c234d38e152db96b24b77bf10d08478cfdd3bde59262f9 |

R full14 closure SHA db693d71de3180fe49910d9a89cf07c494771a1f2e37dfc4c41fbd2750969095：歴史P比overall−2.0927795 pp、CI[−4.26587,−.09675]、25 changed（9 gain／16 loss）。この得失を受けownerはR増分を不採用とした。P復帰後の新state pinはowner記録を参照し、全公式score行は採用判断に合わせて書き換えない。

v3終了・R撤回時の作業親の正式状態はP：working-incumbent.json SHA 2666c54246852bf328be8b6293ff3402f011403fc27ad28113c5bcddaf923c99。R増分の撤回decision SHA 1619f5487e92716434641797eb93971a4ad88e181ccc9f105d96f5d788b1e528。これは既存Pへの復帰で新たな重み改善ではなく、v3-final取得時のR採用状態は履歴として残す。

## v4追加3実行と保存証跡

all-scores-20261106-v4/をfresh出力として既存run/report専用exporterを再使用した。旧115行は全fieldが完全一致し、新しいS/P公式ペアと実在するS full14 running行だけ追加。v1/v2/v3/v3-finalは不変。読取エラー／metadata不一致0。S full14の未完了scoreはunknownのまま、0補完しない。

| Run ID | 状態 | protocol | 項目 / 採点 / 未採点 | code % | overall % |
|---|---|---|---:|---:|---:|
| gemma4-code90-current-p-native-s-c32-dev30-code-20261106 | completed | code90_dev30 | 90 / 90 / 0 | 74.242063 | — |
| gemma4-code90-heretic-public-base-c32-dev30-code-pairs-20261106 | completed | code90_dev30 | 90 / 90 / 0 | 72.575397 | — |
| gemma4-code90-candidate-s-c32-dev10-full14-20261106 | running | full14_tier10 | 778 / 不明 / 不明 | — | — |

S−fresh Pは+1.666667 pp、paired CI[−1.666667,+6.250000]。LCB3777／abc390_gが0→1、MBPP641が1→0、SciCode部分点は不変。proxy科学LL改善を公式SciCode改善に読み替えない。この初回v4 snapshotではSは未採用だった。終了後の採用とscoreは次のv4-final節を参照。[Sの固定編集・実測・運用記録](2026-11-06-current-p-native-science-step-s.md)を参照。

| v4 Artifact | SHA256 |
|---|---|
| catalog.json | 3c8868f29dc06d06a2cdb3022e3c301e8edcfda0fbf2c23902be7920a25764c8 |
| coverage.json | e92c9c7c40b25af77fa1c72bfe8c291a60855ecc11c45ea224b180b94eab004c |
| runs.csv | 3e89b964f823e8157b17c7c54d7031b064f7836269417361c1ae970d300941ae |
| source-scores.csv | 97cc889b2be47219633579c0d4f9f86545c9cd368b6a6f13129d0aca659641a2 |
| preservation-validation.json | a32a42ec73f8f6cf803e03a2a33800955e75b21bf803824de28c245cc53bc801 |


## v4終了確認とSの作業採用

fresh all-scores-20261106-v4-final/へrun/reportのみを再採録。旧v4の117行は全field完全一致し、S full14の1行だけcompletedへ移行した。旧115行も不変。全118件、completed113／partial4／旧recorded-running1、読取エラー・metadata不一致0。元v4のrunning記録は保存し、未知scoreを遡って捏造しない。catalogの旧exporter固定親注記だけは新stateに合わせSへ訂正し、公式run行は変更していない。

| Run ID | 状態 | 項目 / 採点 / 未採点 | code % | overall % |
|---|---|---:|---:|---:|
| gemma4-code90-candidate-s-c32-dev10-full14-20261106 | completed | 778 / 778 / 0 | 84.444444 | 78.129386 |

歴史P比overall−0.345713 pp、95% CI[−2.120989,+1.419686]、10 gain／13 loss。math+3.125／knowledge+1.6667／vision+12.5 ppと、code26−3.3333／tools−5／reasoning−2.8261／long−6.25／multilingual−4.7222 ppを併記する。safety・instructions・agentsは同点。JAMC／RULER各1 truncated/emptyを保持。fresh dev90+1.666667 ppと異なる推定対象の全14領域結果を総合し、rootは中程度の得失を許容してSをworking採用した。有意差・全領域無損失・最終90達成の主張ではない。

現作業親state SHA1049e67a536317876716b516f6abe90b8b0770d5a011e080340a5a8b12ae561a、decision SHA6c5bfbe6d7b1adbd3f780561eb8d87cd4ef80f136266120ee60367dac4fe1ce5。次編集はS tree60520ea1e63b25a69b10da1f39f63339221b1a0597c358d60a7cc6b0001d3be1を起点とする。P2666は履歴へ保存し、元モデルやPへ黙って戻さない。

| v4-final Artifact | SHA256 |
|---|---|
| catalog.json | 9f8bad5ecde633a227f8201017b586862b492f1af944f9ef376337a91661dbcd |
| coverage.json | 176427226f029316bef5f73c7999f63216ae391b44db0c35d6dbb53764ab6514 |
| runs.csv | d6cc0c5737f0c14f0876ec2dfc16441991b870c7ae13ed66043399f982b0baab |
| source-scores.csv | ca835bc4d92bbcabdc8ef9dc0940f2a90af472ab7803d22a529d745e81b44c38 |
| preservation-validation.json | b6e2dfe93a86855acdc78837497f583514c2b3c9e816dc41426258384cce0ada |

S full14 final-summary SHA3f88274a1de75ba0cfd43a98ff254f4ec2f3fc957559053fdde5fecd7b1f408b、独立数値確認 SHA4032427e9139bb1e92f4a215ca4dcb4a61c80f13cedc759684e63ffbb3bd581d。詳細は[S実験記録](2026-11-06-current-p-native-science-step-s.md)。


## v5追加2実行：ACとfresh S

all-scores-20261106-v5/へ既存numeric exporterを再利用した。旧v4-finalの118行は全field完全一致、新しいcompleted2行だけを追加。全120件（completed115／partial4／歴史的recorded-running1）、読取エラー・metadata不一致0。明示した2つの20261107 run IDだけを追加対象とし、run.json/report.json以外の本文は読んでいない。

| Run ID | 状態 | 項目 / 採点 / 未採点 | Code % |
|---|---|---:|---:|
| gemma4-code90-current-s-code-priority-ac-c32-dev30-code-20261107 | completed | 90 / 90 / 0 | 73.825397 |
| gemma4-code90-current-s-c32-dev30-code-pairac-20261107 | completed | 90 / 90 / 0 | 73.269841 |

fresh S73.269841→AC73.825397、差+0.555556 pp、paired CI[-3.194444,+4.583333]。旧S74.2421は別窓であり対照に代用しない。両90採点、119formal stop、error/trunc/empty0。MBPP641 gain、LCB arc195_b loss、Sci14 +0.5、Sci72 −1/9。ACはfull14 retentionへ進んだが未採用、作業親S1049を保持。

| Source | Fresh S score | AC score | denominator |
|---|---:|---:|---:|
| HumanEval+ | 1.000000 | 1.000000 | 30 |
| MBPP+ | 0.766667 | 0.800000 | 30 |
| LiveCodeBench | 0.800000 | 0.750000 | 20 |
| SciCode | 0.364127 | 0.403016 | 10 |

このsnapshotで進行中なのは gemma4-code90-candidate-ac-c32-dev10-full14-20261107（778件）。120行の歴史的未完了1件とは別に active-run-separate.json へ保存し、scoreはunknownとした。開始18:21:37Z、既存S full14の歴史対照だけを利用する。全14結果・採用判断は完成後に追記し、未知値を遡って捏造しない。

| v5 Artifact | SHA256 |
|---|---|
| catalog.json | 3db7ed6285dc6b48780eaa594cd6de286967a5e4895cc3ba0f6fb64bc762a6d9 |
| coverage.json | 8108b514e794fee8b5cc01f980ba623aa861114aa3472d11e67e822b0e574dd6 |
| runs.csv | 073cbaeb905342ce421f6528b04fbc8b0514b25d03b54c886434e3035dc1b225 |
| source-scores.csv | 659d2378fde9c93c930e8af9608406c00afb07777d38abb4086aba0dfba4ceef |
| preservation-validation.json | 4b5c10f0ccaaabf0c57d685b45e1c59d8447d5c6843cb8611427dbe63642198c |
| active-run-separate.json | 9a9e040317bf244f54e5e38d6f4c8d1ea5a54748a98f3f6d0d64097223d5157c |

AC official closure SHA94631713082699f1a46812c503cc5afd375f08995a05ee9011703ca48660367b。詳細は[AC実験記録](2026-11-07-coding-priority-normalized-matrix-ac.md)。


## v6追加1実行：AC full14完了・working採用

all-scores-20261106-v6/へ同じnumeric exporterを再利用。旧v5の120行は全field完全一致、
新しいcompleted1行だけを追加した。全121件（completed116／partial4／歴史的未完了1）、
読取エラー・metadata不一致0。run/report以外のbenchmark本文読取や新規採点はない。

| Run ID | 状態 | 項目 / 採点 / 未採点 | Code26 % | Overall % |
|---|---|---:|---:|---:|
| gemma4-code90-candidate-ac-c32-dev10-full14-20261107 | completed | 778 / 778 / 0 | 87.777778 | 78.934942 |

歴史S78.129386→AC78.934942、差+0.805556pp、CI[-1.059524,+2.805556]。
746 capability＋32 probeを分離し、11 gain／6 loss。code26はSciCodeを含まないため、
fresh dev90の4source平均73.825397と混同しない。full14ではcode+3.3333、
instructions+3.3333、tools+2.5、long-context+6.25、reasoning+5に対し、
knowledge−1.6667、facts−2.5、vision−6.25。Safety21/36は同値。
同一cohort/source・観測されたruntime条件を確認した歴史比較であり、同時対照ではない。

rootはこれらの得失とfresh dev90 +0.555556ppを受けてACをworking採用。
state f0a5c61bdbcf6cdba3b71db6487c786d5af004bbc2b31a8c67ed0a9b72ccd324、
decision01f9d37092b6c51c54e26b1281b044052cae8aa856cd6018b7317773933cc9ac。
元S1049はarchive済みで、次の変更は採用済ACの全保存重みから積み上げる。
90目標・known64最終確認・current DeepSWEは未達／未実施のまま。

| v6 Artifact | SHA256 |
|---|---|
| catalog.json | e9ca04d3a686a7b85af992eab9b8db6ef27038fb09c3abf15594746387001443 |
| coverage.json | 3225b50020c9e62c006b15e9dd285a17b16150256eb8deaeb0561dc342a523d5 |
| runs.csv | bdbc92834fc62375cafe4dbf0b964f69f3889dd4a804844a3585ed81c2e59af2 |
| source-scores.csv | b112199cd2cb5876c611a4809b4d0cf28a6c970839a514dc01f02fa909b286db |
| preservation.json | 5c562b10896e2c0902a1eba7b4ab1a91517055e6b8b9f7d36e362387feac2d40 |

最終summary dc02a16cac4fe8b2a6ebab89981227df7b54a3df35c048a7cde7d7a31638001c。詳細・通信量・runtime留保は[AC実験記録](2026-11-07-coding-priority-normalized-matrix-ac.md)。

## v7追補：fresh AC対AD/AEの同時3条件（124 runs）

既存121行の全fieldは同一。追加した3runはすべて90/90採点、未採点0。
新規採点は行わず、保存run/reportだけを既存exporterで取得した。

| Run ID | Code90 % | 対fresh AC差 pp | 95% CI pp | 状態 |
|---|---:|---:|---|---|
| gemma4-code90-current-ac-c32-dev30-code-pairadae-20261107 | 71.408730 | — | — | completed |
| gemma4-code90-native-ad-c32-dev30-code-20261107 | 73.825397 | +2.416667 | [-1.333333, +6.666667] | completed |
| gemma4-code90-repair-ae-c32-dev30-code-20261107 | 72.158730 | +0.750000 | [-3.916667, +5.833333] | completed |

今回のAC71.408730は同じ採用済み重みでのfresh対照であり、過去AC73.825397を代用しない。
ADはHE29/30、MBPP24/30、LCB15/20、SciCode43.634921%。ACは30/30、22/30、14/20、42.301587%。
AEは30/30、22/30、15/20、40.301587%。4sourceを等重みにし、SciCode部分点を保持する。

ADはHE116 loss、LCB3777 gain、MBPP641/801 gain、Sci35 +1/3、Sci58 −0.2（実数4gain/2loss、公式0.5-crossingは3gain/1loss）。
AEはLCB3777/abc390_g gain・arc195_b loss、MBPP641 gain・792 loss、Sci58 −0.2（実数3gain/3loss、公式crossingは3gain/2loss）。
両CIは0を跨ぐが、追加の有意差・ゼロ損失・code単独の採用gateにはしない。

全arm119正式API calls、正式sourceのtruncated/emptyは0。process counterとの差はstartup1request・input21/output1に一致する。
記録input/outputはAC93,564/323,716、AD96,181/332,608、AE91,739/321,516。
wallは丸め済み354.6/359.9/355.3秒で、因果的な速度差とは解釈しない。agent stepsは未測定。

rootはADのfull14保持確認を選択し、2026-11-06 19:55:19.895 UTCにownerを起動した。
正式runはgemma4-code90-candidate-ad-c32-dev10-full14-20261107、778項目・c32で実行中（launcherの子開始記録19:55:20.420 UTC）。
比較対象は歴史AC full14 78.934942で、fresh同時full14対照ではない。
AD/AEのworking採用はまだなく、AC state f0a5c61b…を保持する。このlive runは旧historical unfinished1件と別である。

| v7 Artifact | SHA256 |
|---|---|
| catalog.json | 2395f78d67982a76465371b4f9bd50a4090bf14b624c11fd0cb782002d07a1bc |
| coverage.json | 6b33cf64955f873d73601ceed60c3f89d55012126ce27e0d0fc2995324be8bee |
| runs.csv | c90ed3e78504a526676105a003ad19b87ee72b9dcd68fcccd951960f83a9b089 |
| source-scores.csv | bbc7b94657b3a3350d0c7ee1c3a296b54bc567fce0ba0302f2fced7e33dd1080 |
| preservation.json | 0b2c7a0eb873036c7cf4485eabbc9d07f77ddd01bc4009e9c6d7fd315577f4a1 |

3arm final-summary SHA 9a0c02af5c33b8b3ff9b1d58cc1e7aa8d38d9d2b8d35bbee77196766a3173564、独立数値review SHA 8ada72828f6083cd6f33f0a04124b748859c31aaf6b1155c7d9740dfe8785d04。詳細は[8候補・3arm実験記録](2026-11-07-eight-current-parent-dependency-sets.md)。

## v8追加1実行：AD full14完了・得失を明示したworking採用

旧v7の124行は全field完全一致。既存numeric exporterでADの保存run/reportだけを追記し、全125件（completed120／partial4／歴史的未完了1）、読取エラー・metadata不一致0。新規採点やモデル呼出しはない。

| Run ID | 状態 | 項目 / 採点 / 未採点 | Code26 % | Overall % |
|---|---|---:|---:|---:|
| gemma4-code90-candidate-ad-c32-dev10-full14-20261107 | completed | 778 / 778 / 0 | 87.777778 | 77.576379 |

歴史AC78.934942→AD77.576379、−1.358562pp、CI[−3.863950,+0.925866]。746capability＋32probe、21gain／27loss。Code26は同点でSciCodeを含まず、fresh dev90の+2.416667ppとは別の測定。RULER8→6（長文領域−12.5pp）、GPQA14→13、BFCL31→30、BBEH18→17に対し、多言語+5.681818pp・vision+6.25pp。RULER失敗2件は各16,384出力、sourceのtruncated/emptyは各2/8。原因を回答本文から調べたとはしない。

563formal chat／3,062logprob calls、記録input1,410,852/output1,148,623、wall1,820s。TerminalBench1/2は同じ報酬0/1、steps/API/tokens/error詳細はunknown。歴史比較のruntime留保、並行O2ツール準備のI/O重複、全source差は実験記録に保持する。

Rootはfresh coding gainと上記の損失を合わせてADをworking採用。積み上げP→S→AC→ADを保持し、次は完全保存ADと新しいADの導関数を使う。全14overall改善、全分野改善、90達成とは主張しない。ACはrollbackとして保持、known64とcurrent DeepSWEは未実施。

state7d65d2181d53b7c67a005e4bfaf3a17739dd9617d6abfa61b2ba2bdb523110fb、decisiond5834a0cb19ebbf81ba951343a6420cc0df0d07433eb09ee9e9ad2e0dc6ac4ff。

| v8 Artifact | SHA256 |
|---|---|
| catalog.json | 46196e9a7e87c69bdbe232ec088d3dff5995a7e5c29b1cee0adb652fdaaf0086 |
| coverage.json | 7b71bafd03e03346c2e382d19b303b55bc048bb6ae991017424d2ac90df533ea |
| runs.csv | 5f1a4ae3ea1a40037e29c9415eac74acfd4ee004972953f38bacc0ec1c39f4f4 |
| source-scores.csv | f77a3df6de79e52b6e951ed297435f015465affc8dd91196ad016e44614a6b69 |
| preservation.json | 4105512ea7167edffd3a48230541f41f370a2124af015c1772dccd283fd61328 |

最終summary40e5dc1f93a98a2172c0655ab90e340455ee197ea4365fd89633ccd4fe4d8798。詳細は[8候補・AD full14実験記録](2026-11-07-eight-current-parent-dependency-sets.md)。

## v9追補：fresh AD対AF/AG、両増分を不採用（128 runs）

旧v8の125行は全field完全一致。追加3runはすべて90/90採点・未採点0。保存run/reportのみを既存exporterで追記し、新規採点・モデル呼出しはない。

| Run ID | Code90 % | 対fresh AD差 pp | 95% CI pp | 状態 |
|---|---:|---:|---|---|
| gemma4-code90-current-ad-c32-dev30-code-pairafag-20261107 | 74.242063 | — | — | completed |
| gemma4-code90-finalist-af-c32-dev30-code-20261107 | 72.658730 | −1.583333 | [−5.750000, +2.166667] | completed |
| gemma4-code90-finalist-ag-c32-dev30-code-20261107 | 73.408730 | −0.833333 | [−2.500000, 0.000000] | completed |

4source等重み。AD/AF/AGのHEは29/29/28（各30）、MBPP24/23/24（各30）、LCB16/15/16（各20）、SciCode平均40.301587/42.301587/40.301587%。AFはHE152 gain・HE95 loss、LCBabc390_g loss、MBPP641 loss、Sci58 0.8→1。AGはHE95の1→0だけが異なる。Sciの部分点を二値化していない。過去AD73.825397は今回の対照に代用しない。

既存のprompt-free RequestRecorderを全arm同一policyで初めて有効化し、各119 HTTP attempts＝119 logical calls、retry/error0を確認。内訳HE30、MBPP30、LCB20、SciCode39 calls（10項目）。agent stepsは未測定で、API callsを手順数と呼ばない。正式input/output tokensはAD94,417/342,082、AF96,204/337,346、AG93,709/339,360。全source truncated/empty0。processとの差は各startup1 request・input21/output1で、item単位のfinish reasonは推定しない。wall400.1/357.3/537.3秒、因果的な速度比較ではない。

全3runnerの実image・RootFS・Image.Config・正規化command・非credential環境・82 installed source/resource・packagesが一致。独立observer directoryを正規化したmountsを確認し、raw HostConfig hashの完全一致は主張しない。親ADのstate/statと各serving identityは閉鎖時も一致した。

Rootは両増分を退け、AD state7d65d218…を変更せず保持する。AF/AGのfull14・採用は行わない。90目標は未達。次の観測も完全保存ADから行い、棄却候補や旧親の導関数を新親のものとして流用しない。

| v9 Artifact | SHA256 |
|---|---|
| catalog.json | 6df331ca7af15f5c91c115e28b71aa2d4225d3f65908f6cfc4a52c6970e122cf |
| coverage.json | b146f5f44d377abcdb1988e78bf63b944f22cb14e3cc25b2c648c8ceef12d0e1 |
| runs.csv | 14151e87ecc1b52da79963e38488e75ed074d7e841f3721383752dab4f9e3219 |
| source-scores.csv | 1d587a38ff37f0bab67e97e7a4f9291df998103c9debb271bfd8e43b2ea4debb |
| preservation.json | 6b1869818425424ee4d55201655e7b6253b814b6c353123f8ba89cd2ba6f5f46 |

最終summary53bee6345ab4d88b120ca41a378d3b7446a3fdab962322e9019161c7695bd19b、postchecks4bb86167ea493d1bee920c92125d30f5d4acd175810b0733536edcb7ce1f2f5a。Root-delegated decision 3d780adc1f9ad6e49d9d29356c0631aa64d4216110359308d91bdc3e52cabb27。


## v10追加2実行：fresh AD対AL（130 runs）

旧v9の128 row objectsは全field完全一致。保存run/reportのみを既存exporterで追記し、125 completed・4 partial・歴史的未完了1。新規採点・モデル呼出しはない。

| Run ID | Code90 % | 対fresh AD差 pp | 95% CI pp | 採点 / 未採点 |
|---|---:|---:|---|---:|
| gemma4-code90-current-ad-c32-dev30-code-pairal-20261107 | 72.158730 | — | — | 90 / 0 |
| gemma4-code90-library-al-c32-dev30-code-20261107 | 68.825397 | −3.333333 | [−7.500000, 0.000000] | 90 / 0 |

AD→ALはHE28/30→27/30、MBPP24/30→24/30、LCB15/20→13/20、SciCode40.301587%→同値。4source等重み、SciCode部分点を維持。3 loss / 0 gain（HumanEval/95、LCB3789、arc195_b）で、rootはAL不採用・full14なしとしAD7d65を維持した。以前のAD74.242063を今回のfresh対照へ代用しない。

正式APIは各119、再試行・error・未採点・truncated・emptyは0。input/outputはAD94,930/330,234、AL96,408/329,144。wall366.8/398.8秒、agent stepsは未測定。Process counter各119stop+1lengthと正式callsの差はstartup1request/input21/output1に一致し、lengthを問題単位の打切りへ帰属させない。実OCI/source82/observer設定/親・配信identityの閉鎖確認はPASS。

| v10 Artifact | SHA256 |
|---|---|
| catalog.json | 18587ed75f9f76a2705c1cd6030b0629afd8140a6ec06dbdeaf43a4ff8f403a9 |
| coverage.json | 5d3ff481e45f3407d16313df20a326d23c0df268b00ed3fe53f1d426e39023d7 |
| runs.csv | f3642f710d9ef7c6779fe407c4f7dfbfe476dbe771e268998a08c0db3a784782 |
| source-scores.csv | de09f294325ef52acc95b1a9bc09cb87de91b328bafb7da24dec6ffc44fc0d5a |
| preservation.json | 852c14f0ea0ebe76fa39e5d94728721f072eacc8256e16ba795e50b7fae6ff7f |

最終summary 7ecd713717bfba6be7b2991a68deb0e7162d04993c13f8b906a73b2e0a961a87、postchecks 1730289eb9d7acc56b8611698a43151fe020b2e1b444b9fb938fbfd4b836cb75、root-delegated working-decision 83334222eeeb350ff843176964fbf40a17aa3b6d358c3fe62e52f890ad31da65。詳細は[science実験記録](2026-11-07-ad-science-dependency-sets.md)。

## v11追補：AN–ASの探索的official overlay（138 runs）

旧v10の130 row objectsは全field完全一致。追加8runはすべて90/90採点・未採点0。公式run/reportの既存numeric exporterのみで追記し、再採点・追加モデル呼出しはない。overlay labelは論理的な編集identityであり、候補の物理checkpoint treeではない。

| Round | Arm | Run ID | Code % | 同round対照差 pp |
|---|---|---|---:|---:|
| 1 | CONTROL | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-control | 72.158730 | — |
| 1 | AN | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-an | 74.242063 | +2.083333 |
| 1 | AO | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-ao | 71.325397 | -0.833333 |
| 1 | AP | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r1-ap | 71.742063 | -0.416667 |
| 2 | CONTROL | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-control | 73.769841 | — |
| 2 | AQ | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-aq | 72.436508 | -1.333333 |
| 2 | AR | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-ar | 74.742063 | +0.972222 |
| 2 | AS | gemma4-code90-ad-functional-overlay-c32-dev30-20261107-r2-as | 71.325397 | -2.444444 |

同一empty-control processを維持したが、各roundで新しい対照runを実行した。対照72.158730→73.769841の変動を保持し、異round間の差をmatched gainとしない。AN +2.083333ppとAR +0.972222ppを保存済み重みで確認する段階へ選択した。ARのSciCodeは同round対照比−1.111111ppであり、全source改善ではない。採用・90超達成の主張はない。

全8runでformal chat119件ずつ、合計952件。unknown/truncated/empty、記録retry/errorは0。起動時相当の1 requestは正式件数から分離し、HTTP回数をagent stepsとは呼ばない。

実行ownerの出力mount不備に対し、rootが許可したopaque artifact preservation（round1）と最終出力のmounted先routing（round2）で8件すべてのoriginal final reportを保存した。raw bodyを読まず、report再構築・再採点・モデル再実行なし。全4サービスは同CIDで復帰。修正版公開owner/wrapperは出力mountを明示検証するが、実行済みimmutable sourceは変更していない。回復I/Oの重複があるため単独latency改善は主張しない。

| v11 Artifact | SHA256 |
|---|---|
| catalog.json | 9e070278e17a654d22de4cd2179affe425c33d22f7a0d425d6f71416fc5f310d |
| runs.csv | 16485de8e577e792260acdc1c965fbdfa2b3b904e62fc541b61108a278cc43ee |
| source-scores.csv | ca82abeb0d1a46f726ebb25601f5d387b717d82654755e2b8a60c113a9877f3f |
| coverage.json | 55b2e2dc63dac7b6043d4e93b1fcc8fe704240e68abb783bb566f984c558e5c5 |
| preservation.json | af94abd9a73d1a7925ed82ecf3caee2f4b6b92c8c7320c1637772f42f88cdf89 |
| export-reproduction.py | 0d62cd62308c95490ae1cdaaa3095c80a7544f12e080620ba7d38472e9ace829 |

最終numeric summary a750513f9f9f4b018c88ad6a9e0e0b3350827af1282bbe9e46f5be5e0081b52f、owner closure 9745550da9742a87813267497886fd81e1c2a7537c950136677d36d081d72a21。[機能target実験記録](2026-11-07-ad-functional-targets.md)にsource得失・復旧範囲を保持した。

## v12追補：保存AN/ARのfresh ordinary確認（141 runs）

旧v11の138 row objectsを全field完全一致で保持し、保存重みの3 runだけ追加した。全90採点・未知0。overlayを保存checkpointと同一視せず、今回のfresh AD対照だけから差を計算する。

| Arm | Run ID | Code % | fresh AD差 pp | HE /30 | MBPP /30 | LCB /20 | SciCode % |
|---|---|---:|---:|---:|---:|---:|---:|
| AD | code90-current-ad-an-ar-ad-20261107-v1 | 74.742063 | — | 29 | 24 | 16 | 42.301587 |
| AN | code90-current-ad-an-ar-an-20261107-v1 | 72.436508 | −2.305556 | 29 | 24 | 15 | 38.079365 |
| AR | code90-current-ad-an-ar-ar-20261107-v1 | 73.075397 | −1.666667 | 30 | 24 | 14 | 42.301587 |

ANはLCB arc195_bとSciCode58/72の数値3 loss、gain0。公式binary flipは1 lossで、Sci部分点2 lossは別記。CI[−5.361111,0]pp。ARはHumanEval/152 gain、LCB3777/abc390_g loss、CI[−5.416667,+1.666667]pp。両増分不採用・full14なし、AD7d65を維持。

正式HTTP各119、error/retry/unknown/truncated/empty0。startup1件/input21/output1を正式回数から分離。AD/AN/AR input94,864/97,133/95,636、output336,577/341,074/325,302、wall374.2/385.1/391.4秒、agent steps不明。OCI/source82/親・配信identity・出力mount閉鎖PASS。

| v12 Artifact | SHA256 |
|---|---|
| catalog.json | e22e9f36b6f0aaf02c60c3c29faf8002c767ea9fe1c2068c0bb1cc86a1612097 |
| coverage.json | e67552a05c5d60c58ddaec9d360e991db4845befe245189e3b85150d65bd4620 |
| runs.csv | 2d9af47c83c7ab78ef113dfc4aba0469e0d8c1ff19f0d3d2c1acea575bfa49a9 |
| source-scores.csv | 8436aa106184c336ec0dd3276f7de60d1faefefee5717d44025b125b3af189d8 |
| preservation.json | ee4cf717c01ea8b45edfd5c56289e36a7590efb11f49f27a2e2c937f44d653db |
| export-reproduction.py | 339f49c72218156a30195024c97c9575f62d9710de4d2f74e362ab8e4be97a78 |

最終summary 2f3a50cdf8610c42f2c753d588c190f64ba7dae71d2405f075b90170a1951b86、postchecks f2253c519b7341f66de894cc1fc8d8b6e152b308ed12f00ef4bc3d7b566a8169、working-decision 66aa64e14eaf5c69d114760bdea3e04af145617e172b7a9dc374c44cc6d5bd8d。[機能target実験記録](2026-11-07-ad-functional-targets.md)に対照窓・partial scoreと不採用理由を保持。

## v13追補：Q/Kを含む1ラウンドの探索的公式比較（145 runs）

旧141行完全保持、追加はlogical-overlay4行。全90採点、未知/切詰め/空0。正式HTTP各119、error/retry0、startup各1を分離。保存済み候補checkpointの結果ではない。

| Arm | Code % | Same-window control delta pp | HE /30 | MBPP /30 | LCB /20 | SciCode % |
|---|---:|---:|---:|---:|---:|---:|
| CONTROL | 72.575397 | — | 28 | 23 | 16 | 40.301587 |
| AX | 72.242063 | -0.333333 | 30 | 23 | 14 | 42.301587 |
| AV | 72.932540 | +0.357143 | 28 | 23 | 16 | 41.730159 |
| AW | 72.158730 | -0.416667 | 28 | 24 | 15 | 40.301587 |

AVはSciCode80の2/7→3/7が総差を生み、HE/MBPPは各1gain/1lossの入替。pairedCI[−2.976190,+3.690476]pp。AVだけ保存後のADとのfull14確認へ選択、採用は未決定。AX/AWは今回の比較では低下。全4元CIDは正常復帰。

| v13 Artifact | SHA256 |
|---|---|
| catalog.json | c694b6a3c10d41e82c78c5fdb78a8affd2a154a1e4f5bd8b35614a3a44ad590c |
| coverage.json | a72b295ea6372ca1639ca6ab978c203ade0e8672dd78ce6864a5f5c68c70091e |
| runs.csv | 4ba5024afafa805fd9b4dc1e4a9db779b9ae57306303c444a5f3558f5a939364 |
| source-scores.csv | a6f37851a87ae035d9f7408f1d6b2b27d22ec161c96bf4c64d6e826902c2ea57 |
| preservation.json | 3fc3890e93df24d958d3df3ae5d59a2e5ccf81b41f7e1abffae2e15dda1da9c7 |
| export-reproduction.py | 515142feb24ee4f2e0bdcee0a67d7076ed82ebb58c6ab965d60c9cb93de8f6c3 |

最終numeric summary65860b00…、postchecks65172ef2…、closurebb4bfb69…。[候補の全経過](2026-11-07-helper-capped-functional-targets.md)にsource得失、partial scores、使用token、actual load/restore proofを保存。
