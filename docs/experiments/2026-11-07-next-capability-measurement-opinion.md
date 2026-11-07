# 次の能力測定と組合せ範囲：独立意見

**推奨は、先に新しい12〜24問のnative自由生成で実在する失敗を確認し、case別correct-LL感度から少数のnative tensorを共同変更する1案に絞ること。** N30 code72.94、SciCode43.41に対し、既知校正は48件中46合格、guard10は全てcorrectへliteral一致だった。guard0_1だけでF02のguard内正margin増分85.84%を占めた事実は、marginを大きくする余地と未解決能力が違うことを示す。native64近天井という最新状態も、新しい失敗を測る理由になる。過去候補の採用判断や層順位は根拠にしない。本メモは提案であり、モデル・GPU・service・候補作成は0。

## 最短案：case別native感度→小さい共同BF16編集

1. 新12〜24問は、公開devの個別問題や正解を使わず、依存関係で設計する。例は「配列軸の意味→ライブラリ呼出し→退化時のshape復元」「状態更新→保存条件→境界/終了判定」。正解実装・foil・入力契約をCPUで先に検査し、未知例外と誤答を分ける。AVで同じnative設定、各case1回答を生成して全問実行、難易度が飽和していたらその試行を残して設計へ戻る。失敗だけの成功率を分母にしない。既存`science_generate.py --thinking-control server-default`を使い、4AVの実default true証拠を結ぶ。
2. 全新caseを**一case/fold**で明示した入力へ固定し、`native.py --mode chat-public-answer --objective correct-logprob --answer-batching serial --max-cases-per-microbatch 1`を同じAV上で1回計測。保持制約に必要なmarginも同じ設定で1回計測する。collectorはcase別LL値を保存するが、勾配はfold平均なので、一case/foldがcase別の勾配を得る既存の方法となる。12〜24問なら2目的×12〜24=24〜48制約で、既存solverの128制約上限内。正答LLも固定reference文字列への条件付き尤度であり、別の正しい実装や自由生成正答率ではない。
3. 失敗caseのLLを上げる方向が複数の依存caseで整合し、既に成功したcaseへの損失が小さい**少数のnative tensor（例2〜4個）**を、実値・BF16刻み・層間の対応を確認して`native_tensors`のexact allowlistへ固定する。attention側の正規化/scaleと後続MLP/residual側など、同じcaseに効く組合せを検討するが、「その能力の担当」とは呼ばない。勾配の大きさ単独やguardの平均marginで選ばない。`propose_native_objectives.py`の既存2目的solverを使い、失敗caseのcorrect-LL増と成功case/他能力の許容損失をcaseごとに事前宣言する。marginは保持の補助で、全caseのmargin増加を目標にしない。既存v1 solverは両objectiveの全fold制約を要求するので、その契約を守り、許容負値を隠さない。
4. 1共同案だけを現在の最高点親AVから保存BF16へ丸め、消えた変更、実差分、L2、対象tensor全体を分母とした相対量を検証する。既存finite replayでAV→案→AVの同case LL/margin・selected-word復元を測り、**一次予測、有限BF16効果、native自由生成＋実行利得を別欄**にする。次に同じ新case全体を同条件で案から1回生成し、合否の改善/退行・時間・tokensを比較する。LLだけ良くても能力改善とはしない。元モデルへ戻して作り直さず、候補を採用した後の計測はその新しい最高点親で取り直す。

## 代案：unit方向が必要なときだけcoherent MLPへ

nativeの少数tensorではcase間の要求が衝突する場合、短い同一case群で`native.py --virtual`のO投影入力head gainとMLP-down入力column gainを補助観測し、同caseのLL感度・既存AV live10の層/attention/MLP/residualの実RMS・保存行列の列normを照合する。live10はSciCodeを含まず、異なる問題/位置/rankのRMSは混ぜない。gate/up入力とdown出力の依存をまとめて見る必要が明瞭なら、1〜2層・各1〜2unit程度の明示supportから`matrix_full_units.py`でgate/up完全行＋down完全列の真の座標微分を計測する。head gainはO列群、MLP gainはdown列に対応する方向であり、gain勾配をgate/upの勾配と読み替えない。virtual配列は現在のcoupled solverへ直接渡せないため、観測用に留め、保存重み案は実座標の計測を用いる。

この代案は最短ではない。`--virtual`はpairedのみでserial/CPU-FP32 offload併用不可。full-unitは最大32unit、1層最大2unitで、同親に束縛したattention標本のreference supportも必要。solver側にもmatrix-only/明示family条件があり、旧観測を当てはめず現行sourceの入力条件を満たす。全head・全層の組合せ探索へ広げず、同じ案数と有限/native確認を保つ。

## 資源・観測限界・公式確認

現在のcollectorは単一可視GPU・空き70GiB以上を要求し、31B BF16の全モデルを載せる。これだけで長いbackwardが収まる保証はない。GPU専有flagと外側のglobal/GPU lock、exact CID復帰手順を別に要し、いまの4AV配信を勝手に停止しない。既存Torch/Transformers環境、NVMe保存・キャッシュ、全9file attestationとstat継続性を使用する。`--max-frame-tokens 1024|2048`は明示opt-in、超過はtruncateせず拒否。長い順・1case即VJPでpeakを確認する。`--saved-activation-storage cpu-fp32`は**serial native-onlyの非leaf CUDA FP32保存活性だけ**をblocking CPU退避し、BF16活性や全GPUメモリを消す機能ではない。host RAMとコピー量も記録する。full-unitはgate/up入力をCPUへ移すが、全graphを退避する保証はない。

観測工具・solver・editor・replay自体の新実装は不要。ただし新12〜24問の実行oracle/契約は別途準備が必要で、既存`science_sandbox.py`は任意の新JSONを採点できず固定catalogと照合する。この部分が未完成なら新問を採点済みとせず、計測/候補へ急がない。teacher-forced frameは空thought条件で実際の思考過程を再現しない。gradientは局所一次、RMSは活性の量、重み相関は対応であり、相互作用・因果的能力局在・一般化を証明しない。

native実行で利得がある1案をOpus 5.5へ根拠/損失込みでレビュー依頼し、正規WSP dev10全778を同じrunner/重み到達chat+logprob/観測条件で親と比較する。ordinary DP4条件の同条件dev10親結果がなければAVも1回必要。**dev10にはSciCodeがない**ので科学codingの改善確認は全2338問dev30（14領域、code90、SciCode10）まで必要で、N30と同条件を固定する。候補のlive captureを加えるなら親も同capture条件にし、既存ordinary/client-only N30をそのままmatched対照にしない。さらに有望な最終案だけ既存dev60と最終testへ進み、各tierを混ぜない。全失敗/未採点/再測定を残し、多少の他領域損失は目的に照らして明示判断する。最終DeepSWE32並列・全113一周とwiki/HF条件は別の必須終端として残る。

[現行source SHAとCPU --helpの確認](records/2026-11-07-next-capability-measurement/source-and-help.json)。GPU実測・新候補作成は行っていない。
