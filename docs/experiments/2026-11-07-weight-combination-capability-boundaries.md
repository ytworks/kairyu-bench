# 公開回答の失敗に対する既存計測の範囲（2026-11-07）

**次の変更方向は未選択。** F02 は AV と公式コード全26件同点で、teacher-forced margin の大幅増加は1件に集中した。[既存の数値監査](2026-11-07-authored-case-audit.md)以上の能力改善を仮定しない。進行中の全2338件 dev30 が完了する前に、対象重み・目的・候補を選ばない。本メモは現行ソースと `--help` の読取りだけで、GPU・モデル呼出・solver・サービス・lock 操作は0回。

| 入口 | 同時に扱える重みと境界 |
|---|---|
| [native](../../scripts/experiments/gemma4-purpose/native.py#L91) | text decoder の scalar/vector と layer scalar。`--virtual` は O 入力 head gain と MLP-down 入力 column gain の微分で、個々の行列座標の微分ではない。実重みへの対応付けは別途必要。serial は native-only。 |
| [matrix](../../scripts/experiments/gemma4-purpose/matrix.py#L240) | q/k/v/o/gate/up/down の実在する全対象行列を併測、各512標本座標。`mlp-triplets` は gate/up の同じ hidden unit 行と down 列をそろえる。全行列の勾配ではなく標本の真の dL/dW。 |
| [matrix_full_units](../../scripts/experiments/gemma4-purpose/matrix_full_units.py#L38) | 明示した1〜32 units、各層最大2 units の gate/up **全行**＋down **全列**。attention は固定した標本座標を併測し native は固定。support JSON に source・attestation・attention coverage/sample plan の SHA が必要。旧勾配や候補判断は不要。 |
| [finite replay](../../src/kairyu_bench/wsp/weights/finite_replay.py#L204) | 同一 base の固定1〜8候補。native scalar/vector＋疎な行列座標を同時変更可能。通常は各行列512、行列合計262144座標が上限。full-unit 拡張も扱うが、その場合は native 混在を拒否。BF16 実値で base→候補→復元 base、per-case margin、fold 平均 NLL・限定位置 KL・A/A を記録（per-case LL は保存されない）。 |
| [module replay](../../scripts/experiments/gemma4_lora/finite_module_replay.py#L62) | donor LoRA の各 projection に固定した符号付き係数を掛ける全行列変更。同一 source/cases/frame を固定した **disjoint native companion** も併用可能。CPU FP32 の固定 merge 演算→BF16 丸めを RAM 上で再現し、実 byte と復元を検証。任意の新しい dense 更新を学習する道具ではない。 |

**実行環境とメモリ。** 指定 CPU venv は NumPy 2.5.4 のみで Torch/Transformers/safetensors は未導入（import はせず存在確認）。5入口の `--help` は全て成功。実計測は対応 Torch・Gemma4 Transformers・safetensors と full BF16 model が必要。collector は CPU 指定もあるが、GPU 指定時は排他承認・露出1枚・空き70 GiB以上を検査する。これは必要条件であり OOM 回避保証ではない。[F02 の既存 finite 実行](records/2026-11-07-weight-combination-capabilities/audit.json)は image `af9e746b9130…`、128 GiB container、serial/text SDPA/2048 tokens/88 GiB allocator cap で完了済み。この証拠を他方式の実行済み証明に転用しない。

[native の CPU 退避](../../scripts/experiments/gemma4-purpose/native.py#L16)は serial/native-only の nonleaf CUDA FP32・2次元以上の保存活性だけ。BF16 全活性や matrix の graph を退避する機能ではない。full-units は gate/up 入力を元 dtype で CPU 保存するが他の graph は残る。collector は gradient checkpointing に未対応。[module replay](../../scripts/experiments/gemma4_lora/finite_module_replay.py#L198)は元BF16＋FP32 delta の `6×対象要素数＋1 GiB` cache、host はさらに32 GiB、cgroup はさらに checkpoint bytes＋16 GiBを要求する（全410行列の運用文書は256 GiB container）。eager 固定で、共通 SDPA policy の入口はない。固定 donor/recipe と信頼済み compiled BF16 merge library も必要。

**native-thinking との距離。** [共通 frame](../../src/kairyu_bench/wsp/weights/finite_replay.py#L308)は thinking-ON の公式 prefix に **empty-thought closure を明示追加**し、正答/foil の公開回答 token だけを teacher-force する。内部思考を読まず生成・採点もしない。native free-thinking の生成後文脈での勾配ではない。native/matrix/full-units/通常 finite の explicit policy は serial・1 case で1024/2048 tokens と text SDPA を扱い、長すぎる完全 frame は切り詰めず拒否する。既定の matrix/full-units は512、native は上限未指定。長さ拡張だけでは思考文脈対応にならない。**通常 finite の CLI だけは `--run` gate がなく、`--help` 以外を preview と誤認して起動しない。**

**次の実測に必要な最小入力／出力（実施・採用は未決定）。** 既存 collector は per-case の LL または margin と fold 平均勾配を保存する。寄与の集中を調べるための per-case 勾配が必要なら、独自作成ケースを1件ずつ fold に配置し、元の集約重みを明示した別入力にする方法は既存 API で可能（旧結果の再解釈やケース除外はしない）。入力はケース全件・目的・frame・support・checkpoint の SHA、出力は case ID 付き微分/数値・実 frame 長・coverage・丸め後変化・A/A。実際の自由生成能力は既存の全問 WSP で測れるので、その確認のための新規実装は不要。

公開の multi-turn/tool 履歴へ条件付けた感度が必要になった場合だけ、現行 [case reader](../../scripts/experiments/gemma4-purpose/recipe_common.py#L78) の `prompt/correct/foil` 境界を拡張する必要がある。最小設計は公開 messages と明示した回答境界を入力、新 frame protocol・完全 prefix/token hash・回答 token mask・per-case LL/margin・復元検証を出力とし、collector/replay 両方で同じ frame を検証すること。内部思考の採取を前提にせず、forced-empty-thought と同条件だとは扱わない。評価問題や正解を校正入力へ移さない。

出典固定: [12 source/doc SHA・5 help・CPU依存・既存実行証拠](records/2026-11-07-weight-combination-capabilities/audit.json)、[検算用 SHA 一覧](records/2026-11-07-weight-combination-capabilities/SHA256SUMS)。help/metadata 採取0.689秒、help失敗0/5。読取り時の存在しない旧ファイル名と表示打切りは audit に記録し、正しい source の限定範囲を再読した。モデル再測定や payload hash 再走査はしていない。
