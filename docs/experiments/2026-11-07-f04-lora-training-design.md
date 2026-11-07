# F04からの小規模LoRA学習：最小実装案

設計段階。GPU・forward・依存導入・学習本体の実装は0。F05は固定native選別で親F04から改善せず不採用。累積親は全2338 dev30が完了したF04（code72.9365、overall76.9911）で、旧候補の採用判断は根拠にしない。

- **再利用**：[recipe_common.load_model](../../scripts/experiments/gemma4-purpose/recipe_common.py#L203) のlocal BF16/SDPA/全base凍結と、native.pyのFP32保存活性退避。既存学習loopは無いので、モデル固有recipeへ普通のLinearの低rank A/B・固定AdamW loop・F32factor exportだけを追加する。baseのvision、embedding/head、norm/scalarは凍結。A=random/B=0、最初のA勾配0は正常で、2step以降A/B双方への勾配を検証する。scope/rankは親が観測を見て決定し、全410/rank8は容量見積り用の仮置き。

- **損失**：公式templateのuser+assistant正答content、enable_thinking=Trueをそのままrenderし、人工empty-thoughtを挿入しない。serial exact length・foil/padding無し、公開回答（終了tokenを含むかも事前明示）のみmaskしたCEをoptimizerで何stepも実際に再評価する。promptをlossへ入れない。通常thinking推論と直接回答教師は完全一致しないが、既存forced-empty-thoughtの局所一次近似の反復とは区別できる。templateはreasoning fieldを扱えるものの、既存生成はreasoning_saved=falseで再利用素材が無い。今回は追加reasoning生成も無い。独立public TRAIN/holdoutと非重複の固定データを使い、19校正・WSP答案を新学習へ混ぜない。

- **merge/export**：[dense_lora_recipe factor_format v2](../../scripts/experiments/gemma4_lora/README.md#explicit-factor-format-v2-standalone-dense-only) はrank1–256・F32/BF16を既に受理する。A=(rank,input)、B=(output,rank)、名前はbase_model.model.model.language_model.layers.*.{mlp,self_attn}.*_proj.lora_{A,B}.weight、全820factor。F32 BA→alpha/r→係数→BF16一回丸めと全未変更span検証を再利用する。module_lora_recipeはrank16固定なので使用しない。非対象factorをB0で保持すれば全410形式でexportできるが、非対象のBF16 byte不変は検証し、signed-zeroも隠さない。MLP180だけなら既存structured_scopeのzero-coefficient経路がある。現在の公開donor_repo/revision必須条件を架空値で満たさず、local training receiptを表す小さい明示opt-inが必要。native-only faststoreはmatrix mergeに使えない。

- **数値・メモリ**：F04現物headerの410行列からrank8=61,214,720パラメータ（MLP180=38,707,200、attention230=22,507,520）。F32factor244,858,880B、factor/grad/Adam2momentで0.91217GiB。本体約58.25GiBにactivationとlogitsが加わる。語彙262,144のF32 logitsは回答512tokenで0.5GiB/コピー。旧native max290token時の62.256745GiB peakを1024token学習の保証にしない。既存offloadはFP32 nonleafだけでBF16活性は残る。microbatch1、accumulation4、まず2–4stepの実行でpeak/速度/finite loss・gradを測ってから固定64–128stepへ進める案。入力超過は黙って切詰め/除外せずpreviewを止める。gradient checkpointingを使う場合はnonreentrant・train mode・use_cache=Falseとdropout挙動を別途固定する（現config attention_dropout=0）。まだそのGPU動作は未検証。

- **最小検証・出力**：tinyLinearでzero-initの親一致、2step後A/B勾配、FP32 mergeとの許容誤差内一致、BF16保存丸め差、無関係重み不変を確認する。推論時unmerged adapterと保存BF16mergeのbit一致は仮定しない。新出力限定、parent/data/source/順序seedをpinし、全step loss/gradnorm/LR/tokens/時間/peak/失敗、A/B safetensors/config/学習receiptを先に保存。省略や最良stepだけの採用はしない。既存ownerの排他・同CID復帰を再利用し、大きなownerを追加しない。dense保存は別候補へ約63GB必要、現在約69GBのためadapterを先に出し保存直前にreserveを検査する。

学習速度は未測定。既存F04 native19のcorrect-LLは43.970秒/19forward、margin73.084秒/38forward（load除外、LoRA trainingではない）。歴史的dense+native materializeは430.45秒、別saved verifier112.55秒で、今回rank8の所要時間を保証しない。新しい損失学習がcoding能力を改善するかは保存候補の通常生成と同条件の正規WSPで判断する。

[現物410形状、コード/設定SHA、調査範囲](records/2026-11-07-f04-lora-training-design/source-and-inventory.json)を保存。設計合意までは学習コード本体を作らない。

Historical-stage note: the statements above describe the initial read-only design before implementation. The agreed implementation and CPU validation are now recorded in [the current guide](2026-11-07-f04-lora-training-implementation.md); GPU pilot/continuation results belong to the separate execution record. The original design claims are retained as history.
