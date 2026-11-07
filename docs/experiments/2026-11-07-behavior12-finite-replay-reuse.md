# 全12caseの有限再生：既存経路の再利用調査

**最短は、今回実測済みの `fresh-av-native-gpu4-20261107.py` に「finite 1job」modeだけを追加し、既存 `finite_replay.py` をそのまま呼ぶ経路。** 新しい大型ownerは不要。現時点では候補/spec未作成、ソース読取りと有限CLIの `--help`（exit0）だけを行った。コード変更、Docker/model/service操作、有限計測は0。以下は未実装の設計メモである。

`finite_replay.py` は件数を120に固定しておらず、現在のraw A..F→各2 singleton fold（全12）をそのまま受理する。native-only planは `architecture=gemma4_native_bf16`。candidate1件のexact BF16 spec（既存scalar/vector、indices/set_values、非noop、全changed<=1,000,000）を束縛すれば、1ロードで **AV→candidate→AV** を評価する。各sweepは全12correct/foilの24forward、合計72forward。correct token平均NLL、correct−foil margin、各correct回答の最初/最後位置だけのfull-vocabulary KLをcase別に得られる。correct-LLの利得はNLL deltaの符号を反転する。全token KLや自由生成能力の得点と呼ばない。

## そのまま使えるもの・使えないもの

- `fresh-f01-finite.py` はprepareだけでなくvalidate/command/runがGPU6・CPU48-55・600秒・2048tokensに固定される。新planを手作りしてもGPU4/1024同条件にはならない。旧120case条件はprepare固有だが、hardware/policy固定はrunにも残る。さらに今回のownerが修正したownership/late-create/inactive証明を持たない。
- `replay-ad-authored-finite.py` はgemma-2/GPU4と可変fold数を受理し、planのrun_argvへ1024/SDPAを入れられる。しかし6source pins・特殊attestation wrapper・旧runtime helper依存が残り、globalではなくlocal lockのみ。cleanup失敗後にstartし、client終了後のfreshcid再照合もない。このまま再使用する利点は小さい。
- 今回のownerはglobal/GPU4 lock・全27snapshot・他26保護・exact sameCID復帰・plan labels・client終了後のfreshcid確認が実測/CPU故障検証済み。この安全なlifecycleを維持し、job roster/commandと完了検査だけを分岐する方が短い。

## 最小差分の設計（未実装）

1. 既定modeは今のnative二目的のまま。明示 `mode=finite` のときだけstage名を1つにし、command payloadを `-m kairyu_bench.wsp.weights.finite_replay` にする。既存Docker image/UID/networknone/GPU4only/memory128g/read-only入力/NVMe出力/labels/cleanupを共用する。新planでは復帰済みgemma-2の最新StartedAtを取り直す。
2. cases/frame/attestation/現AV tree/index/sourceを既存共通ガードで固定し、finite-planとcandidate specのSHAを追加。finite-plan内のAV treeがattestation treeと一致することも外側で明示確認する。native感度出力を再計測せず、保存済み全12frame/現在の観測policyを束縛する。source保存時の丸め済みBF16をspecへ用い、candidate保存済みならmanifest/verifyを同spec/parentへ束縛する。RAM spec試験はsaved checkpoint全体の検証を代替しない。
3. CPU previewはCLI呼出しではなく、既存の `Checkpoint`, `finite_replay.load_plan`, `AutoTokenizer(local_files_only=True)`, `prepare_frames(max_frame_tokens=1024)` を使ってspec・frameを検証する。GPU無しの既存imageで実施できる。**有限CLIには `--run` gateがないため、引数を渡した通常実行をpreviewと称してはいけない。** 今回実行したCLIは `--help` だけ。
4. finiteのruntime引数は `--answer-batching serial --threads 4 --seed 0 --max-frame-tokens 1024 --text-attention-backend sdpa --cuda-memory-cap-gib 88 --execution-order longest_frame_first --maximum-seconds 600 --gpu-exclusive-approved`。`--max-cases-per-microbatch` はfinite CLIに存在せず、serial intrinsic policyが1を宣言する。`--saved-activation-storage` も存在しない。有限評価はinference_mode/grad無効なのでFP32 saved-autograd退避は適用対象外。coreのframe/serial/SDPA/BF16/seed/capは一致するが、native感度とのforward値一致は実base測定で確かめる。
5. 完了検査をnative-summary用から、全12labels/候補1件/source_stable/全sweepのselected-byte proof exact/base_restoration matchedへ分岐。base/candidate/restored全case、NLL・margin・endpoint KL、全失敗/partialを保持する。finite内は開始から最大秒数−60で通常sweepを中断して復元を試みるが、recovery sweepはそのdeadlineを外す。外側ownerの900秒deadlineとexact collector停止/同CID復帰を維持する。600秒と900秒は別の計測・保護範囲として記録する。

残るblockerはcandidate/specが未作成であることと、この小さい明示mode分岐がまだ無いこと。native-onlyのためfull-unitのmatrix-only制約は関係しない。選択BF16 byte proofは変更座標unionに対する証拠で、全modelのfresh hashではない。その他の能力確認、native再生成、全問公式WSP、Opusレビューの採用判断を有限代理値だけで代替しない。

ソース位置: finite_replay.py load_plan L156–272 / inference evaluator L378–486 / selected-word proof L495–549 / run L552–695 / main L698–720。fresh-f01-finite.py validate L211–277、replay-ad-authored-finite.py validate L99–210とcleanup L507–570。この調査では何も実行許可していない。

## 後続実装（調査時点の記述は保持）

親の実装指示後、[既存ownerの明示finite mode](2026-11-07-behavior12-finite-owner.md)を追加しCPU22tests/Ruffを完了した。上の「未実装」は初回調査時点の履歴。実candidateと実planは未作成、GPU/model/service操作は引き続き0。
