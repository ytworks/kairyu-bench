# 次の測定に必要な条件：F01/F02の新しい証拠から

2026-11-07、AV dev30の進行中に行った独立検討。AV30の部分得点は参照せず、
候補・変更方向・新しい目標は選ばない。新候補、solve、GPU、生成、ベンチマーク、
既存入力の修正は行っていない。

根拠は[完全dev10比較](2026-11-07-f02-official-results.md)と
[120件監査](2026-11-07-authored-case-audit.md)に限る。F02はAVとコード全26件同点、
probe-code NLLは4.087639→4.155552へ悪化。120件の**正のmargin差の総和**の74.378%が
有効な1 token問題1件に集中した。これはLLや能力利得の74%ではない。
11件の問題は明示dtype契約違反3件と入力域・例外・Git index前提の不足8件であり、
大利得の主因ではなかった。correct優位110／foil優位10というペア順位も不変で、
argmax tokenや自由生成の正解率とは別物である。

**1. 生成した回答を実行して、成功件数の増加を測る。**
[science_generate.py](../../scripts/experiments/gemma4-purpose/science_generate.py)
はpromptのみをnative thinking ONで送り、公開回答、終了理由、request ID、usage、時間、
失敗を保存する。temperature/top_pはserver既定、1件1choice、1〜256件対応。
ただし今回の120件にはsplit欠落56件、train明示64件があり、入力validatorは拒否した。
原120件を保持し、全ID/prompt・元purpose/fold・原SHAを結び付けた新規生成envelopeで
splitを明示する必要がある。CLIのcheckpoint SHA申告とは別に配信実体の証拠が必要。

[science_sandbox.py](../../scripts/experiments/gemma4-purpose/science_sandbox.py)
は既存の5 catalogとの全件・本文一致を要求し、隔離実行で数値、shape、dtype、入力保全、
構文／format／timeout／unknownを区別する。**混合120件全体の汎用採点器ではない**。
任意algorithm、参照追跡、Git workflowには別途明確な入力契約とoracleが要る。
11件の疑義を持つcorrect本文をそのままoracleにせず、将来修正する場合も元版と変更理由を残す。
既存catalogの妥当性も再確認する。公式問題の参照実装や正解は取り込まない。
sandboxは説明付き正答をformat failureにし得るため、WSP sanitizerと同じ得点とは呼ばない。

**2. 通常thinkingの自由生成と、empty-thought条件の尤度を分ける。**
生成CLIで通常thinking ONの公開回答・実行合否を測る機能はある。一方、
[prepare_frames](../../src/kairyu_bench/wsp/weights/finite_replay.py)は
empty-thoughtを明示的に強制し、native/matrix/finiteもこの条件を使う。
[境界診断](../../scripts/experiments/gemma4-purpose/public_answer_diagnostic.py)
は最大3件のtoken順位・境界・head計算を比べる機能であり、思考を生成しない。
既存微分を「実thinking文脈に整合済み」と再解釈できず、同文脈での目的微分は未対応。
公開回答のみを保存する既存方針で、まず自由生成の合否を測ることは可能である。

**3. per-caseの利得集中を事前に監視する。**
native/matrixはper-case marginまたは正答token平均LLを保存する。finiteはper-case marginを
保存するが、正答NLL/KLはfold平均へ集約されるため、保存F02からcase別LLを復元できない。
fold合成gradientもcase別へ分解できず、自動的な影響capやrobust集約は未実装。
既存入力形式で将来**全件を保持しcase別fold**にする測定は可能だが、新しい入力・padding条件
として両モデルを比較し直す必要がある。過去gradientの再解釈にはならない。

次の測定では全件平均を残し、purpose/fold・token長別の得失、正答LL増加とfoil LL低下、
成功件数、top1の正利得比率・残余和を併記する。top1除外値は感度診断であり、元得点を
置換しない。case単位の許容損失／影響上限、反復数・seed・全反復の集約規則は結果を見る前に
固定する必要があり、現時点でその数値や最適化方式を選ぶ根拠は足りない。

**必要な成果物と未解決点。**
全IDについて生成完了／抽出／実行／合否／unknown、入力・code・設定SHA、比較の得失遷移、
完了数・失敗・tokens・手数・時間を残す。1件1choiceだけで生成揺らぎは推定できず、
未採点を0にはしない。通常thinking下の微分、混合120件のoracle、契約修正版、
case別LL記録／影響制限は未解決。AV30全問の結果を見る前に候補を推奨せず、
必要性を確かめてから最小の追加を決める。自作測定は正規WSP全問の代用にしない。

## 今回のCPU確認

sourceとscience_generate --helpを読んだ。純粋なvalidator
`runpy.run_path("scripts/experiments/gemma4-purpose/science_generate.py",
run_name="_source_audit")["public_cases"](json.loads(train_cases_path.read_bytes()), "train")`
は `ValueError: all case splits must match --split`。
HTTP・生成・出力作成0で、当初の「120件をそのまま渡せる」という見込みを訂正した。
原入力SHAは前後とも
`bd6fcba70e93b1870ee882f6e92893b93490ad4099af7a9a290ba3d324d68951`。

<details><summary>確認したsource SHA256</summary>

- science_generate.py: `8d05c960d4b8dbb3112f7fd2214df50b7b31c29c78f07ad1294f021a5fafb8af`
- science_sandbox.py: `b49964b08da7bd219ebafa1f136b77c717cc224c2a73304c9b3024eea888b620`
- native.py: `c9b294242d6e1c8c2786b1f4264dfcaf6d1c8927d576e815c53c96fca51ec108`
- matrix.py: `bcda8de026bb70ca93d821afb2a4987f4bc11b33672d861635459610540e5a22`
- finite_replay.py: `71c424bffac775ba112e709632df0a37351a79c3121827be7c9844f321c7c4d3`

</details>
