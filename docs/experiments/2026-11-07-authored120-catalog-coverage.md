# 著者作成120件の既存 executor 対応と生成入力（2026-11-07）

**全120件を生成対象に保持でき、既存5 catalog で本文・全 row が完全一致する採点対象は48件。残る72件は今回生成記録のみとする。** 生成も採点もまだ実行していない。全ケースは既知の校正資料として扱い、`train`/`transfer` という既存名から未見性を推測しない。これは公式 WSP の問題や分母の変更ではない。

元入力は `analysis/code90-resume-20261106/long-helper-chain-curriculum-v1/train-cases.json`、SHA256 `bd6fcba70e93b1870ee882f6e92893b93490ad4099af7a9a290ba3d324d68951`。19 purpose・各2 fold・120 unique IDs。全 ID と対応区分は [120行表](records/2026-11-07-authored120-catalog-map/all-120-map.md)、全 prompt 原文・各回答 hash・source の purpose/fold/index・catalog ごとの全文一致/差分は [case-map.json](records/2026-11-07-authored120-catalog-map/case-map.json) に残した。

| 既存 catalog | 既存 train/transfer 件数 | 元120とID一致 | 元 row 完全一致 |
|---|---:|---:|---:|
| basic | 16 / 8 | 16 | 16 |
| advanced | 16 / 8 | 16 | 15 |
| advanced-v2 | 16 / 8 | 16 | 16 |
| library-train-v1 | 16 / 0 | 16 | 16 |
| library-transfer-v1 | 0 / 8 | 0 | 0 |

advanced と v2 を重複加算しない。元120は `science-advanced-budget-prefix-cardinality` の prompt が v2 と一致し、旧 advanced とは不一致。旧版への置換はせず、unified diff を case-map に保存した。[現ソースの revision 差](../../scripts/experiments/gemma4-purpose/advanced_science_cases.py#L362)は prompt-only であり、同じ ID だけで同じ問題と判断していない。

4入力の remote root は `analysis/fresh-purpose-20261107-authored120-generation-inputs-v1/inputs/`。同一 bytes の [PRコピー](records/2026-11-07-authored120-catalog-map/inputs/basic-train-cases.json) も保存した。

| 入力 | 件数 | SHA256 |
|---|---:|---|
| basic-train-cases.json | 16 | ed17e06c04c8d6416168e67501b4a07e9490fa4a76d5355a746370447ba99ae4 |
| advanced-v2-train-cases.json | 16 | 9d51dfdeaa9e61a4d989b23459c4eff8b3d5b7d7d9291de3258bd3c12921797b |
| library-train-v1-cases.json | 16 | d3059d4d1c073ba372d103ace9760398341a1add1133e70e306cfefab94db6fc |
| uncataloged72-train-cases.json | 72 | 30606fd3b14b4c7809a488fda2bc7bdf318fead9543228b50e7617b0754ca7be |

新 envelope は `split=train`、元 purpose/fold の配置と各 row の既存全フィールドを保存した。[science_generate](../../scripts/experiments/gemma4-purpose/science_generate.py#L31) は **各 row の split** を要求するため、元に無い56件に限って derived row に `split=train` を追加した。元ケースは未変更で、120→120 の一対一対応・ID重複/欠落0・prompt/correct/foil 不変・元 SHA/stat 継続を検証した。後の生成では各 envelope をそのまま使うので、生成済み responses の `cases_sha256` を書き換えて抽出する必要はない。

CPUで4入力の `public_cases()`、3 catalog の `science_sandbox.read_inputs()` が成功した（後者は **空 response の validator 専用fixture** であり生成結果ではない）。同 validator は complete catalog split と row 全体一致を要求する。元120の split 欠落、旧 advanced への v2 入力は予期通り拒否された。採点コード・oracle・sample・生成関数は呼び出していない。[検証証拠](records/2026-11-07-authored120-catalog-map/validation.json)、[実行した一回処理の原文](records/2026-11-07-authored120-catalog-map/operation.py.txt)、[全SHA](records/2026-11-07-authored120-catalog-map/SHA256SUMS)。主処理0.129秒、失敗0、予期した拒否2。

**分母と契約疑義。** 次の記録は生成完了/失敗/未応答を **120**、実行採点の pass/fail/unknown を **48** で別表示する。48の raw fixture 結果を全て保持し、[既存11反例](2026-11-07-authored-case-audit.md)に該当する3件（`science-cumulative-trapezoid`、`science-advanced-reset-running-stats`、`science-advanced-capped-reservoir-chain`）と残る45件を併記する。現在の sample は basic `n>=5`、advanced `n>=4` で空入力の反例を含まないため、fixture合格は公開契約全域の正しさを保証しない。3件を削除・救済・減点変更しない。11反例の残る8件は未採点72側で、全件表に個別分類を付けた。optimizer の収束や極端入力までの保証も今回新たに行っていない。

未採点72は非science 56件と helper/long chain 16件。後者には別 recipe の `score/evaluate` が存在するが5 catalog CLI外であり、今回は新たな接続やoracleを作らない。固定 prompt の supplied helper と、別測定の generated helper を混ぜない。**48/120 の採点対応率や48件の合格率を、120件全能力得点・公式WSP・未見の能力改善と呼ばない。**

host採点に使える既存 Python は `/mnt/nvme/kairyu/dev-logs/wsp/venv-scicode/bin/python`。旧 `science-library-executor-v1/context/stage-receipt.json` の `source_python` と一致し、現在の NumPy 2.2.6 / SciPy 1.13.1 / SymPy 1.14.0 と SciPy integrate/linalg/optimize/special/stats の import が成功した（1.052秒、新installなし）。[環境証拠](records/2026-11-07-authored120-science-environment/host-science-python.json)は探索時の不適切な symlink 解決を使った probe と、その訂正も記録している。package全bytesの一致や新sandbox実行の証拠とはしていない。モデル・GPU・Docker・サービス・lock 操作は全て0回。
