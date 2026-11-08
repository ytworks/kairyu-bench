# 凍結0397の最終DeepSWE比較：1/113で元モデルと同点

最終実行は113/113課題の公式採点を完了し、正解は **1/113（0.8849557522%）** でした。元モデルR3と同点で、新規成功1課題・喪失1課題・公式reward不変111課題です。未採点・canonical例外・archive再試行はいずれも0です。正解数の厳密増加というHF登録条件は不成立で、重みのアップロードは行いません。

[WSP固定確認の結果と運用上の除外](2026-11-06-frozen-confirmation.md)とは別の、113課題を32並列で1周する最終評価です。WSPの結果で最終DeepSWEを置き換えません。

## 凍結した候補と判定

候補 `resume20261106-h-grid-0397` は2026-11-05 19:04:10 UTCに、確認評価の結果を見る前に選択を凍結しました。確認評価後の重み再選抜・再調整は行っていません。

| 項目 | 凍結値 |
| --- | --- |
| 変更要素 / tensor数 | 9,127 / 90 |
| 元重みとの差のFrobenius norm | 0.0004125233976985216 |
| checkpoint全ファイルtree SHA-256 | `ba5e6ab526c8c08182a030f459f4acb532d5f50ebc1d35e4745b09fe59a8768d` |
| spec SHA-256 | `08607ec221d8d7a9d3bf15c04e33fc604c743f69c9293b591bb49dd894435e4b` |
| 重み編集manifest SHA-256（変更前後の値・spec対応） | `26aab17df0f2f0dce4b3df76339fdbae3d14cc8d3a590bbb1449521f92c2e4b0` |

上表のmanifestは実験rootの `manifests/resume20261106-h-grid-0397.json` です。重み編集を検証する記録であり、WSPの確認cohort manifest（SHA-256先頭 `52a760`）やDeepSWEのselected/trial-planとは別のファイルです。

比較の主結果は各課題の公式rewardと113課題の公式pass@1です。全113課題の採点完了を確認し、候補の正解数が元モデルの **1/113を厳密に上回る場合だけ** HF登録の条件を満たすものとします。WSPの加点、部分テストの合計、tokens削減をこの条件の代用にはしません。未採点・例外はnullのまま保持し、0に補完しません。1周ずつの比較だけから一般的な能力差や重み変更の因果効果を確定しません。

## 実行と比較条件

| 項目 | 元モデルR3 | 凍結0397 |
| --- | --- | --- |
| run ID | `gemma4-base-c32-thinking-20261104-r3` | `gemma4-h0397-c32-thinking-20261106` |
| 状態 | completed、113/113採点 | completed、113/113採点 |
| 課題 / 試行 | DeepSWE v1.1、113課題、各1試行 | 同一113課題・slot、各1試行 |
| workers | 32 | 32 |
| agent / 通信 | mini-swe-agent 2.4.6 / Chat Completions | 同一 |
| 推論設定 | native thinking ON、追加thinking capなし | 同一。観測collectorなし |
| 配信構成 | Gemma 4 31B、8GPU、TP2×4 replica、MTP4 | 同構成 |
| API timeout / 最大retries | 3,600秒 / 3 | 同一 |
| task timeout / agent setup倍率 | 1 / 4 | 同一 |
| mini step_limit / cost_limit | 0 / 0（agent側追加上限なし） | 同一 |
| miniへの明示max_tokens / reasoning_effort | 追加指定なし | 同一 |
| 完了後の全trial・checkpoint検証 | R3既存完成監査 | 全113 UUID/slot/job一致、全settings一致、stat継続性確認済み |

0397のrun開始は2026-11-05 20:38:09.748693 UTC、終了は同日22:26:14.905618 UTC（JSTは11月6日05:38〜07:26）です。「追加capなし」は通常のcontext、出力、タスク時間の制限がなくなる意味ではありません。数値effort=100として実行したものでもありません。miniの実行に、別agentのDeepSeek Harness用max_tokens=49,152を適用したとは記載しません。

DeepSWE datasetは `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`、Pier 0.3.1は `0c802fc067a425345b24d1c69411aa98acf61a1d`、mini-swe-agent 2.4.6は `a83fcae82d2a08f0ee0c688f9d137b3566c097f8` に固定されています。今回の起動時bench commitは `3c9bfd07fac0b3c53e2de0898a6a656849d160e3` です。

開始時監査に加え、終了後もselectedの113課題、trial-planのslot、runtime.json、全113 UUID/jobと解決済みagent設定がR3との比較条件を満たすことを確認しました。113 task imageのRepoDigestsもR3記録と一致しました。run・normalized result・report・公式summaryはすべてcompletedで、専用task/proxy containerの残数は0でした。終了後のcheckpoint確認は、元モデルと候補の各12ファイルinventory/statの継続性に基づいて、凍結時に確定したcheckpoint hashを再利用しています。元モデルのfull shard hashは以前のcaptureを再利用した記録で、候補は凍結時に全fileを再計算しています。終了後にfull payloadを新たにhashし直したわけではありません。全4配信replicaのidentity・image・command・read-only mount・health、専用daemon identity、runtime/scorer9ファイルとwrapperも継続一致しました。R3開始時の配信Docker image IDは歴史記録にないため、当時の配信binaryまで同一だったとは断定できません。

## 集計の定義と最終値

`agent_steps` はcanonicalの `agent_result.n_agent_steps` です。API列は、canonical UUIDとjob UUIDを照合したtrajectory先頭の `info.model_stats.api_calls` であり、手数から推定していません。元R3は後者も113/113件にあり、**8,306 recorded API calls** と **8,274 agent steps** は異なる値でした。これはagent内の記録counterであり、独立したHTTP・gateway・server telemetryではありません。L1やtransport再試行数は未計測です。

入力・出力tokensはcanonical `n_input_tokens` / `n_output_tokens` の記録です。canonicalとarchiveのusageはtrial UUID単位で識別し、同じUUIDを二重加算せず、再試行の追加消費と課題の正解数も混同しません。存在しないusageやcounterはunknownとし、0にはしません。

| 指標 | 元モデルR3 | 凍結0397 |
| --- | ---: | --- |
| 採点済み / 計画課題 | 113 / 113 | 113 / 113 |
| 公式正解数 | 1 | 1 |
| 公式pass@1 | 0.8849557522%（1/113） | 0.8849557522%（1/113） |
| 未採点 | 0 | 0 |
| unique canonical trial UUID | 113 | 113 |
| 記録済みagent steps | 8,274（113/113） | 8,071（113/113） |
| agent記録API calls | 8,306（113/113） | 8,098（113/113） |
| 独立HTTP / L1回数 | unknown | unknown |
| 入力tokens | 379,689,393（113/113） | 346,460,961（113/113） |
| 出力tokens | 3,488,483（113/113） | 3,143,746（113/113） |
| run wall time | 7,608.619834秒 | 6,485.156925秒 |
| archive result / archiveだけの追加UUID | 0 / 0 | 0 / 0 |
| HTTP再試行回数 | unknown | unknown |
| canonical例外 | 0 | 0 |
| 0-byte patch / 非空patch / 欠測 | 76 / 37 / 0 | 73 / 40 / 0 |
| patch bytes合計 | 19,581,557 | 870,644 |
| 公式F2P passed / total合計 | 625 / 5,877 | 473 / 5,877 |
| 公式P2P passed / total合計 | 226,371 / 231,352 | 78,067 / 231,352 |

F2P/P2Pは公式verifier reward metadataを転記するだけで再採点しません。異なる課題のテスト数を合算した値は診断用であり、DeepSWEの課題単位scoreではありません。patch bytesはファイルサイズであり、正しさ・変更行数・有効な実装量を示しません。欠落patchと実在する0-byte patchも区別します。

## 実測された得失

| 課題 | 公式reward 元→0397 | agent steps | agent記録API | patch bytes | F2P | P2P |
| --- | --- | --- | --- | --- | --- | --- |
| true-myth-iterable-collection-combinators | 0→1 | 54→48 | 54→49 | 293,712→19,850 | 17/96→96/96 | 115/561→561/561 |
| claude-code-by-agents-recursive-delegation | 1→0 | 33→29 | 33→30 | 27,834→0 | 7/7→0/7 | 31/31→31/31 |

True Mythの公開要件はMaybe/Result/Taskのiterator、sequence/traverse/zip/zipWith、短絡・直列処理、compact/filterMap/partition、retry/tapと型間combinatorです。元R3では部分テストだけを通過していましたが、0397では公式F2P/P2Pがすべて通過しました。一方、Claude再帰委任では回収patchが0-byteとなりF2P7件を失いました。下記のpatch回収仕様に照らし、0-byteだけから作業内容を推測しません。

WSP固定確認では公式codeが+5.1190 percentage pointsでしたが、この最終DeepSWEでは課題単位の正解数増加に移りませんでした。これを全般的なコーディング能力の向上確定とは呼びません。課題別の利得と損失が併存する結果です。

記録上の手数は203、agent API callsは208、入力tokensは33,228,432、出力tokensは344,737減りました。wall timeも1,123.462909秒短くなりました。ただし1周ずつの負荷・生成経路を含む観測差であり、速度の因果的改善を証明しません。

部分テストにも重要な退行があります。公式F2Pは625/5,877→473/5,877（152件減）、公式P2Pは226,371/231,352→78,067/231,352（148,304件減）でした。特にexpr-try-catch-errorsは66,265→140、meriyah-explicit-resource-declarationsは51,469→1,454、mashumaro-flattened-dataclass-fieldsは29,877→0でした。いずれも元・候補の課題rewardは0で、これらはscoreの新規喪失ではなく部分テストの退行です。異なる規模のtest suiteの合計を能力の加重scoreにしません。原因を数値だけから断定しません。

## 公開課題とpatch回収の解釈

課題は短いコード生成だけでなく、既存repositoryの変更と公式verifierによる確認を要求します。保存済みの公開課題要約には、例えば次があります。

- `abs-module-cache-flags`: 同一ファイルのmodule cache共有、探索path、cache情報/reset、循環import検出、関連CLI flags。
- `etree-xml-diff-patch`: XML構造の比較、差分適用・逆patch、3方向mergeと競合解決。
- `geo-shapeindex-serialization`: shape IDと空間cellを保った保存/復元、再Buildなしの検索、破損入力の安全な処理。
- `cliffy-config-file-parsing`: 設定読込・型変換・継承とCLI/環境変数/設定の優先順、false/0の保持。

元R3の唯一の公式成功は `claude-code-by-agents-recursive-delegation` です。この課題の固定task設定は、agent終了後に **base commit treeとHEAD commit treeのbinary差分** をpatchとして回収します。未commitのstaged/unstaged変更やuntracked fileはこの回収対象に入りません。Pierが自動でstage/commitするわけでもありません。

したがって、この課題の0-byte patchだけを見て「一切編集しなかった」と断定することはできません。回収hook失敗でも空の提出になり得ますが、今回その失敗が起きたと決めつけることもできません。この性質は固定した当該task設定のものです。全Pier課題が同じ回収方法だとは一般化しません。公式rewardはそのまま保持し、回収方法を変更した再評価は行いません。

公開記録は、公式reward、usage、patchサイズと既存の公開課題要約に限定しています。モデルのreasoning本文、toolの生コマンド、gold/reference patch、binary core内容は公開しません。

## 完了判定と公開範囲

全113件の採点・同一性・usage coverageを照合し、候補は固定したまま評価を完了しました。成功数1対1のためHF登録条件は不成立で、アップロードはありません。[Wikiの最終記録](https://github.com/ytworks/kairyu-bench/wiki/DeepSWE-Comparison#gemma4-h0397-c32-thinking-20261106)には、得失と条件限界を含む結果、および元R3のusageラベル訂正を反映しました。Wiki commitは `9a5335927336e339ebaccc272110e52ca61d1369` で、push後のremote一致とpage SHAを `wiki-publication.json` に記録しています。

## 証跡

以下は実験rootからの相対pathです。巨大な生出力をこの文書へ埋め込まず、元の証跡を保持します。

- `analysis/final-h0397-confirmation-20261106/selection-freeze.json`
- `analysis/final-h0397-confirmation-20261106/checkpoint-freeze.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-initial-parity.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-final-parity.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-final-integrity.json`
- `analysis/final-h0397-confirmation-20261106/hf-publication-decision.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-final-recorded-usage.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-candidate-public-summary.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-paired-public-summary.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-baseline-recorded-api-usage.json`
- `analysis/final-h0397-confirmation-20261106/tools-api-metadata-final-checks.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-baseline-public-summary.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-baseline-recorded-usage.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-baseline-api-metadata-audit.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-patch-collection-semantics.json`
- `analysis/final-h0397-confirmation-20261106/deepswe-public-progress.json`（途中経過、最終summaryではない）
- `analysis/final-h0397-confirmation-20261106/wiki-baseline-usage-correction.md`（元の訂正案）
- `analysis/final-h0397-confirmation-20261106/wiki-publication.json`（適用・公開・remote一致）
- `runtime/resume-20261106/deepswe-final-command.json`
- `analysis/prefetch-tool-port-20261106/validation.json`

元R3と候補の最終usageには[既存recipe](scripts/deepswe-recorded-usage.py)のopt-in `--agent-api-metadata`（commit `da7dace24579839942e1126757003ff7129b5898`）を使いました。各113 final UUIDをcanonical/jobに照合し、先頭infoのAPI counterだけを読み、messages前で停止しています。fixture・実元R3検証済みで、先行の一度限りのmetadata監査ともAPI8,306が一致しました。先行canonical summaryのAPI unknownを過去に遡って書き換えず、補足の証拠を保持しています。元8,306と候補8,098はいずれも113/113 coverageです。

## 課題別の比較表（元R3 → 凍結0397）

全113課題のcanonical resultと公式verifier metadataをtask ID・checksumで対応させました。API列はagent内の記録counterであり、独立HTTP telemetryではありません。F2P/P2Pはpassed/totalです。全行の数値は元R3→0397の順です。

| 課題 | 公式reward | agent steps | 記録API | 入力tokens | 出力tokens | patch bytes | F2P | P2P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs-module-cache-flags | 0 → 0 | 69 → 98 | 69 → 98 | 2,135,112 → 2,895,561 | 17,190 → 20,706 | 9,854 → 0 | 12/20 → 0/20 | 2/3 → 3/3 |
| abs-stepped-slices | 0 → 0 | 76 → 70 | 76 → 71 | 3,426,537 → 3,026,540 | 57,443 → 22,119 | 0 → 0 | 0/6 → 0/6 | 6/6 → 6/6 |
| actionlint-action-pinning-lint | 0 → 0 | 78 → 125 | 79 → 125 | 2,831,981 → 6,535,719 | 21,556 → 25,849 | 0 → 17,655 | 0/55 → 51/55 | 145/145 → 143/145 |
| adaptix-name-mapping-aliases | 0 → 0 | 89 → 121 | 90 → 121 | 4,039,051 → 6,961,575 | 32,749 → 41,039 | 0 → 0 | 0/44 → 0/44 | 2,738/2,738 → 2,738/2,738 |
| aiomonitor-task-snapshots-diff | 0 → 0 | 83 → 37 | 84 → 37 | 2,649,894 → 1,231,891 | 25,068 → 32,841 | 0 → 0 | 0/53 → 0/53 | 8/8 → 8/8 |
| anko-default-function-arguments | 0 → 0 | 113 → 134 | 115 → 134 | 7,051,208 → 6,350,172 | 63,735 → 31,044 | 0 → 8,056 | 0/2 → 0/2 | 119/119 → 26/119 |
| anko-typed-variable-bindings | 0 → 0 | 108 → 79 | 108 → 79 | 5,544,404 → 3,298,153 | 23,754 → 22,111 | 0 → 0 | 0/9 → 0/9 | 94/94 → 94/94 |
| arcane-drift-detection-baselines | 0 → 0 | 38 → 40 | 38 → 40 | 963,668 → 1,449,345 | 25,061 → 23,067 | 0 → 0 | 0/82 → 0/82 | 2/2 → 2/2 |
| arktype-json-schema-refs-dependencies | 0 → 0 | 79 → 65 | 80 → 66 | 4,858,040 → 2,844,178 | 25,279 → 41,779 | 0 → 0 | 0/25 → 0/25 | 1,679/1,679 → 1,679/1,679 |
| awilix-async-container-initialization | 0 → 0 | 180 → 132 | 180 → 133 | 12,546,516 → 8,537,674 | 46,264 → 28,115 | 0 → 0 | 0/24 → 0/24 | 162/162 → 162/162 |
| bandit-incremental-cache-control | 0 → 0 | 72 → 30 | 72 → 30 | 3,262,179 → 838,975 | 47,571 → 17,859 | 0 → 23,135 | 0/88 → 5/88 | 275/275 → 0/275 |
| bandit-interprocedural-taint-checks | 0 → 0 | 45 → 79 | 45 → 80 | 1,803,920 → 4,290,862 | 25,500 → 38,280 | 0 → 0 | 0/66 → 0/66 | 293/293 → 293/293 |
| bandit-structured-nosec-directives | 0 → 0 | 58 → 28 | 60 → 28 | 2,271,919 → 1,029,436 | 26,209 → 41,599 | 13,947 → 0 | 32/69 → 0/69 | 275/282 → 282/282 |
| boa-hierarchical-evaluation-cancellation | 0 → 0 | 154 → 82 | 155 → 82 | 8,822,540 → 2,939,853 | 38,972 → 19,829 | 0 → 62,802 | 0/17 → 0/17 | 7/7 → 0/7 |
| cattrs-partial-structuring-recovery | 0 → 0 | 77 → 93 | 77 → 94 | 1,859,696 → 3,367,550 | 24,135 → 48,197 | 8,860 → 0 | 24/69 → 0/69 | 7/7 → 7/7 |
| clack-async-autocomplete-options | 0 → 0 | 32 → 28 | 33 → 28 | 1,209,713 → 1,157,898 | 33,011 → 24,132 | 0 → 0 | 0/82 → 0/82 | 643/643 → 643/643 |
| claude-code-by-agents-recursive-delegation | 1 → 0 | 33 → 29 | 33 → 30 | 746,923 → 715,111 | 18,738 → 16,424 | 27,834 → 0 | 7/7 → 0/7 | 31/31 → 31/31 |
| cliffy-config-file-parsing | 0 → 0 | 147 → 104 | 148 → 104 | 6,217,954 → 4,139,884 | 33,367 → 19,815 | 0 → 8,698 | 0/37 → 0/37 | 451/451 → 451/451 |
| csstree-shorthand-expansion-compression | 0 → 0 | 38 → 56 | 38 → 56 | 1,027,489 → 2,173,850 | 21,678 → 37,593 | 18,807 → 0 | 61/79 → 0/79 | 16,715/16,715 → 16,715/16,715 |
| dasel-html-document-format | 0 → 0 | 49 → 62 | 49 → 62 | 1,344,367 → 2,077,910 | 28,932 → 36,506 | 13,920 → 17,288 | 93/146 → 4/146 | 1,012/1,012 → 1,012/1,012 |
| dateutil-rfc5545-timezone-interop | 0 → 0 | 66 → 69 | 66 → 69 | 2,018,587 → 2,196,319 | 31,295 → 31,910 | 0 → 0 | 0/67 → 0/67 | 2,035/2,035 → 2,035/2,035 |
| drizzle-orm-window-function-builders | 0 → 0 | 60 → 56 | 60 → 56 | 2,277,951 → 2,277,979 | 25,203 → 18,420 | 0 → 0 | 0/130 → 0/130 | 566/566 → 566/566 |
| dynamodb-toolbox-conditional-attribute-requirements | 0 → 0 | 82 → 114 | 83 → 115 | 3,654,933 → 7,314,916 | 22,395 → 34,822 | 0 → 0 | 0/31 → 0/31 | 1,267/1,267 → 1,267/1,267 |
| dynamodb-toolbox-lazy-recursive-schemas | 0 → 0 | 107 → 107 | 107 → 107 | 4,037,975 → 4,734,680 | 23,175 → 27,787 | 0 → 0 | 0/37 → 0/37 | 1,267/1,267 → 1,267/1,267 |
| effect-sse-httpapi-streaming | 0 → 0 | 35 → 64 | 35 → 64 | 1,023,440 → 2,363,093 | 13,256 → 18,809 | 0 → 0 | 0/47 → 0/47 | 70/70 → 70/70 |
| eicrud-keyset-pagination-cursor | 0 → 0 | 45 → 87 | 45 → 87 | 1,475,405 → 4,911,693 | 12,745 → 33,692 | 6,154 → 37,906 | 0/14 → 0/14 | 161/168 → 160/168 |
| etree-xml-diff-patch | 0 → 0 | 35 → 31 | 35 → 31 | 636,954 → 455,766 | 25,419 → 12,817 | 0 → 24,044 | 0/52 → 0/52 | 15/15 → 0/15 |
| expr-try-catch-errors | 0 → 0 | 137 → 72 | 138 → 73 | 7,708,707 → 2,291,573 | 32,216 → 22,694 | 0 → 12,562 | 0/79 → 0/79 | 66,265/66,265 → 140/66,265 |
| fastapi-deprecation-response-headers | 0 → 0 | 123 → 141 | 123 → 141 | 6,348,302 → 7,048,942 | 45,179 → 27,605 | 20,421 → 0 | 0/137 → 0/137 | 0/3,134 → 3,134/3,134 |
| fastapi-implicit-head-options | 0 → 0 | 80 → 132 | 80 → 132 | 2,694,476 → 6,928,719 | 16,360 → 30,345 | 0 → 57,475 | 0/43 → 11/43 | 3,134/3,134 → 0/3,134 |
| fd-deterministic-multi-key-sorting | 0 → 0 | 66 → 68 | 66 → 69 | 4,224,830 → 3,657,936 | 36,923 → 36,347 | 0 → 0 | 0/43 → 0/43 | 109/109 → 109/109 |
| geo-shapeindex-serialization | 0 → 0 | 97 → 140 | 97 → 140 | 4,323,897 → 4,531,396 | 23,777 → 19,562 | 0 → 11,814 | 0/24 → 4/24 | 599/599 → 599/599 |
| go-critic-doc-link-checker | 0 → 0 | 87 → 51 | 88 → 51 | 3,865,313 → 1,464,731 | 61,955 → 19,284 | 0 → 0 | 0/3 → 0/3 | 16/16 → 16/16 |
| go-genai-streamed-function-args | 0 → 0 | 38 → 79 | 39 → 79 | 594,364 → 2,799,197 | 20,523 → 34,635 | 0 → 0 | 0/6 → 0/6 | 62/62 → 62/62 |
| go-git-worktree-merge-conflicts | 0 → 0 | 84 → 83 | 84 → 83 | 1,612,033 → 1,710,608 | 15,898 → 16,915 | 19,445 → 12,050 | 0/17 → 0/17 | 0/2 → 0/2 |
| goreleaser-retry-publish-auditing | 0 → 0 | 54 → 75 | 54 → 75 | 1,744,519 → 3,496,903 | 12,505 → 29,872 | 30,532 → 0 | 1/29 → 0/29 | 29/29 → 29/29 |
| gql-incremental-graphql-delivery | 0 → 0 | 59 → 79 | 59 → 79 | 1,976,439 → 2,814,013 | 21,367 → 20,171 | 0 → 16,156 | 0/17 → 0/17 | 811/811 → 203/811 |
| happy-dom-abort-pending-body-reads | 0 → 0 | 61 → 65 | 61 → 65 | 3,483,858 → 2,734,078 | 16,543 → 22,260 | 5,185 → 13,568 | 1/14 → 1/14 | 165/165 → 165/165 |
| happy-dom-deterministic-intersectionobserver | 0 → 0 | 28 → 33 | 28 → 33 | 676,088 → 1,068,729 | 15,452 → 18,554 | 0 → 0 | 0/14 → 0/14 | 9/9 → 9/9 |
| helm-array-merge-strategies | 0 → 0 | 146 → 76 | 146 → 76 | 10,680,284 → 2,907,132 | 49,544 → 30,337 | 0 → 0 | 0/47 → 0/47 | 12/12 → 12/12 |
| helm-unified-manifest-stream | 0 → 0 | 74 → 71 | 74 → 71 | 4,016,448 → 3,199,400 | 17,977 → 20,082 | 6,332 → 0 | 0/5 → 0/5 | 1/2 → 2/2 |
| httpx-deterministic-cookie-store | 0 → 0 | 41 → 70 | 41 → 70 | 998,874 → 1,784,428 | 24,794 → 20,434 | 0 → 0 | 0/115 → 0/115 | 1,281/1,281 → 1,281/1,281 |
| httpx-multipart-response-parsing | 0 → 0 | 88 → 103 | 88 → 103 | 3,547,718 → 3,956,045 | 55,900 → 37,133 | 0 → 16,989 | 0/122 → 111/122 | 1,272/1,272 → 1,271/1,272 |
| httpx-streaming-json-iteration | 0 → 0 | 74 → 59 | 74 → 59 | 2,490,865 → 2,773,000 | 38,391 → 46,659 | 0 → 0 | 0/108 → 0/108 | 1,404/1,404 → 1,404/1,404 |
| igel-persist-feature-schema | 0 → 0 | 50 → 65 | 50 → 66 | 1,386,139 → 2,051,120 | 20,692 → 24,220 | 0 → 0 | 0/24 → 0/24 | 2/2 → 2/2 |
| ink-grid-box-layout | 0 → 0 | 63 → 94 | 63 → 95 | 3,013,867 → 4,173,170 | 29,761 → 24,182 | 20,736 → 0 | 1/25 → 0/25 | 45/49 → 49/49 |
| ipython-session-bundle-replay | 0 → 0 | 46 → 58 | 46 → 58 | 1,229,280 → 1,536,976 | 34,390 → 19,136 | 0 → 0 | 0/17 → 0/17 | 29/29 → 29/29 |
| katex-multicolumn-array-spans | 0 → 0 | 65 → 67 | 65 → 67 | 3,570,715 → 3,180,364 | 17,163 → 24,751 | 0 → 0 | 0/94 → 0/94 | 599/599 → 599/599 |
| kcp-go-multiplexed-kcp-streams | 0 → 0 | 39 → 24 | 39 → 24 | 1,202,900 → 624,849 | 45,275 → 35,723 | 12,238 → 13,987 | 25/30 → 0/30 | 12/12 → 0/12 |
| kea-atomic-signal-selectors | 0 → 0 | 26 → 53 | 26 → 53 | 508,853 → 1,620,875 | 21,829 → 28,419 | 36,886 → 0 | 1/12 → 0/12 | 95/139 → 139/139 |
| kgateway-consistent-hash-policy | 0 → 0 | 44 → 39 | 44 → 39 | 1,387,814 → 892,160 | 18,414 → 13,633 | 0 → 18,389 | 0/2 → 0/2 | 214/214 → 0/214 |
| kombu-single-active-consumer-priority | 0 → 0 | 57 → 69 | 58 → 69 | 2,482,133 → 3,067,918 | 48,580 → 42,135 | 0 → 0 | 0/85 → 0/85 | 1,421/1,421 → 1,421/1,421 |
| kombu-virtual-queue-dead-lettering | 0 → 0 | 25 → 107 | 25 → 107 | 593,328 → 7,390,206 | 23,426 → 57,316 | 0 → 0 | 0/76 → 0/76 | 1,412/1,412 → 1,412/1,412 |
| koota-composite-trait-aspects | 0 → 0 | 106 → 79 | 107 → 79 | 6,386,952 → 2,939,141 | 39,438 → 25,664 | 0 → 0 | 0/51 → 0/51 | 172/172 → 172/172 |
| koota-deferred-mutation-buffer | 0 → 0 | 89 → 93 | 89 → 93 | 3,475,970 → 4,370,844 | 37,762 → 35,580 | 15,532 → 26,142 | 44/71 → 10/71 | 128/128 → 65/128 |
| koota-entity-snapshot-rollback | 0 → 0 | 55 → 76 | 55 → 77 | 2,158,490 → 3,156,401 | 31,107 → 27,116 | 0 → 0 | 0/84 → 0/84 | 47/47 → 47/47 |
| koota-pair-relation-tracking | 0 → 0 | 85 → 78 | 85 → 78 | 4,803,498 → 3,300,367 | 46,079 → 40,051 | 0 → 44,129 | 0/38 → 0/38 | 172/172 → 54/172 |
| koota-query-predicates | 0 → 0 | 87 → 60 | 88 → 62 | 3,659,385 → 2,253,498 | 36,892 → 25,749 | 0 → 30,543 | 0/43 → 3/43 | 172/172 → 81/172 |
| kysely-window-grouping-helpers | 0 → 0 | 71 → 53 | 71 → 53 | 1,981,363 → 1,399,182 | 22,407 → 17,720 | 0 → 0 | 0/254 → 0/254 | 22/22 → 22/22 |
| langchain-request-coalescing | 0 → 0 | 61 → 49 | 61 → 49 | 1,522,321 → 1,142,241 | 37,281 → 27,063 | 0 → 20,230 | 0/50 → 40/50 | 232/232 → 0/232 |
| mashumaro-flattened-dataclass-fields | 0 → 0 | 82 → 82 | 83 → 82 | 3,564,672 → 3,681,115 | 38,695 → 42,203 | 71,688 → 70,984 | 14/66 → 0/66 | 29,877/30,014 → 0/30,014 |
| meriyah-explicit-resource-declarations | 0 → 0 | 176 → 134 | 176 → 134 | 13,709,023 → 9,631,689 | 31,137 → 23,680 | 0 → 42,003 | 0/49 → 0/49 | 51,469/51,469 → 1,454/51,469 |
| mnamer-daemon-watch-lifecycle | 0 → 0 | 69 → 29 | 69 → 29 | 2,906,099 → 700,533 | 40,005 → 20,932 | 24,493 → 31,501 | 18/51 → 31/51 | 293/319 → 319/319 |
| mobly-grouped-test-barriers | 0 → 0 | 59 → 56 | 61 → 56 | 2,726,613 → 2,254,917 | 53,708 → 33,011 | 0 → 0 | 0/79 → 0/79 | 808/808 → 808/808 |
| narwhals-rolling-window-suite | 0 → 0 | 93 → 81 | 93 → 81 | 2,482,376 → 2,552,843 | 22,927 → 34,599 | 0 → 0 | 0/103 → 0/103 | 10,093/10,093 → 10,093/10,093 |
| numba-stencil-boundary-modes | 0 → 0 | 63 → 33 | 63 → 33 | 2,693,520 → 550,145 | 38,004 → 15,324 | 0 → 4,748 | 0/29 → 0/29 | 827/827 → 0/827 |
| obsidian-linter-auto-table-of-contents | 0 → 0 | 39 → 24 | 39 → 24 | 865,404 → 683,666 | 21,350 → 22,915 | 7,475 → 12,470 | 0/41 → 36/41 | 582/1,131 → 1,131/1,131 |
| obsidian-linter-link-format-conversion | 0 → 0 | 22 → 35 | 22 → 35 | 552,207 → 948,050 | 20,598 → 18,344 | 0 → 0 | 0/60 → 0/60 | 1,131/1,131 → 1,131/1,131 |
| obsidian-linter-scoped-ignore-markers | 0 → 0 | 24 → 45 | 24 → 51 | 792,382 → 2,576,795 | 15,493 → 70,535 | 18,045 → 0 | 0/33 → 0/33 | 1,118/1,133 → 1,133/1,133 |
| ofetch-per-origin-circuit-breaker | 0 → 0 | 29 → 18 | 30 → 18 | 1,232,174 → 281,900 | 30,586 → 20,517 | 0 → 0 | 0/47 → 0/47 | 13/13 → 13/13 |
| onedump-dump-encryption-pipeline | 0 → 0 | 34 → 25 | 34 → 25 | 757,516 → 363,435 | 23,708 → 17,661 | 24,460 → 0 | 72/82 → 0/82 | 6/6 → 6/6 |
| opa-rego-rule-profiling | 0 → 0 | 117 → 174 | 117 → 174 | 5,948,851 → 8,230,367 | 25,430 → 35,389 | 0 → 14,728 | 0/25 → 0/25 | 6/6 → 0/6 |
| opa-template-string-reconstruction | 0 → 0 | 121 → 206 | 121 → 206 | 4,148,645 → 12,541,972 | 18,605 → 30,064 | 1,021 → 1,805 | 0/5 → 0/5 | 4/4 → 0/4 |
| optique-conditional-option-dependencies | 0 → 0 | 70 → 91 | 70 → 91 | 2,777,554 → 5,456,523 | 20,854 → 19,787 | 0 → 0 | 0/36 → 0/36 | 2,034/2,034 → 2,034/2,034 |
| oxvg-structural-selector-preservation | 0 → 0 | 57 → 40 | 57 → 40 | 2,228,204 → 1,887,002 | 8,855 → 18,762 | 3,556 → 14,338 | 0/6 → 0/6 | 0/62 → 0/62 |
| participle-grammar-conflict-analysis | 0 → 0 | 51 → 43 | 51 → 43 | 1,759,365 → 783,195 | 26,530 → 18,033 | 0 → 0 | 0/91 → 0/91 | 153/153 → 153/153 |
| pebble-durability-wait-apis | 0 → 0 | 98 → 104 | 98 → 104 | 5,104,347 → 6,031,013 | 21,881 → 19,773 | 0 → 0 | 0/59 → 0/59 | 44/44 → 44/44 |
| pest-character-class-coalescing | 0 → 0 | 107 → 141 | 107 → 141 | 5,534,869 → 7,850,377 | 34,920 → 32,070 | 0 → 0 | 0/104 → 0/104 | 250/250 → 250/250 |
| prometheus-transactional-reload-status | 0 → 0 | 87 → 107 | 87 → 107 | 3,581,509 → 3,838,257 | 13,315 → 16,181 | 0 → 24,709 | 0/15 → 0/15 | 82/82 → 0/82 |
| prometheus-typed-label-sorting | 0 → 0 | 55 → 41 | 55 → 41 | 1,028,252 → 1,212,925 | 18,149 → 22,997 | 9,025 → 7,261 | 0/17 → 0/17 | 28/28 → 0/28 |
| psd-tools-blend-range-api | 0 → 0 | 35 → 56 | 35 → 57 | 573,763 → 1,377,801 | 15,477 → 24,951 | 0 → 0 | 0/45 → 0/45 | 979/979 → 979/979 |
| pwntools-tube-multiplexing | 0 → 0 | 21 → 26 | 21 → 27 | 381,571 → 631,553 | 21,523 → 21,995 | 0 → 0 | 0/73 → 0/73 | 1/1 → 1/1 |
| python-statemachine-state-data-scoping | 0 → 0 | 135 → 85 | 136 → 85 | 9,812,578 → 2,905,054 | 47,008 → 25,795 | 0 → 9,587 | 0/72 → 3/72 | 1,286/1,286 → 659/1,286 |
| query-persist-restored-query-state | 0 → 0 | 50 → 95 | 50 → 95 | 1,179,302 → 3,857,311 | 11,443 → 22,873 | 3,149 → 0 | 3/8 → 0/8 | 42/42 → 42/42 |
| quill-shared-toolbar-focus | 0 → 0 | 66 → 31 | 67 → 31 | 3,970,869 → 994,648 | 45,066 → 24,625 | 6,148 → 0 | 6/13 → 0/13 | 21/22 → 22/22 |
| returns-validated-error-accumulation | 0 → 0 | 41 → 70 | 41 → 71 | 1,222,668 → 2,359,984 | 22,837 → 38,466 | 0 → 0 | 0/159 → 0/159 | 61/61 → 61/61 |
| scc-bounded-memory-spilling | 0 → 0 | 93 → 113 | 93 → 113 | 3,767,600 → 5,047,570 | 25,127 → 24,830 | 8,632 → 0 | 26/31 → 0/31 | 281/286 → 286/286 |
| scriggo-method-declarations | 0 → 0 | 164 → 135 | 165 → 135 | 11,758,625 → 7,188,694 | 27,932 → 31,076 | 14,398,129 → 0 | 0/48 → 0/48 | 795/1,049 → 1,049/1,049 |
| skrub-duration-encoding | 0 → 0 | 77 → 47 | 77 → 47 | 3,447,453 → 1,780,726 | 28,450 → 28,985 | 0 → 0 | 0/130 → 0/130 | 2,784/2,784 → 2,784/2,784 |
| sql-formatter-bigquery-pipe-formatting | 0 → 0 | 76 → 82 | 77 → 83 | 3,550,251 → 3,565,280 | 22,431 → 17,900 | 0 → 0 | 0/26 → 0/26 | 5,709/5,709 → 5,709/5,709 |
| sqlfmt-create-table-ddl-formatting | 0 → 0 | 36 → 20 | 37 → 21 | 1,122,691 → 383,139 | 37,431 → 23,091 | 0 → 0 | 0/32 → 0/32 | 1,273/1,273 → 1,273/1,273 |
| sqlite-utils-safe-import-checkpoints | 0 → 0 | 119 → 79 | 119 → 79 | 4,860,146 → 2,802,784 | 53,463 → 31,431 | 0 → 0 | 0/60 → 0/60 | 1,038/1,038 → 1,038/1,038 |
| superjson-error-stack-serialization | 0 → 0 | 45 → 24 | 45 → 24 | 1,481,782 → 535,208 | 29,086 → 19,904 | 25,665 → 0 | 45/80 → 0/80 | 102/116 → 116/116 |
| task-task-graph-export | 0 → 0 | 103 → 99 | 104 → 100 | 3,979,739 → 6,934,732 | 30,690 → 53,130 | 0 → 0 | 0/20 → 0/20 | 17/17 → 17/17 |
| tengo-callable-instance-isolation | 0 → 0 | 176 → 62 | 176 → 62 | 11,596,057 → 2,686,997 | 55,677 → 23,410 | 8,179 → 6,201 | 0/23 → 0/23 | 0/122 → 122/122 |
| tengo-destructuring-bindings | 0 → 0 | 200 → 65 | 200 → 65 | 13,982,748 → 2,238,061 | 74,924 → 36,007 | 4,345,110 → 0 | 0/91 → 0/91 | 0/132 → 132/132 |
| termenv-preserve-ansi-resets | 0 → 0 | 50 → 32 | 50 → 32 | 1,303,076 → 658,852 | 25,653 → 26,497 | 13,781 → 16,660 | 21/35 → 15/35 | 87/87 → 0/87 |
| testem-bail-on-test-failure | 0 → 0 | 49 → 64 | 49 → 64 | 1,790,093 → 3,339,571 | 19,490 → 31,983 | 0 → 33,726 | 0/90 → 0/90 | 489/489 → 0/489 |
| testem-per-launcher-reports | 0 → 0 | 38 → 33 | 38 → 33 | 968,987 → 826,534 | 19,900 → 20,629 | 17,530 → 0 | 47/65 → 0/65 | 469/469 → 469/469 |
| textual-kitty-key-phases | 0 → 0 | 59 → 55 | 60 → 55 | 2,627,947 → 2,457,180 | 33,868 → 35,462 | 0 → 0 | 0/23 → 0/23 | 57/57 → 57/57 |
| textual-richlog-follow-state | 0 → 0 | 34 → 28 | 34 → 28 | 643,367 → 651,539 | 23,142 → 22,987 | 0 → 20,599 | 0/20 → 6/20 | 6/6 → 4/6 |
| tomlkit-toml-table-converters | 0 → 0 | 97 → 46 | 97 → 46 | 3,473,658 → 1,354,216 | 44,574 → 25,176 | 9,148 → 7,544 | 41/60 → 46/60 | 964/964 → 964/964 |
| true-myth-iterable-collection-combinators | 0 → 1 | 54 → 48 | 54 → 49 | 3,599,296 → 2,616,490 | 61,531 → 21,041 | 293,712 → 19,850 | 17/96 → 96/96 | 115/561 → 561/561 |
| ts-pattern-match-each | 0 → 0 | 28 → 30 | 28 → 30 | 809,261 → 763,918 | 18,121 → 18,584 | 10,022 → 0 | 0/85 → 0/85 | 6/6 → 6/6 |
| updo-policy-alerting | 0 → 0 | 38 → 22 | 38 → 22 | 1,239,133 → 449,712 | 28,562 → 22,542 | 0 → 0 | 0/17 → 0/17 | 123/123 → 123/123 |
| valibot-recursive-schema-composition | 0 → 0 | 69 → 60 | 69 → 60 | 4,612,461 → 2,046,426 | 26,873 → 13,367 | 0 → 10,785 | 0/10 → 0/10 | 209/209 → 168/209 |
| vitest-duration-sharding | 0 → 0 | 40 → 49 | 40 → 49 | 1,092,075 → 982,600 | 16,561 → 14,928 | 0 → 0 | 0/56 → 0/56 | 24/24 → 24/24 |
| vulture-persistent-analysis-cache | 0 → 0 | 86 → 41 | 86 → 41 | 3,789,741 → 956,602 | 67,471 → 23,056 | 15,906 → 0 | 13/24 → 0/24 | 277/295 → 295/295 |
| wasmi-trap-coredumps | 0 → 0 | 101 → 95 | 101 → 95 | 9,319,891 → 6,167,641 | 89,684 → 38,448 | 0 → 27,527 | 0/22 → 0/22 | 58/58 → 57/58 |
| wazero-multi-module-snapshots | 0 → 0 | 49 → 40 | 49 → 40 | 1,327,710 → 766,354 | 28,550 → 22,916 | 0 → 0 | 0/78 → 0/78 | 2/2 → 2/2 |
| yaegi-go-embed-directives | 0 → 0 | 100 → 152 | 101 → 152 | 4,800,210 → 6,522,390 | 22,450 → 27,179 | 0 → 0 | 0/38 → 0/38 | 58/58 → 58/58 |
| yjs-map-conflict-detection | 0 → 0 | 92 → 136 | 93 → 136 | 5,578,801 → 6,477,652 | 33,657 → 29,408 | 0 → 0 | 0/9 → 0/9 | 231/231 → 231/231 |
| ytt-jsonpath-query-api | 0 → 0 | 46 → 61 | 47 → 61 | 1,543,882 → 3,694,298 | 49,581 → 79,789 | 0 → 0 | 0/103 → 0/103 | 1/1 → 1/1 |
