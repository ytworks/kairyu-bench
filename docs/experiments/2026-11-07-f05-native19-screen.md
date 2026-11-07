# F05 candidate-only native screen

F05 failed the pre-fixed screen: 16/19 complete numerical passes, 3 failures, 0 unknown, 264/304 fixtures. F04 has exactly the same case/fixture results. All 16 previously passing guards stayed passing, but none of A1/F2/J1 became fully correct. The [machine gate](records/2026-11-07-f05-native19-screen-v1/score-v1/gate.json) therefore says REJECT_AT_NATIVE_SCREEN_OFFICIALLY_UNSCORED. F04 remains the parent. This is a known calibration screen, not a WSP score, independent transfer result, or general coding improvement; F2/J1 repeat one prompt.

Root fixed [the policy](records/2026-11-07-f05-native19-screen-v1/fixed-policy.json) before any candidate forward. Go required at least one target full pass, at most one regression among 16 guards, at least 16 full passes and 264 fixtures, complete 19-row generation/execution with zero unknown/API error/truncation. The target requirement alone failed. No threshold was changed, no candidate/model generation was repeated, and no parent was remeasured.

Generation completed all 19 once in 25.839991 s owner time, no error/empty/truncation/retry, default native thinking and the same settings as F04. Exact completed response SHA is 17d808d34f7fd7867fadaee6acfae2579d11dc1c5b0d75025d8cadb71a296ad3. It was scored only after the owner completed and released its lock.

The F04 scoring one-off was reused with exactly two replacements: the F05 output root and the final response SHA. The six shared input/case/scoring source hashes match F04. All 19 cases ran once, 16 fixed fixtures each; no reference/foil validation was rerun. Score time was 15.239229 s / 19 bounded CPU Docker calls / zero model calls. The existing image/daemon/worker and CPU1, 512 MiB, 15 s, user1003, network none, no GPU/no host mount/cap-drop restrictions were retained. Expected results stayed on the trusted host; generated code executed only in the existing worker. Raw numeric worker outputs, all public code, usage, request IDs, timings and failures are preserved.

| Case | F04 fixtures | F05 fixtures | F05 result |
|---|---:|---:|---|
| behavior-A1 | 0/16 | 0/16 | FAIL |
| behavior-A2 | 16/16 | 16/16 | PASS |
| behavior-B1 | 16/16 | 16/16 | PASS |
| behavior-B2 | 16/16 | 16/16 | PASS |
| behavior-C1 | 16/16 | 16/16 | PASS |
| behavior-C2 | 16/16 | 16/16 | PASS |
| behavior-D1 | 16/16 | 16/16 | PASS |
| behavior-D2 | 16/16 | 16/16 | PASS |
| behavior-E1 | 16/16 | 16/16 | PASS |
| behavior-E2 | 16/16 | 16/16 | PASS |
| behavior-F1 | 16/16 | 16/16 | PASS |
| behavior-F2 | 4/16 | 4/16 | FAIL |
| behavior-G1 | 16/16 | 16/16 | PASS |
| behavior-G2 | 16/16 | 16/16 | PASS |
| behavior-H1 | 16/16 | 16/16 | PASS |
| behavior-H2 | 16/16 | 16/16 | PASS |
| behavior-I1 | 16/16 | 16/16 | PASS |
| behavior-I2 | 16/16 | 16/16 | PASS |
| behavior-J1 | 4/16 | 4/16 | FAIL |

The scorer cannot prove B1/B2's symbolic-library method requirement; numerical passing is not that proof. A1 remains 0/16; F2/J1 remain 4/16. The failing targets and all other cases were retained without cherry-picking.

One metadata preparation attempt stopped before creating the score driver or launching Docker: a hash read while generation was still writing differed from the final file. The final owner completion records the authoritative response hash, which matched before scoring. [The failed assertion is preserved](records/2026-11-07-f05-native19-screen-v1/score-preparation-failure-v1.json); this was zero scoring/model calls and did not cause a generated answer or fixture rerun.

After scoring, owned sandbox containers were zero; the exact 13 running-container set matched before/after, and all Docker containers numbered 27. All scoring work completed before notification to root. No model-service operation occurred. [The complete records](records/2026-11-07-f05-native19-screen-v1/SHA256SUMS) include plan/source pins, final generation closure, raw worker report, summary, gate, exact one-off sources, and lossless packaging provenance. Report SHA e717f268cf3c129d288878097e6e791e7834d1685367323db82be26070619af4; gate SHA 77548bc4f71fcb74afafc3e964fd8a39293255e649e02373001a5ffdec3c1d2f.

The proposal, 16-round history, saved weights and actual Opus 5.5 review are retained in [the preceding record](2026-11-07-f05-proposal-save-opus.md). The full 2338-item WSP was conditional on this screen passing and was not started for this rejected candidate. No official capability score is assigned to F05.
