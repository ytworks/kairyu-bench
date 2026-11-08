# F06 generic deployment preparation — 2026-11-07

All four existing guarded deployment previews passed in3.452 seconds. The frozen parent remained F04 and all27 containers/23 non-target services were unchanged. No model call, payload rehash, stop, restart or deployment was performed during preparation.

The candidate is the completed dense LoRA merge, tree `0fc0ef29346c4d8a06bea8570dee96d8e0ba7038fee0d5aded8db57852474ec5`. The generic `redeploy-replica.py --freeze` contract receives all9 manifest file hashes and saved full POSIX stats; the only schema conversion is `dev` to `device`. No native-faststore handler is used. The existing parallel owner and generic helper are unchanged.

Existing create/verify supplied the full hashes and exact regenerated BF16 recipe checks. The compact identity was captured after verification: all mtime/ctime values were no later than the verify output time. The verifier checks its own before/after stats, but that internal snapshot was not separately saved. The new freeze retains this limitation; it does not claim a saved verification-start snapshot or new full payload hashing.

Under the global nonblocking lock, preparation checked four exact healthy idle F04 CIDs, captured fresh27 membership and current gemma2 StartedAt, ran the four previews using one common baseline, then rechecked source pins and complete candidate stat equality. The lock was normally released. The fully bound plan remains unapproved until root provides exact-hash authorization.

| Artifact | SHA256 |
| --- | --- |
| dense manifest | `22b0603cc2dc9a2060c76f8baf99c28eceac08b56116727173bc61a719b5c0d4` |
| full verify output | `42140a38d3f2412887952c0dfe836433f0fabb6a8aa68196978eb8e5e479788d` |
| compact identity | `50aa8056b9e48257b48c8621396030b4796c2e72d1c94fd8f64ad84a6867bdf7` |
| generic freeze | `cee54e1c7f79cf2116ea7b6036c9d2d3c0709f83117cb95c270542b616b84221` |
| bound unapproved plan | `361d57aef4c6a77495f0dd1220259e1ddfd075c4a890e84423ca137c291a8643` |

The [recorded one-time binding source and previews](records/2026-11-07-f06-generic-deployment-preparation/bind-generic-previews.py.txt) preserve all commands and receipts. Actual execution uses the unchanged parallel owner with a separately approved plan; all four children are joined and any failure invokes the existing full four-replica rollback before another benchmark may start.
