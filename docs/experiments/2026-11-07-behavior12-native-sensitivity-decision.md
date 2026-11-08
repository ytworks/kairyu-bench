# Native sensitivity measurement after observed behavior failures

The completed AV train12 generation and isolated execution show10 passing
and2 failing cases,164/192 fixture trials,0 unknown. A1 has0/16 passes from
batched RHS shape interpretation; F2 has4/16 and sums indices0,k,2k instead
of the required k-1,2k-1,... . F2 does not reproduce its authored zero-removal
foil. That foil remains frozen and its margin is not a direct measure of the
observed off-by-one repair. Generation used12 POST requests in18.620 seconds;
execution used12 isolated CPU workers in10.301 seconds. These are calibration
results, not official WSP scores. Transfer6 remains ungenerated.

Measure all12 singleton folds in the current cumulative parent AV, once each
for correct-answer mean log probability and correct-minus-foil margin. Collect
all421 text-native scalar/vector tensors and60 layers, without assigning a
coding role to large weights or gradients. The purpose is to inspect shared
and conflicting sensitivities between the two failures, their A2/F1 relation
controls and every other successful case. No failed/successful case is removed.
The exact12 case bytes, source, frame and AV9-file attestation are pinned.

Correct-answer LL is the primary candidate-direction evidence; foil margin
remains a separate contrast. A future joint edit must use actual BF16 differences,
per-tensor/family denominators and finite/native behavior tests. There is no
preselected tensor count, support mask, gain amplitude, candidate or adoption
in this read-only weight measurement. The forced-empty-thought teacher-forced
frame is not the usual native generation context and its gain is not ability.

The concrete owner plan v2 is8edba22f5aa38ec5eebf443a3b4b2eab2794b23c9968fcd0c09d13d003f20316.
Owner source9a8d9255312a97ae2762ec97c34c9701ac5d64dcb0dc273e01f8d7f6700b7183
has13 focused CPU guard tests and an independent review. The current full552
CPU suite predates this final race fix; it is not mislabeled as testing that
source. Only the exact idle gemma-2 CID may pause; GPU4 performs the observation,
other26 containers stay fixed, and the same original CID is restored after
owned collector inactivity is proven. No checkpoint bytes are written.

Two sequential collector subprocesses share a900-second limit including their
model loads; serving stop and restoration are separately timed. Runtime is the
existing af9 image, UID1003, network none, one GPU4,128GiB host memory,88GiB
Torch allocator limit, serial VJPs, SDPA text attention and CPU FP32 saved
activations. All18 complete frames were measured at at most290 tokens; the
collector limit is1024 with no truncation. Output/cache remain on NVMe.
Global/GPU4 locks, exact ownership, source continuity, partial failures and
restoration status remain in the execution receipts. This decision authorizes
one measurement under the user's research task, not a saved-weight edit.
