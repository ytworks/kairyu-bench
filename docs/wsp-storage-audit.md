# WSP capacity reconciliation

Run on the designated SSH host as `y-takagi`. This task-owned standard-library
tool inventories files and dedicated Docker daemons. It does not delete files,
stop processes, execute models, or perform learning.

```bash
python3 scripts/wsp-storage-audit.py --help
python3 scripts/wsp-storage-audit.py --trial-id YOUR_REGISTERED_TRIAL
python3 scripts/wsp-storage-audit.py --trial-id YOUR_REGISTERED_TRIAL --update-ledger
```

The optional write updates the work root's `storage-state.json` atomically.
Normal updates preserve adopted and candidate pointers. Only the explicit
`--restart-from-original` flag clears them; use it solely for an authorized
fresh start. Historical pointers never restore checkpoints.

Regular files are counted once by device/inode across inspected paths.
Symlinks are recorded and not followed. Read errors remain unknown and block
creation; they are never interpreted as zero. The conservative weight bound
counts all readable work-root data, including JSON specs, auxiliary matrices,
evidence and caches. Tensor suffix counts alone are not the weight budget.
The original checkpoint is separate and permanently retained.

Docker `LayersSize` and unshared build-record sizes are logical accounting.
Shared build-record sizes are excluded from that total; the result is not
reported as filesystem allocation. Host paths that cannot be inspected still
need reconciliation. Other services are excluded explicitly and never pruned.

The tool records mount references but cannot prove absence of every process
reference. It deliberately never authorizes a new checkpoint on its own.
Before creation, verify complete accounting, writer/evaluator state, peak
temporary capacity, the 150 GiB weight and 100 GiB cache limits, and at least
200 GiB free NVMe throughout the operation. Keep unique evidence until its
preservation is verified. No cleanup operation is provided by this script.
