from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkerPoolTest(unittest.TestCase):
    def test_refills_a_free_slot_before_the_slowest_peer_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.txt"
            events = root / "events.txt"
            queue = root / "queue"
            inputs.write_text("fast\nslow\nrefill\n", encoding="utf-8")

            script = f"""
set -eu
. {ROOT / 'scripts/lib/worker-pool.sh'}
events=$1
queue=$2
inputs=$3
run_item() {{
    item_index=$1
    item_name=$2
    printf 'start %s %s\\n' "$item_index" "$item_name" >>"$events"
    case "$item_name" in
        slow) sleep 0.5 ;;
        *) sleep 0.05 ;;
    esac
    printf 'end %s %s\\n' "$item_index" "$item_name" >>"$events"
}}
worker_pool_run 2 "$queue" "$inputs" run_item
"""
            subprocess.run(
                [
                    "/bin/sh",
                    "-c",
                    script,
                    "worker-pool-test",
                    events,
                    queue,
                    inputs,
                ],
                check=True,
            )

            order = events.read_text(encoding="utf-8").splitlines()
            self.assertLess(order.index("start 3 refill"), order.index("end 2 slow"))
            self.assertFalse(queue.exists())

    def test_propagates_worker_failure_and_removes_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.txt"
            queue = root / "queue"
            inputs.write_text("ok\nbroken\n", encoding="utf-8")

            script = f"""
set -eu
. {ROOT / 'scripts/lib/worker-pool.sh'}
queue=$1
inputs=$2
run_item() {{
    [ "$2" != broken ]
}}
worker_pool_run 2 "$queue" "$inputs" run_item
"""
            completed = subprocess.run(
                ["/bin/sh", "-c", script, "worker-pool-test", queue, inputs],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("one or more workers failed", completed.stderr)
            self.assertFalse(queue.exists())


if __name__ == "__main__":
    unittest.main()
