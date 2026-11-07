# Dependency proposal postcheck

After the existing coupled proposer finishes, scripts/experiments/gemma4-purpose/summarize_dependency_set_proposals.py reconstructs each emitted BF16 edit from the recorded native values and matrix sample coordinates.

    python scripts/experiments/gemma4-purpose/summarize_dependency_set_proposals.py \
      --roster /absolute/path/plan-roster.json \
      --observation-root /absolute/path/four-completed-collections \
      --output /absolute/path/fresh-numeric-postcheck.json

The tool reports exact selected-coordinate counts, family support, raw L2, unscaled relative L2, and the declared metric's L2. The fixed eight-set policy uses a declared weighted cap of 1 and a continuous proposal box of max(.025 * abs(base), 1e-6). Nearest BF16 values can exceed that continuous box; their excess is reported without clipping or rescaling. The mixed native/matrix alternative has a different normalized metric, so its cap cannot be compared directly with unscaled relative norms. A block with zero realized changes is reported as a collapse.

The 84 predictions and shortfalls come from the pinned existing solver; this postcheck does not recompute gradients or run a model. It verifies the report's plan/cost/constraints, source-array hashes when included in the report, and input hash/stat continuity. Output must be fresh and outside source observations and the checkpoint. A missing proposal remains missing. Passing the cap permits a finite diagnostic; it does not establish an actual response improvement or authorize deployment.

The executed host-specific analysis script and its original result remain historical artifacts. This portable version changes path selection and input/output checks, not the numeric formulas. See [plan preparation](wsp-dependency-set-plans.md).
