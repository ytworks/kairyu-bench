# Disjoint-purpose observation composition

The CPU-only [composer](../scripts/experiments/gemma4-purpose/compose_purpose_collections.py) combines already completed observations for disjoint purposes into the existing native/matrix coupled-objective input format. It never loads a model, computes a gradient, solves a proposal, or writes a checkpoint.

The intended use retains the accepted parent's old152-case,42-fold observations and adds only the newly observed8 helper/composition cases,4folds. The resulting full160-case,46-fold view has explicit composed provenance. It is not represented as a single160-case model run.

## Input and output

A schema1 plan with kind disjoint_purpose_collections_v1 contains checkpoint (path,tree_sha256,attestation), parent_state and cases (path/SHA references), baseline_tolerance, and ordered segments. Each segment contains its cases reference and receipt reference. Receipts use the existing completed output-hash schema: completion reference, source_parent, all_stat_stable, and outputs keyed native-correct-logprob, native-margin, matrix-correct-logprob and matrix-margin; each output has path, files and stat.

The unchanged proposer validates every complete input segment. Composition then requires identical checkpoint/attestation, numerical collector settings, source implementation pins, tensor/sample layout and exact source-array bytes, including signed zero. Purpose groups and case IDs must be disjoint, and the combined cases must preserve the exact ordered source cohort.

Each gradient file is copied byte-for-byte to a direct output artifact; only its filename is prefixed to prevent collisions. No averaging, scaling or numerical reserialization occurs. Shared source-native/sample arrays are compared by dtype/shape/raw bytes and copied unchanged. Source receipts remain pinned. Completion publishes only after input continuity and unchanged adapter metadata validation pass.

The output has four collection directories, composition.json and an adapter-input-template.json. The latter has zero placeholder bounds solely to validate ingestion; it is not a solved or selected candidate. Replace the candidate policy through the existing reviewed builder before optimization.

    python compose_purpose_collections.py --plan PLAN.json --plan-sha256 SHA --output FRESH_DIR
    python compose_purpose_collections.py --plan PLAN.json --plan-sha256 SHA --output FRESH_DIR --write

Default preview validates existing metadata/source arrays without publishing or reading gradient payloads. --write copies the receipt-pinned gradient bytes. Fresh output must be disjoint from all source inputs; failed unpublished output is cleaned up only if created by this invocation.

## Explicit optimization cohort selection

The optional plan field purpose_selection contains selected_purposes (unique valid names in original order) and omitted_reasons (a nonempty reason for every omitted purpose). No omission is implicit.

A selected view writes its own cases.json and the exact kept fold/frame/gradient data, while retaining full-cohort and all segment input pins. The full160 view can be published separately with selection omitted. Selection changes the optimization cohort, not the original observations or any formal benchmark. No artificial large negative bounds or solver changes are needed.

For the next planned112-case view, six legacy groups may be explicitly omitted because their older answer units or intended role do not match the selected optimization task. The frozen prospective policy supplies the exact names/reasons; this tool contains no experiment-specific exclusion list.

One existing native/matrix end-to-end test is extended: disjoint composition passes through the unchanged proposer to a verified tiny saved checkpoint, original/kept gradient bytes are exact, protected checkpoint bytes remain unchanged, and overlap or signed-zero source mismatch rejects before publication. No test method or model run is added.

## Completed AD composition

The actual CPU operation completed the full 160 cases / 23 purposes / 46 folds and explicit selected 112 cases / 17 purposes / 34 folds in 5.55 seconds, with zero model or solver calls. The selected view omits tools, instruction, general, reasoning, lookup and japanese under the frozen answer-unit policy, while preserving the full view and original segment receipts. All four output directories pass unchanged adapter metadata validation.

Evidence: A/current-ad-composed-purpose-v1/full160/composition.json SHA256 733c007bfcb58a8e798a86550292188096e771d9744401d462ea67cd922e5a30; selected112/composition.json 3a98e5171963aa5b81e3734e0091fb2bd799099bc2523a186ec4bb3306009a12. The initial new-segment receipt omitted eight matrix NPY files; the preview rejected it before writing. A separate corrected source receipt72def917… preserves the original and pins all40 output files. No collector rerun was needed.
