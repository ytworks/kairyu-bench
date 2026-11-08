# Behavior12 public symbolic-method review

The AV native response file has SHA `5ea0e169922048a2e5bf649ceb6b1add7bb81db5ee19bcf48c620cb00f3de3db`.
The two public B answers were read without executing generated code. B1 calls
SymPy diff and lambdify, then broadcasts the evaluated derivative to x.shape.
B2 differentiates while retaining symbolic parameter sa, lambdifies both
partials, and passes numerical a only during array evaluation; the values feed
the returned columns. Both calls are on the direct public return path, not
merely mentioned in unused strings. The empty-input branch returns shape(0,2).

This is public source/data-flow inspection, not an instrumented runtime call
trace or proof of the entire input contract. The original automated
`method_requirement_checked=false` remains unchanged. Numeric fixture outcomes
and every failed trial are recorded by the separate isolated executor.

[Answer identity and exact inspection findings](records/2026-11-07-behavior12-public-method-review/review.json).
No new generation, execution or service operation was performed.
