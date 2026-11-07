# Public-answer generator gateway compatibility, 2026-11-07

The original authored120 trial produced 120 HTTP errors and no public answers in
2.1462 seconds, with zero retries. Its original inputs, outputs, owner and
generator bytes remain unchanged. Every correctness result remains unknown.
The generator's receipt `completed` only means all attempts ended.

Read-only inspection of the actual gateway image showed that
`chat_template_kwargs` is accepted by the request schema, but its text
`_upstream_conversation` path rejects a nonempty value when no Kairyu text
template is installed. The existing generator sent that field. Exactly 120
HTTP 400 access records corroborate the request failure; the discarded response
body cannot be recovered and is not inferred. Successful WSP calls omit the
field. See the [original failure and actual-image evidence](2026-11-07-av-authored120-native-generation.md).

The minimal request change is explicit `--thinking-control server-default`.
It omits only `chat_template_kwargs`; model, user prompt, seed and max_tokens
remain unchanged, and temperature/top_p remain omitted. The default
`--thinking-control request` preserves the previous direct-vLLM request
body, including `{"enable_thinking": true}`. No gateway, model, sampling,
benchmark or checkpoint configuration is changed.

The opt-in provenance records `thinking_control=server-default`,
`enable_thinking=null` and `frame=live-server-default-public-answer`.
The generator does not infer the server's thinking state. For this deployment,
separate evidence shows all four actual AV replica commands contain
`--default-chat-template-kwargs {"enable_thinking":true}` and
`--reasoning-parser gemma4`. The checkpoint Jinja by itself defaults thinking
to false, so that template alone cannot establish the served state. The
immutable server identity, effective flags, template and AV mount proof must
remain bound by the trial owner. This configuration evidence does not assert
identical generated tokens or recover the old forced-empty-thought proxy frame.

HTTPError rows now retain only `error_type` and an integer `http_status`
in the range 400–599, alongside the case ID. Error body, message, URL, headers
and reasoning are not saved. The existing sandbox accepts this numeric field
only with HTTPError and no public answer, rejects strings, booleans and invalid
codes, and records `generation_error` / `passed=null`. It also accepts
and checks the new thinking-control provenance. Its fixed-catalog scope is
unchanged; this is not a new executor for arbitrary 120-case input.

The generator's five CPU tests passed in 0.017 s; the sandbox's seven CPU tests
passed in 0.583 s. They protect the exact old request body, the new request and
honest provenance, a real Python HTTPError with status 400 and no retry/private
content, and the resulting unknown status without answer execution. Existing
preview, output isolation and execution-contract fixtures also pass. Focused
Ruff and `git diff --check` passed. These are mocked transport / CPU checks:
no new model request, serving operation, GPU job or authored trial was run by
this change. A new trial requires a separately reviewed fresh plan and retains
the failed first trial.

[Compact validation and source hashes](artifacts/science-generator-gateway-fix-20261107.json)
record all four changed Python files, exact CPU commands and actual-image
evidence hashes. The usage guide and tool inventory are updated. The generator
source is `df88767ebcd5a3138747a36fe7e0976dbb0c2ceb5b0f1eece74740818f0b7f89`;
the sandbox source is
`cf26e77e90d5789422ad2ebd3ac43599ee0a2ebc781b7a1c0c8ae834953a817e`.
