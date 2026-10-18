# Probe fixtures

`loadmaster probe` runs every fixture in `crates/loadmaster-probe/fixtures/` (also at
`fixtures/probe/`), each three times at temperature 0 and seed 42; a fixture passes only if
all three runs pass. Adding a fixture is adding a YAML file there (or pass `--fixtures DIR`
for your own); no Rust changes.

**Verdict:** `pass` if no fixture fails, `partial` if every fixture marked `core: true` passes
(tool.single, tool.result_roundtrip, schema.strict), otherwise `fail`. Exit code 0 / 1 / 2.
Skipped fixtures (e.g. the template check with nothing to compare against) do not count.

## Format

```yaml
id: tool.single                      # unique, shown in reports
description: One tool, a prompt that needs it.
core: true                           # counts toward `partial`
repeat: 3                            # optional
steps:                               # requests, in order
  - stream: false                    # optional
    body:                            # merged over {model, temperature: 0, seed: 42, max_tokens: 512}
      messages: [...]
      tools: [...]
  - body:
      messages:
        - { "$from": "0:$.choices[0].message" }   # a value from an earlier response
pass: <condition>                    # all of it must hold
fail:                                # ordered; the first that holds names the failure
  - class: call_in_content
    if: <condition>
  - class: no_call                   # no `if`: the catch-all
```

Responses are checked in non-streamed shape: streamed answers are reassembled first
(content and reasoning concatenated, tool calls merged by index).

### Conditions

| Condition | Holds when |
|---|---|
| `{ path: P, exists: true }` | the JSONPath matches a non-null value |
| `{ path: P, count: N }` / `count_min` / `count_max` | the match (an array's length, or number of matches) |
| `{ path: P, equals: V }` | the first match equals V |
| `{ path: P, contains_any: [..] }` | a matched string contains any of the needles |
| `{ path: P, json: true }` | every matched string is valid JSON (`false`: not) |
| `{ path: P, schema: S, parse: true }` | every match (parsed from a string with `parse`) validates against JSON Schema S |
| `{ status: 200 }` / `{ status_class: 4 }` | HTTP status |
| `{ stream_done: true, exchange: N }` | the stream ended with `[DONE]` |
| `{ same: [A, B] }` | two references resolve to equal values (JSON strings compared parsed) |
| `{ ran: N }` | step N ran (earlier steps produced what it needed) |
| `all: [..]`, `any: [..]`, `not: ..` | combinators |

Paths are JSONPath over the response; prefix `N:` for step N (default: the last step).

## Failure classes

| Class | Fixtures | Meaning |
|---|---|---|
| `4` | schema.enum_pattern, schema.strict, schema.union |  |
| `args_invalid_json` | tool.nested, tool.parallel, tool.single | `arguments` is not valid JSON. |
| `call_in_content` | tool.single | The tool call is written as text in `content`; `tool_calls` is empty. Usually a tool-call parser that does not match the chat template. |
| `call_in_reasoning` | reasoning.leak, tool.single | The tool call sits in `reasoning_content` or a `<think>` block. |
| `enum_ignored` | schema.enum_pattern | A value outside the `enum`. |
| `ignores_result` | tool.result_roundtrip | The final answer does not use the tool result. |
| `loops_on_tool` | tool.result_roundtrip | After the tool result, the model calls the tool again instead of answering. |
| `no_call` | stream.tool, tool.nested, tool.parallel, tool.result_roundtrip, tool.single | No tool call where one was required. |
| `parallel_in_content` | tool.parallel | Several calls written as text instead of `tool_calls`. |
| `parallel_partial` | tool.parallel | Some, but not all, of the parallel calls. |
| `parallel_serialised` | tool.parallel | Only one of several independent calls in the turn. |
| `pattern_ignored` | schema.enum_pattern | A string that does not match the `pattern`. |
| `schema_dropped_silent` | schema.enum_pattern, schema.strict | HTTP 200, but the output ignores the schema (not JSON, or invalid). |
| `schema_flattened` | tool.nested | Nested object or array arguments arrive as strings, or otherwise fail the schema. |
| `schema_rejected` | schema.enum_pattern, schema.strict, schema.union | `response_format` with a JSON schema is answered with a 4xx. |
| `stream_args_split_invalid` | stream.tool | Streamed argument fragments do not concatenate to valid JSON. |
| `stream_truncated` | stream.tool | The stream ends without `[DONE]` (and usually mid-call). |
| `union_unconstrained` | schema.union | Output matches neither `anyOf` branch. |
| `template_mismatch` | template.hash | The runtime renders with a different chat template than the model file's. |
| `unclassified` | any | No `fail` rule matched; look at the captured responses. |
| `unreachable` | any | The endpoint did not answer. |

Every class above is produced by the fake runtime in the probe's tests
(`crates/loadmaster-probe/tests/conformance.rs`).

## Debugging a failure

Raw requests and responses are saved under `~/.loadmaster/probe/<run>/<fixture>/`
(`runN-stepM.request.json`, `runN-stepM.response.raw`).
