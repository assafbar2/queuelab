# Development pilot notes

For the subsequent full local runs of the scripted reference, Falcon, and SmolLM2, see the [November 20 run record](local-run-2026-11-20/README.md). These notes preserve the earlier checks.

The scripted reference completed all 21 draft worlds successfully (all ten openings), with no timeouts or unsupported recommendations. This verifies the fixture and grader wiring, not model capability. The saved trace is [`docs/demo/pilot.json`](../docs/demo/pilot.json); `python3 -m harness rescore docs/demo/pilot.json` verifies it without a model.

## First Falcon CPU check

The pinned Falcon checkpoint loaded and generated locally on arm64 CPU with float32. The longer run was stopped during the fifth world because generation was slow and the first four worlds had timed out. No full model scorecard was produced. In a separate first-action diagnostic, the unchanged model emitted the input state instead of an action:

```json
{"opening": {"customer_message": "Our nightly export stopped arriving after the upgrade.", "account": "fictional-export-account", "clock": "2026-11-19T10:00:00Z"}, "history": []}
```

This establishes a formatting failure for the pilot's plain-JSON prompt, not an inability to investigate. The pinned checkpoint's chat template has a distinct tool-calling mode activated by supplying tool schemas.

## One native tool call

The CPU run of `probe-falcon` supplied the `inspect` schema and explicitly asked for `runs`. Falcon produced this single reply:

```text
<tool_call>
{"name": "inspect", "arguments":{"source": "export"}}
</tool_call>
```

The call has a recognizable structure, but `export` isn't an allowed source. The check reported `matched_expected_call: false` and did not execute anything. This is one diagnostic, not an investigation score. Try a clearer description of the available records before connecting the full loop. The [pilot guide](../docs/PILOT.md#from-one-call-to-a-baseline) lists the checks before training.
