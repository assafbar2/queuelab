# QueueLab local run: November 20, 2026

This run uses the 21 synthetic development scenarios, not a held-out benchmark. Nothing was trained. The scripted reference and the two unchanged checkpoints are distinct ways to run the current program, not the before/after scores in the post-training animation.

## Run record

| Configuration | Evidence-backed decisions | Timeouts | Accepted record reads or questions | Records |
| --- | ---: | ---: | ---: | --- |
| Scripted reference | 21/21 | 0 | 44 | [Trace](scripted.json) · [Terminal log](scripted.log) |
| Falcon 90M, unchanged | 0/21 | 21 | 0 | [Trace](falcon.json) · [Terminal log](falcon.log) |
| SmolLM2 1.7B, unchanged | 0/21 | 21 | 86 | [Trace](smollm2.json) · [Terminal log](smollm2.log) |

All three runs finished and rescored exactly. Falcon's 126 replies were rejected by the action parser. Its separate native-call probe on [CPU](falcon-probe-cpu.json) and [GPU](falcon-probe.json) produced an `inspect` request for `export`, which isn't an allowed source. No tool was executed in either probe.

SmolLM2's first action was a valid run-history request in every scenario. It continued to reread that source and never submitted an accepted final action. Its other 40 turns were malformed `recommend` requests that used `fact` instead of `finding`; the grader flagged these as unsupported recommendation attempts. That flag is not a count of factual falsehoods.

## What was recorded

The [combined browser recording](all-backends-replay.mp4) shows the three alternate scenarios for the opening export ticket in each configuration. Individual recordings: [scripted](scripted-replay.mp4), [Falcon](falcon-replay.mp4), and [SmolLM2](smollm2-replay.mp4). These are recordings of the local replay, not live inference.

The JSON traces cover all 63 episodes and 317 actions, including rejected text. Browser verification checked every rendered turn and verdict against the saved reports. See the [scripted browser log](browser-scripted.log) and the [model browser log](browser-models.log).

## Configuration

The model runs called the existing Python runner on Apple Metal (`device='mps'`, float32, CPU operator fallback enabled) with downloaded weights cached locally. The prompt, action parser, six-action limit, generation cap and grader were unchanged. Each full report records its pinned revision, backend, versions, and fixture hash. The scripted log records SHA-256 fingerprints of the local runner, lockfile and fixture files.

From the repository root, `python3 -m harness rescore reports/local-run-2026-11-20/scripted.json` verifies the saved scripted run without model weights. Use the same command with any completed model report.
