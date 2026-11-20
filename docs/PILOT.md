# QueueLab pilot guide

The quickest check uses a saved scripted run. Try that before downloading Falcon. All commands here run from the QueueLab folder.

## Watch the local replay

With Python 3.10–3.14 installed, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Leave the terminal open and visit [the replay on your computer](http://127.0.0.1:8000/demo/). On the first ticket, select **Scenario 1**, step through the actions, and reveal the verdict: the schedule was disabled. Now select **Scenario 2**. The complaint is identical, but the records point to the wrong delivery folder. **Scenario 3** should end in escalation because the records are missing. Press Ctrl+C in the terminal to stop the local web server.

There’s no public replay URL yet. The repository owner can publish it in GitHub under **Settings → Pages → Deploy from a branch → `main` / `docs`**, then share the site’s `/demo/` path.

## Verify the grader

```sh
python3 -m harness rescore docs/demo/pilot.json
```

The bundled score should show 21 attempted and 21 supported decisions across all 10 tickets, with zero timeouts and unsupported claims. The command rebuilds the observations and scores from the saved actions. If a saved observation, verdict, or total differs, the check fails. This verifies the recorded run and the scoring path. It says nothing about whether Falcon gives good answers.

To run the scripted checklist yourself and check the fresh file:

```sh
python3 -m harness run --backend rules --output reports/reference.json
python3 -m harness rescore reports/reference.json
```

Expect the same totals. You can open that file in the replay with **Open another report**. The runner won’t overwrite a report, so use a new filename when you repeat it.

## Check one Falcon action

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv run --locked --extra model python -m harness probe-falcon
```

The first run downloads the Python libraries and the exact Falcon version used by the project. This check gives Falcon the export ticket, describes the `inspect` action and its accepted inputs in a machine-readable format, and explicitly asks it to check the run history. You’ll see the model’s actual reply in `raw`. The command does not execute the requested action or save a score.

Look for `"matched_expected_call": true`. It means Falcon returned one request to `inspect` with `source` set to `runs`, wrapped in Falcon’s expected `<tool_call>` tags. If the result is `false`, read `raw` to see what it did instead. A pass only means it could follow that explicit instruction.

On the local CPU check, Falcon requested `inspect` but supplied `export`, which isn’t an allowed source. The check returned `false`. No record was read. The Colab notebook includes the same diagnostic if you prefer to run it in the cloud.

### From one call to a baseline

1. **Get a valid first request.** Try clearer descriptions of the allowed record names on the development cases. Keep the failed outputs so we can see what changed. If Falcon still can’t make a valid request after a few attempts, try the fallback model rather than quietly replacing bad arguments with good ones.
2. **Connect the full investigation loop.** Give Falcon the descriptions for all four actions (`inspect`, `ask`, `recommend`, `escalate`). Stop at the end of its turn, convert each well-formed call to QueueLab’s existing format, and keep the original reply in the trace. The simulator reads the fake records and returns the evidence. Unrecognized or malformed calls still use up a turn.
3. **Start with the three export scenarios.** Give Falcon the complaint without telling it what to inspect. Check whether it uses the records it gets back, whether a different cause changes its decision, and whether it escalates when records are missing. A well-formatted request by itself is not enough.
4. **Run all 21 development scenarios with the unchanged model.** Save every trace, including failed attempts. Review errors before training: was the request malformed, was the wrong record checked, or did the model make a claim the evidence didn’t support? Train only after the cases are reviewed and there is a recurring mistake worth teaching the model to avoid.

The one-action check exists today. The full native-format loop in step 2 does not. The current runner below still uses the original plain-JSON prompt, and running it does not complete the native-format baseline.

## Rerun the current model pilot

For a local run, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and start with one ticket (three scenarios). On a laptop this can take several minutes. A report can be valid even if the model fails every scenario; `rescore` checks what happened, not whether it went well.

```sh
uv run --locked --extra model python -m harness run --backend falcon --limit-openings 1 --output reports/falcon-smoke.json
python3 -m harness rescore reports/falcon-smoke.json
```

Remove `--limit-openings 1` to try all 21 scenarios. Add `--offline` if you already downloaded the model and want to use cached files. `--backend smol` selects the larger fallback model described in the [spec](SPEC.md#post-training-recipe).

For a cloud run, use the [Colab notebook](https://colab.research.google.com/github/assafbar2/queuelab/blob/main/notebooks/pilot.ipynb) once it is published on `main`. It runs the same command on a remote computer. Start with the default one-ticket check. Set `OPENINGS = 10` for the full dataset, or choose a GPU runtime and set `DEVICE = 'cuda'` if one is available. The notebook downloads the resulting report to your machine; it does not retrain the model.
