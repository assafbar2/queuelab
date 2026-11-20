# Run the post-training experiment

This is a real local LoRA run, separate from the scripted demo and the earlier unchanged-model pilot. It trains adapters for the pinned Falcon 90M and SmolLM2 1.7B checkpoints. It does not train either base model from scratch or touch a customer account.

## Prepare

Use Python 3.14 on an Apple Silicon Mac with Metal available. Set up a separate training environment so the existing pilot's PyTorch installation is unchanged:

```sh
python3 -m venv .venv-mlx
.venv-mlx/bin/pip install -r reports/post-training-2026-11-20/environment.txt
.venv-mlx/bin/python -m training prepare --output /tmp/queuelab-cases
```

The data generator saves a split manifest with checksums. It creates 200 scripted, evidence-checked training investigations (610 next-action examples), 100 validation investigations in separate groups, and 60 holdout investigations across 30 paired openings. The holdout isn't read by the trainer. These are fiction and haven't been independently reviewed. The holdout changes the tickets and record wording, but covers the same failure families. It is not a production test or the full independent benchmark proposed in the spec.

## Train, reload, and compare

```sh
.venv-mlx/bin/python -m training fit --backend smol --data /tmp/queuelab-cases \
  --steps 180 --batch 4 --rate 0.0002 --output /tmp/queuelab-smol-adapter
.venv-mlx/bin/python -m training fit --backend falcon --data /tmp/queuelab-cases \
  --steps 180 --batch 4 --rate 0.0003 --output /tmp/queuelab-falcon-adapter

.venv-mlx/bin/python -m training compare --backend smol --data /tmp/queuelab-cases \
  --adapter /tmp/queuelab-smol-adapter --output /tmp/queuelab-smol-eval
.venv-mlx/bin/python -m training compare --backend falcon --data /tmp/queuelab-cases \
  --adapter /tmp/queuelab-falcon-adapter --output /tmp/queuelab-falcon-eval
```

The first use of each model downloads its pinned checkpoint if it isn't cached. Set `HF_HUB_OFFLINE=1` to use cached models only. Use fresh output directories; the program won't overwrite a previous run. No model or result is uploaded by these commands.

Only LoRA matrices on the attention query/value projections are trainable. The loss covers the expert's next JSON action and its end-of-turn token. Customer text, tool records, prior actions and padding are excluded. This run uses a fixed final step chosen before evaluation. A nonzero finite adapter and its checksum are checked after training; the comparison reloads a fresh copy with and without the saved adapter.

The two arms of each comparison see the same prompt, cases, tools, six-action limit and greedy generation settings. These settings are not identical to the earlier PyTorch pilot, so compare only the two arms of each new run. The scorer records the paired change, a 95% interval resampled by opening, and the change in unsupported recommendations. A higher score does not count as an unqualified win if unsupported claims increase.

## Verify without model weights

```sh
python3 -m training audit --data /tmp/queuelab-cases --results /tmp/queuelab-smol-eval
python3 -m training audit --data /tmp/queuelab-cases --results /tmp/queuelab-falcon-eval
```

The audit rebuilds each episode from the saved raw model responses and checks the paired scorecard. It needs no adapter, GPU, or network access. The run bundle includes `manifest.json` and `holdout-cases.json`, so `--data` can also point to the bundle instead of the generated dataset.

Once both model reports are complete, rebuild the assets straight from the audited traces:

```sh
run=reports/post-training-2026-11-20
python3 -m training.replay --run "$run" --output /tmp/queuelab-replay.html
python3 -m training.scorecard --run "$run" --output /tmp/queuelab-scorecard.svg
```

The replay is self-contained. Open the HTML file in a browser, switch models and cases, and unfold each action to inspect the raw model reply and the simulator's observation.

The audited [public replay](https://assafbar2.github.io/queuelab/post-training/) is served from `docs/post-training/index.html`. With the local server from the [pilot guide](PILOT.md#watch-the-local-replay) running, open `/post-training/` to view a local copy.
