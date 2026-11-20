# Models and cases behind the results

QueueLab is an open-source experiment in engineering support, not a support service. It tests whether a model learns to inspect the available evidence, choose a supported next step, and escalate when it can't establish a cause. No real customer accounts were connected or changed.

| Base checkpoint | Why it was included | Measured outcome |
| --- | --- | --- |
| [Hugging Face SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) | General instruction-following model | 0/60 → 53/60 evidence-backed decisions; unsupported recommendations fell from 36 to 7 cases. |
| [TII Falcon-H1-Tiny-Tool-Calling-90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M/tree/fc099d791a229464b995fdd2fd24e8ffd2d699e8) | Much smaller model trained to request tool actions | 0/60 → 3/60 evidence-backed decisions; unsupported recommendations rose from 0 to 39. This did not pass the gain rule. |

These are **within-model comparisons**. Each model faced the same test twice, with only its adapter changed. The comparison doesn't show why one architecture performed better or what model size a support team needs. An unsupported recommendation flag includes attempts the grader could not ground or accept; it isn't a count of false sentences.

## What the cases can tell us

Every case is fictional. We used 200 scripted training investigations, 100 validation investigations, and 60 held-out investigations across 30 pairs. The holdout contains different tickets and record wording from known failure families. A deterministic reference checked the example actions; support practitioners have not independently reviewed the cases. The [split manifest](../reports/post-training-2026-11-20/manifest.json) and [full run record](../reports/post-training-2026-11-20/README.md) show how the cases and scores break down. All seven remaining SmolLM2 failures were disabled-schedule cases. These results don't establish customer impact.

The paired post-training experiment ran on a Mac using MLX. The Colab notebook runs only the earlier pilot. The browser replay and the saved-score audit need no model. Anyone choosing to reproduce training is responsible for their own setup; see the [recorded procedure](TRAINING.md).

## Release terms

QueueLab’s original code, documentation, synthetic fixtures, and illustrations are available under the [MIT License](../LICENSE). Upstream checkpoints and their derivatives are not covered by QueueLab’s license. The pinned SmolLM2 checkpoint lists Apache-2.0; the pinned Falcon checkpoint lists the Falcon LLM License. The repository contains the trained adapters’ recipes, hashes, and traces, not downloadable base weights or trained adapter binaries.
