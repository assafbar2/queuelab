# QueueLab v1: train a miniature support investigator

**Current stage: local post-training experiment.** The repository includes the scripted pilot and a [local run](../reports/post-training-2026-11-20/README.md) with real LoRA adapters for Falcon and SmolLM2. The run holds out new openings from known failure families. The independent corpus review and broader sealed benchmark described below are still future work.

## The experiment

A customer says: *Our nightly export stopped arriving after the upgrade.* A convincing reply is easy to write. A useful investigator first establishes whether the export ran. QueueLab tests whether that habit can be taught to a small model by changing learned parameters, rather than by giving it a longer prompt.

The primary deliverable is a trained LoRA adapter for an existing small instruction model. The harness provides evidence for what the adapter learned and where it regressed. We are not pretraining a foundation model or claiming the entire parameter count was trained. The model card must report the actual total and trainable parameter counts separately.

**Readers:** a support engineer can inspect or contribute a case; an ML engineer can reproduce the comparison; a CX leader can watch two investigations of the same ticket without installing anything. The book supplies the distinction between a plausible answer and a supported outcome, not an answer key for the test cases.

## One ticket, three possible worlds

| What the simulator knows | What the investigator can learn | Defensible end state |
| --- | --- | --- |
| The schedule was disabled during the upgrade. | No run since the change; the schedule record says disabled. | Recommend checking and re-enabling the schedule. Do not claim the export is fixed. |
| The export ran, but the delivery configuration still points at the old destination. | A successful run, its delivery receipt, and the customer's stated expected destination disagree. | Identify the destination mismatch and recommend correcting it. Do not suggest rerunning the job blindly. |
| Run records expired; there is no delivery evidence or additional customer evidence. | The same complaint, but the diagnostic records are explicitly unavailable. | Escalate with the missing evidence and a useful next check. Do not invent a cause. |

These are fictional examples. The first two form a matched pair: identical opening words, different justified answers. The third tests whether the model knows when to stop. No one gets credit just for naming the hidden cause.

## Investigator and harness contract

1. Start both checkpoints with the same customer message, account scope, simulated clock, tool descriptions and action vocabulary. Never expose the world ID, cause, family, expected path, grading rules for that case, or future tool responses.
2. Each turn emits one JSON action. `inspect` reads one allowlisted source (`runs`, `schedule`, `delivery`, or `destination`); `ask` requests one allowed fact from a scripted customer; `recommend` and `escalate` end the episode. All reads are scoped to the fictional account. The simulator has no network access or production credentials and cannot perform a fix.
3. `inspect` and `ask` return deterministic observations with opaque evidence IDs, visible values, and explicit missing-data states. Fresh runs reset the world. Reject invented evidence IDs, unknown fields, or out-of-scope sources. Invalid actions consume a turn. An episode may use up to six actions including its terminal action; a timeout scores as a failed outcome.
4. A terminal action supplies an enumerated finding, a next step, and evidence IDs from this run. A shared vocabulary across worlds prevents the available actions from giving away the answer. Render reader-facing prose from the structured action; do not let an LLM judge decide correctness in v1.

Illustrative excerpt (IDs are examples, not stable labels):

```json
{"action":"inspect","source":"runs"}
```

The simulator returns `{"evidence_id":"obs-7","value":"no run since upgrade"}`. That alone does **not** prove the schedule is disabled. After checking `schedule` and observing `obs-8`, a supported ending is:

```json
{"action":"recommend","finding":"schedule_disabled","next_step":"review_and_enable_schedule","evidence":["obs-7","obs-8"]}
```

The grader accepts alternate evidence-backed paths. It checks the terminal finding, next step, and evidence predicates against what the model actually observed. A lucky guess, or a correct diagnosis backed by an irrelevant receipt, fails. In an underdetermined world, an explicit justified escalation is a success, not a failure to resolve.

## Data and splits

- **Pilot:** write ten distinct ticket openings, each with at least two plausible worlds. Include missing evidence, a misleading coincidence, and a case where asking one question is necessary. A support practitioner reviews the intended decisions. Use these cases for development only; never report this pilot as a sealed benchmark.
- **Training:** target 200 reviewed synthetic investigation transcripts for the first pass. Author a small set of case families, then vary symptoms, account details, event order, and wording independently of outcomes. Include warranted recommendations, warranted escalations, and cases where an extra question would waste the customer's time. For v1, support only domains for which the harness actually has fixtures and tools.
- **Validation:** reserve at least 20 separately authored paired openings for recipe selection. **Sealed evaluation:** target another 30 paired openings (at least 60 worlds) with both unseen scenarios in known families and whole failure families absent from training and validation. Include explicit unknowns. Publish actual counts by stratum; targets are not data.
- Assign splits by scenario group *before* paraphrasing or generating variants. Keep every sibling world, paraphrase, transcript and generation seed in the same split. Balance outcomes and permute opaque IDs so wording, list order and identifiers cannot encode the diagnosis. Check normalized hashes and manually review near matches. Keep sealed fixtures out of prompts, training, model selection and the interactive demo until scoring is complete.
- Store private grader state separately from visible messages. A training row records `scenario_group`, `split`, `messages` and annotation provenance; only visible messages and expert next actions enter the training formatter. No employer tickets, account data, copied logs, credentials or customer quotations. All cases must be independently publishable.

## Post-training recipe

**Pilot candidates:** try [`tiiuae/Falcon-H1-Tiny-Tool-Calling-90M`](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M/tree/fc099d791a229464b995fdd2fd24e8ffd2d699e8) first (`--backend falcon`). The original [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) is a fallback (`--backend smol`). Decide on the development pilot, then use the selected model and tokenizer revision for both benchmark arms. Check model, dataset and adapter redistribution terms before public release. An adapter alone is not a standalone model.

1. Run the unchanged model on the development pilot with a reasonable investigation prompt. Compare a simple evidence-first rule as a harness sanity check. Train only after the pilot identifies a recurring, learnable error; if it does not, revisit the task, not the leaderboard.
2. Freeze the base weights. Use supervised fine-tuning on expert **assistant action tokens only**, masking system prompts, customer turns, tool responses and padding. Unit-test the loss mask on a multi-turn example; skip overlength examples rather than truncating away the evidence or target without notice.
3. Starting recipe, not a measured optimum: LoRA on attention `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05, learning rate `1e-4`, effective batch 8, up to 3 epochs, 2,048-token sequences, seed 42. Verify target modules and trainable parameter count before launch. Select the checkpoint by validation evidence-backed success; break ties by fewer unsupported claims, then earlier checkpoint. Freeze prompt and recipe before sealed evaluation.
4. Save adapter weights/config, base and tokenizer revisions, chat template, data and split hashes, loss curve, package lock, hardware/dtype, actual optimizer steps, runtime and cost. Verify a nonzero adapter update and reload the saved adapter onto a fresh copy of the pinned base. **No paid compute or external upload is authorized by this spec.** Select hardware and obtain a budget before a paid run.

## What counts as evidence

Run base and base-plus-adapter on the **same** held-out worlds with the same prompt, examples, tools, turn limit, precision, and greedy decoding settings. Reset tool state and caches between episodes. Cap generated tokens per action identically. Record seed, environment and full visible trace for each arm. Do not silently drop parse errors, escalations, or timeouts.

The grader must reproduce the stored verdicts from raw action traces and pinned fixture content without loading model weights. Fresh generations on a different device or software build need not be identical; compare both conditions in the same environment and publish its details. In the structured pilot, the unsupported-claim count covers parsed recommendations without grounding, not arbitrary prose in malformed output; malformed actions still count as invalid.

| Measure | Definition |
| --- | --- |
| Primary: evidence-backed success | Episodes with an accepted terminal decision, next step and sufficient observed evidence / all attempted episodes. |
| Unsupported claim rate | Episodes asserting a cause or completed fix without sufficient observed evidence / all episodes. Report invalid and timed-out episodes separately. |
| Unnecessary questions | Count of `ask` actions for facts already known, obtainable from an available record, or irrelevant to any remaining valid decision; use pre-reviewed fixture rules. Report per episode. |
| Matched-pair success | Ticket pairs where both arms of the *scenario pair* were solved correctly by a given checkpoint / all pairs. Always guessing the same cause should fail this measure. |

Publish raw numerator/denominator, paired base-to-adapter changes, exposure to missing data, and a breakdown by known vs unseen families. Report wins, ties, regressions and a 95% interval for the primary difference resampled by opening (not individual sibling world). Report the sealed result once per model version; if used for tuning later, retire that test set.

**Claims gate:** “I fine-tuned a miniature model” requires a reproducible real training run and a reloadable adapter. “It investigated better on this benchmark” additionally requires a positive paired primary difference, a 95% interval excluding zero, and no increase in observed unsupported-claim rate. Otherwise describe the actual tradeoff or inconclusive result. Even a positive synthetic result is not proof of customer impact.

## Reader experience and deliverables

The public release has three entry points, all sourced from one repository:

- **View:** GitHub Pages serves static HTML, CSS, JavaScript, and saved JSON traces from `docs/demo/`. The pilot shows one trace at a time; after evaluation, show base and adapted actions side by side. Let readers step through the evidence, switch between sibling scenarios, and reveal the verdict. Label the scripted reference as such; label all saved runs as replays, not live inference. Include a regression or tie when available and link the full scorecard. Viewers need no login, model, helpdesk connection, or API key.
- **Evaluate or audit:** publish one Python CLI and a pinned dependency lock. Allow model runs locally (CPU first, CUDA optional) or via a Colab notebook calling the same CLI. Only the compute runtime needs the weights. Anyone can use the model-free `rescore` command to reconstruct tool observations, grades, and totals from a saved trace. For a released scorecard, publish the exact fixture snapshot after sealed evaluation is complete.
- **Retrain (after the gate):** add a training command and notebook cell once reviewed data, token-mask validation, and the pilot error analysis are ready. Publish the recipe, permitted adapter artifacts, and reproducibility metadata. Use a local runtime or cloud GPU runtime; no live inference service is required. Obtain budget approval before paid compute.

| Stage | Deliverable and acceptance check |
| --- | --- |
| 1. Simulator and pilot | Replay the ten paired openings deterministically. Test unavailable data, distinct sibling outcomes, valid alternate evidence paths, and rejection of invented evidence. Save unmodified baseline traces and a failure analysis. |
| 2. Reviewed corpus | Dataset card, split manifest, leakage checks, trainer input examples, and a token-mask test pass before training. Document case authorship and synthetic provenance. |
| 3. First adapter | Reproducible training command, weights or authorized artifact reference, hashes, metrics, and model card. Reload in a fresh process; confirm the base-only control uses no adapter. |
| 4. Evaluation and demo | One command scores both arms. Report every held-out episode and validate that a fresh replay reproduces the stored actions and scores. Show results even when they do not favor training. |

Repository layout: `fixtures/` for draft worlds; `harness/` for simulation, grading and the CLI; `notebooks/` for cloud entry points; `docs/demo/` for the static replay. `data/`, `training/`, and the official release scorecard are later deliverables.

**Pilot gate:** practitioner-review the ten draft openings and analyze the full unchanged-model trace with the reference as a harness check. Proceed to training only when the team identifies a recurring, learnable error. The lightweight static replay is not a gate for pilot scoring; live hosting, preference optimization, scratch pretraining, and production actions are out of scope.
