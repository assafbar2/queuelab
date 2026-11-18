# QueueLab v1: train a miniature support investigator

**Deliverable in this repository today:** a specification. All datasets, traces, metrics, interfaces, and release artifacts below are proposed unless explicitly identified as examples. There are no trained weights or observed results yet.

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

**Proposed base:** [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674), pinned revision `31b70e2e869a7173562077fd711b654946d38674`. Use that revision and tokenizer for both arms. Check model, dataset and adapter redistribution terms before any public release.

1. Run the unchanged model on the development pilot with a reasonable investigation prompt. Compare a simple evidence-first rule as a harness sanity check. Train only after the pilot identifies a recurring, learnable error; if it does not, revisit the task, not the leaderboard.
2. Freeze the base weights. Use supervised fine-tuning on expert **assistant action tokens only**, masking system prompts, customer turns, tool responses and padding. Unit-test the loss mask on a multi-turn example; skip overlength examples rather than truncating away the evidence or target without notice.
3. Starting recipe, not a measured optimum: LoRA on attention `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05, learning rate `1e-4`, effective batch 8, up to 3 epochs, 2,048-token sequences, seed 42. Verify target modules and trainable parameter count before launch. Select the checkpoint by validation evidence-backed success; break ties by fewer unsupported claims, then earlier checkpoint. Freeze prompt and recipe before sealed evaluation.
4. Save adapter weights/config, base and tokenizer revisions, chat template, data and split hashes, loss curve, package lock, hardware/dtype, actual optimizer steps, runtime and cost. Verify a nonzero adapter update and reload the saved adapter onto a fresh copy of the pinned base. **No paid compute or external upload is authorized by this spec.** Select hardware and obtain a budget before a paid run.

## What counts as evidence

Run base and base-plus-adapter on the **same** held-out worlds with the same prompt, examples, tools, turn limit, precision, and greedy decoding settings. Reset tool state and caches between episodes. Cap generated tokens per action identically. Record seed, environment and full visible trace for each arm. Do not silently drop parse errors, escalations, or timeouts.

| Measure | Definition |
| --- | --- |
| Primary: evidence-backed success | Episodes with an accepted terminal decision, next step and sufficient observed evidence / all attempted episodes. |
| Unsupported claim rate | Episodes asserting a cause or completed fix without sufficient observed evidence / all episodes. Report invalid and timed-out episodes separately. |
| Unnecessary questions | Count of `ask` actions for facts already known, obtainable from an available record, or irrelevant to any remaining valid decision; use pre-reviewed fixture rules. Report per episode. |
| Matched-pair success | Ticket pairs where both arms of the *scenario pair* were solved correctly by a given checkpoint / all pairs. Always guessing the same cause should fail this measure. |

Publish raw numerator/denominator, paired base-to-adapter changes, exposure to missing data, and a breakdown by known vs unseen families. Report wins, ties, regressions and a 95% interval for the primary difference resampled by opening (not individual sibling world). Report the sealed result once per model version; if used for tuning later, retire that test set.

**Claims gate:** “I fine-tuned a miniature model” requires a reproducible real training run and a reloadable adapter. “It investigated better on this benchmark” additionally requires a positive paired primary difference, a 95% interval excluding zero, and no increase in observed unsupported-claim rate. Otherwise describe the actual tradeoff or inconclusive result. Even a positive synthetic result is not proof of customer impact.

## Reader experience and deliverables

A browser replay starts on a ticket, shows base and adapted actions side by side, opens the evidence each saw, then reveals what really happened and why each verdict was scored. Use saved traces from actual runs and label them as replays, not live inference. Include a regression or tie when available and link the complete scorecard. No login, helpdesk integration or API key for the reader. A future book link is a separate publication decision.

| Stage | Deliverable and acceptance check |
| --- | --- |
| 1. Simulator and pilot | Replay the ten paired openings deterministically. Test unavailable data, distinct sibling outcomes, valid alternate evidence paths, and rejection of invented evidence. Save unmodified baseline traces and a failure analysis. |
| 2. Reviewed corpus | Dataset card, split manifest, leakage checks, trainer input examples, and a token-mask test pass before training. Document case authorship and synthetic provenance. |
| 3. First adapter | Reproducible training command, weights or authorized artifact reference, hashes, metrics, and model card. Reload in a fresh process; confirm the base-only control uses no adapter. |
| 4. Evaluation and demo | One command scores both arms. Report every held-out episode and validate that a fresh replay reproduces the stored actions and scores. Show results even when they do not favor training. |

Planned ownership: `fixtures/` and `data/` for worlds and reviewed splits; `harness/` for simulation and grading; `training/` for the recipe; `reports/` and `demo/` for the scorecard and replay. These paths describe future implementation, not files already present.

**First task:** author the export example and nine more paired openings, then run the unmodified baseline. No web UI, hosted inference, preference optimization, scratch pretraining, real customer actions, or general-purpose support chatbot is needed to reach that milestone.
