import argparse
import gc
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

from harness.__main__ import rescore
from harness.backends import MODELS
from harness.simulator import digest, episode, score
from training.cases import encode_json, prepare, visible_prompt, write_new


SEED = 42
MAX_LENGTH = 1024
MAX_ACTION_TOKENS = 96
LORA = {"rank": 8, "scale": 2.0, "dropout": 0.05, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)


def token_examples(path, tokenizer):
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or tokenizer.convert_ids_to_tokens(end_id) != "<|im_end|>":
        end_id = tokenizer.eos_token_id
    if end_id is None:
        raise ValueError("Expected a tokenizer end-of-turn token")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        prompt = render(tokenizer, entry["prompt"])
        target = tokenizer.encode(entry["completion"], add_special_tokens=False)
        if not target or tokenizer.decode(target) != entry["completion"]:
            raise ValueError("Action tokens do not round-trip")
        tokens = [*prompt, *target, end_id]
        if len(tokens) > MAX_LENGTH:
            raise ValueError("Sequence would be truncated; revise the dataset before training")
        result.append((tokens, len(prompt)))
    return result


def target_indices(sequence_length, prompt_length):
    return range(prompt_length, sequence_length)


def masked_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn

    targets = batch[:, 1:]
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, :1]) & (positions < lengths[:, 1:])
    losses = nn.losses.cross_entropy(model(batch[:, :-1]), targets).astype(mx.float32)
    tokens = mask.sum()
    return mx.where(mask, losses, 0).sum() / tokens, tokens


def read_verified(data, manifest, split, kind):
    filename, check = (f"{split}.jsonl", "row_sha256") if kind == "rows" else (f"{split}-cases.json", "fixture_sha256")
    path = data / filename
    if digest(path.read_bytes()) != manifest["partitions"][split][check]:
        raise ValueError(f"Dataset hash mismatch: {filename}")
    return path


def fit(args):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model_id, revision = MODELS[args.backend]
    manifest = json.loads((args.data / "manifest.json").read_text())
    mx.random.seed(SEED)
    np.random.seed(SEED)
    model, tokenizer = load(model_id, revision=revision)
    train_rows = token_examples(read_verified(args.data, manifest, "train", "rows"), tokenizer)
    valid_rows = token_examples(read_verified(args.data, manifest, "valid", "rows"), tokenizer)
    if len(train_rows) < args.batch or len(valid_rows) < args.batch:
        raise ValueError("Batch size exceeds available rows")
    largest, prefix = max(train_rows + valid_rows, key=lambda item: len(item[0]))
    selected = target_indices(len(largest), prefix)
    if not selected or selected.start != prefix or selected.stop != len(largest):
        raise ValueError("Assistant mask check failed")
    if not tokenizer.decode(largest[prefix:-1]).startswith('{"action"'):
        raise ValueError("Loss mask must start at the next assistant action")
    total = sum(tensor.size for _, tensor in tree_flatten(model.parameters()))
    model.freeze()
    layers = len(model.layers)
    linear_to_lora_layers(model, layers, LORA)
    adapters = tree_flatten(model.trainable_parameters())
    if len(adapters) != 4 * layers or any(not key.endswith((".lora_a", ".lora_b")) for key, _ in adapters):
        raise ValueError("Unexpected trainable parameters; expected q/v LoRA only")
    trainable = sum(tensor.size for _, tensor in adapters)
    args.output.mkdir(parents=True, exist_ok=False)
    weights_path = args.output / "adapters.safetensors"
    config = {
        "model": model_id, "revision": revision, "fine_tune_type": "lora", "num_layers": layers,
        "lora_parameters": LORA, "seed": SEED, "steps": args.steps, "batch_size": args.batch,
        "learning_rate": args.rate, "base_parameters": total, "trainable_parameters": trainable,
        "base_dtypes": sorted({str(tensor.dtype) for _, tensor in tree_flatten(model.parameters())}),
        "precision": "pinned checkpoint dtypes, no quantization", "runtime": platform.platform(),
        "versions": {key: version(key) for key in ("mlx", "mlx-lm", "transformers")},
        "mask": "next assistant action and end-of-turn token only; prompt and padding excluded",
        "largest_sequence": len(largest), "mask_probe": {"prompt": prefix, "target": len(selected)},
        "data": manifest,
    }
    write_new(args.output / "adapter_config.json", encode_json(config))
    print(json.dumps(config, indent=2), flush=True)
    start = time.monotonic()
    train(
        model, optim.Adam(learning_rate=args.rate), train_rows, valid_rows,
        args=TrainingArgs(batch_size=args.batch, iters=args.steps, val_batches=8,
                          steps_per_report=10, steps_per_eval=max(20, args.steps // 2),
                          steps_per_save=args.steps + 1, adapter_file=weights_path,
                          max_seq_length=MAX_LENGTH),
        loss=masked_loss,
    )
    weights = mx.load(str(weights_path))
    changed = [key for key, value in weights.items()
               if key.endswith(".lora_b") and mx.any(value != 0).item()]
    if not changed or any(not mx.all(mx.isfinite(value)).item() for value in weights.values()):
        raise ValueError("Adapter is zero or contains non-finite values")
    summary = {"seconds": round(time.monotonic() - start, 2), "optimizer_steps": args.steps,
               "updated_lora_b_tensors": len(changed), "adapter_sha256": digest(weights_path.read_bytes())}
    write_new(args.output / "training-result.json", encode_json(summary))
    return summary


def compare_reports(base, adapted):
    paired = {}
    for before, after in zip(base["episodes"], adapted["episodes"], strict=True):
        if before["world_id"] != after["world_id"]:
            raise ValueError("Different case order in comparison")
        delta = int(after["verdict"]["outcome"] == "success") - int(before["verdict"]["outcome"] == "success")
        paired.setdefault(before["opening_id"], []).append(delta)
    group_means = [sum(group) / len(group) for group in paired.values()]
    rng = random.Random(SEED)
    samples = sorted(sum(rng.choices(group_means, k=len(group_means))) / len(group_means) for _ in range(4000))
    gained = sum(delta == 1 for group in paired.values() for delta in group)
    lost = sum(delta == -1 for group in paired.values() for delta in group)
    lower, upper = samples[100], samples[3899]
    unsupported_change = adapted["metrics"]["unsupported_claims"] - base["metrics"]["unsupported_claims"]
    return {"base": base["metrics"], "adapted": adapted["metrics"], "gains": gained, "regressions": lost,
            "primary_delta": sum(group_means) / len(group_means), "bootstrap_95_interval": [lower, upper],
            "unsupported_claim_episode_delta": unsupported_change,
            "met_measured_gain_rule": lower > 0 and unsupported_change <= 0,
            "scope": "Synthetic holdout of known failure families; no independent practitioner review or production claim."}


def compare(args):
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    holdout_path = read_verified(args.data, manifest, "holdout", "cases")
    holdout = json.loads(holdout_path.read_text())
    fixture_hash = digest(holdout_path.read_bytes())
    model_id, revision = MODELS[args.backend]
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    trained = json.loads((args.adapter / "training-result.json").read_text())
    weights_hash = digest((args.adapter / "adapters.safetensors").read_bytes())
    if (config["model"], config["revision"], config["data"], trained["adapter_sha256"]) != (model_id, revision, manifest, weights_hash):
        raise ValueError("Adapter, model revision, or split manifest does not match the training run")
    arms = {}
    for label, path in (("base", None), ("adapted", args.adapter)):
        print(f"Loading fresh {label} checkpoint", flush=True)
        model, tokenizer = load(model_id, revision=revision, adapter_path=str(path) if path else None)
        if tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<|im_end|>")) == "<|im_end|>":
            tokenizer.add_eos_token("<|im_end|>")
        sampler = make_sampler(temp=0.0)

        def choose(context, history):
            return generate(model, tokenizer, prompt=render(tokenizer, visible_prompt(context, history)),
                            max_tokens=MAX_ACTION_TOKENS, sampler=sampler)

        traces = []
        start = time.monotonic()
        for opening in holdout["openings"]:
            for world in opening["worlds"]:
                trace = episode(opening, world, choose)
                traces.append(trace)
                print(f"{label} {world['id']}: {trace['verdict']['outcome']}", flush=True)
        report = {"schema_version": 1, "split": "holdout", "fixture_sha256": fixture_hash,
                  "opening_ids": [opening["id"] for opening in holdout["openings"]],
                  "backend": {"label": label, "model": model_id, "revision": revision,
                              "adapter_sha256": weights_hash if path else None,
                              "max_action_tokens": MAX_ACTION_TOKENS, "decoding": "greedy"},
                  "seconds": round(time.monotonic() - start, 2), "episodes": traces, "metrics": score(traces)}
        report_path = args.output / f"{label}.json"
        write_new(report_path, encode_json(report))
        rescore(report_path, holdout, fixture_hash)
        arms[label] = report
        del model, tokenizer, sampler
        gc.collect()
        mx.clear_cache()
    result = compare_reports(arms["base"], arms["adapted"])
    write_new(args.output / "comparison.json", encode_json(result))
    return result


def audit(data, results):
    manifest = json.loads((data / "manifest.json").read_text())
    holdout_path = read_verified(data, manifest, "holdout", "cases")
    fixture_hash = digest(holdout_path.read_bytes())
    holdout = json.loads(holdout_path.read_text())
    arms = []
    for name in ("base", "adapted"):
        report = results / f"{name}.json"
        rescore(report, holdout, fixture_hash)
        arms.append(json.loads(report.read_text()))
    expected = [opening["id"] for opening in holdout["openings"]]
    if any(report["opening_ids"] != expected for report in arms):
        raise ValueError("Expected the complete holdout in both arms")
    if arms[0]["backend"]["adapter_sha256"] is not None or not arms[1]["backend"]["adapter_sha256"]:
        raise ValueError("Expected a base-only arm and a reloaded adapter arm")
    for key in ("model", "revision", "max_action_tokens", "decoding"):
        if arms[0]["backend"][key] != arms[1]["backend"][key]:
            raise ValueError(f"The two arms differ on {key}")
    result = compare_reports(*arms)
    if result != json.loads((results / "comparison.json").read_text()):
        raise ValueError("Paired scorecard does not match the raw traces")
    return result


def main():
    parser = argparse.ArgumentParser(description="QueueLab local post-training experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("audit")
    verify.add_argument("--data", type=Path, required=True)
    verify.add_argument("--results", type=Path, required=True)
    for command in ("fit", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--backend", choices=MODELS, required=True)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if command == "fit":
            sub.add_argument("--steps", type=int, default=180)
            sub.add_argument("--batch", type=int, default=4)
            sub.add_argument("--rate", type=float, default=2e-4)
        else:
            sub.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit" and (args.steps < 1 or args.batch < 1 or args.rate <= 0):
        parser.error("Training steps, batch size, and learning rate must be positive")
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "fit":
        result = fit(args)
    elif args.command == "compare":
        result = compare(args)
    else:
        result = audit(args.data, args.results)
    print(encode_json(result))


if __name__ == "__main__":
    main()
