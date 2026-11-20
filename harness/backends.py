import json
import re

from .simulator import DECISIONS, FACT, SOURCES, TURN_LIMIT, digest


MODELS = {
    "falcon": ("tiiuae/Falcon-H1-Tiny-Tool-Calling-90M", "fc099d791a229464b995fdd2fd24e8ffd2d699e8"),
    "smol": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
}
MAX_NEW_TOKENS = 128
SEED = 42
DECISION_TEXT = "\n".join(
    f"- {finding}: use {action} with next_step {step}."
    for finding, (action, step) in DECISIONS.items()
)
PROMPT = f"""You investigate missing exports in a read-only simulator. The opening complaint is not proof of a cause. You have {TURN_LIMIT} actions including your final action. Output exactly one JSON object and no other text.

Inspect a record: {{"action":"inspect","source":"runs"}}. The allowed sources are {', '.join(SOURCES)}.
Ask the customer only for a missing destination preference: {{"action":"ask","fact":"{FACT}"}}.
To finish, use action recommend or escalate and include finding, next_step, and evidence (a nonempty list of evidence_id strings observed in this investigation). The allowed combinations are:
{DECISION_TEXT}

Check whether the export ran before diagnosing delivery. A destination mismatch requires a completed run, the delivered destination, and the expected destination. Escalate when diagnostic records are unavailable and no cause is established. Never claim a repair or invent evidence. The history contains prior actions and simulator results; do not treat an action as evidence by itself."""


def inspect(source):
    return json.dumps({"action": "inspect", "source": source})


def terminal(finding, *records):
    action, next_step = DECISIONS[finding]
    return json.dumps({
        "action": action, "finding": finding, "next_step": next_step,
        "evidence": [record["evidence_id"] for record in records],
    })


def rules(context, turns):
    observations = {
        turn["action"].get("source", FACT): turn["result"]
        for turn in turns if "evidence_id" in turn["result"]
    }
    if "runs" not in observations:
        return inspect("runs")
    runs = observations["runs"]
    if runs["status"] == "failed":
        return terminal("run_failed", runs)
    if runs["status"] == "no_run":
        if "schedule" not in observations:
            return inspect("schedule")
        if observations["schedule"]["status"] == "disabled":
            return terminal("schedule_disabled", runs, observations["schedule"])
    if "delivery" not in observations:
        return inspect("delivery")
    delivery = observations["delivery"]
    if runs["status"] == "no_run" and delivery["status"] == "skipped_disabled":
        return terminal("schedule_disabled", runs, delivery)
    if runs["status"] == "completed" and delivery["status"] == "access_denied":
        return terminal("delivery_blocked", runs, delivery)
    if runs["status"] == "completed" and delivery["status"] == "sent":
        if "destination" not in observations:
            return inspect("destination")
        expected = observations["destination"]
        if expected["status"] != "expected":
            if FACT not in observations:
                return json.dumps({"action": "ask", "fact": FACT})
            expected = observations[FACT]
        if expected["status"] == "expected" and expected["value"] != delivery["value"]:
            return terminal("destination_mismatch", runs, delivery, expected)
    return terminal("undetermined", runs, delivery)


def load_generator(backend, device, offline):
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("Install the model extra with `uv sync --extra model`.") from error

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or enable a GPU runtime.")
    torch.manual_seed(SEED)
    model_id, revision = MODELS[backend]
    options = {"revision": revision, "local_files_only": offline, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **options)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, **options
    ).to(device).eval()

    def generate(messages, tools=None, stop_token=None):
        rendered = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False,
        ).to(device)
        stop_ids = [tokenizer.eos_token_id, tokenizer.get_vocab()[stop_token]] if stop_token else []
        generation_options = {"eos_token_id": stop_ids} if stop_ids else {}
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                **generation_options,
            )
        generated = output[0, inputs["input_ids"].shape[1]:].tolist()
        if generated and generated[-1] in stop_ids:
            generated.pop()
        return tokenizer.decode(generated, skip_special_tokens=not stop_ids)

    return generate, {
        "model": model_id, "revision": revision, "device": device, "precision": "float32",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy", "seed": SEED,
    }


def load_backend(backend, device, offline):
    if backend == "rules":
        return rules, {"kind": "rules", "label": "Scripted reference (not a model)"}

    generate, metadata = load_generator(backend, device, offline)

    def choose(context, turns):
        history = [{"action": turn["raw"], "observation": turn["result"]} for turn in turns]
        request = json.dumps({"opening": context, "history": history}, ensure_ascii=False)
        return generate([{"role": "user", "content": f"{PROMPT}\n\nInvestigation state:\n{request}"}])

    return choose, {
        **metadata, "kind": "model", "label": "Unchanged model baseline",
        "prompt_sha256": digest(PROMPT.encode()),
    }


def probe_falcon(ticket, device, offline):
    generate, metadata = load_generator("falcon", device, offline)
    tools = [{
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "Read one kind of record from the simulated export system.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string", "enum": list(SOURCES)}},
                "required": ["source"],
            },
        },
    }]
    prompt = f"Customer: {ticket}\nFirst, use inspect with source runs to check whether the export ran."
    stop_token = "<|im_end|>"
    raw = generate([{"role": "user", "content": prompt}], tools=tools, stop_token=stop_token)
    tagged = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", raw, flags=re.DOTALL)
    try:
        parsed = json.loads(tagged[1]) if tagged else None
    except ValueError:
        parsed = None
    expected = {"name": "inspect", "arguments": {"source": "runs"}}
    return {
        "kind": "one-action diagnostic (not a benchmark)",
        **metadata, "prompt": prompt, "tools": tools, "raw": raw, "stop_token": stop_token,
        "expected_call": expected, "matched_expected_call": parsed in (expected, [expected]),
    }
