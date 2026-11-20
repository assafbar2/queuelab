import json
from pathlib import Path

from harness.backends import rules
from harness.simulator import DECISIONS, FACT, digest, episode


TRAIN_TICKETS = [
    f"{question} The {report} was due {timing}."
    for report in ("inventory file", "invoices export", "usage report", "partner feed", "orders archive", "activity extract", "finance snapshot", "operations export")
    for question, timing in (
        ("Our transfer never appeared.", "overnight"),
        ("We are waiting on a missing file.", "at the start of the shift"),
        ("Can you investigate a gap in our downloads?", "just after midnight"),
        ("The team can't find the scheduled output.", "before opening"),
        ("We're missing the most recent delivery.", "this morning"),
    )
]

VALIDATION_TICKETS = [
    f"The {report} isn't where the team normally gets it. {question}"
    for report in ("ledger copy", "retention list", "audit extract", "analytics drop", "campaign summary", "capacity snapshot", "stock report", "worklog CSV", "daily archive", "billing file")
    for question in ("Can you check our scheduled job?", "What should we look at first?")
]

HOLDOUT_TICKETS = [
    "The warehouse loader is waiting for today's manifest; there is nothing in the intake directory.",
    "We have a gap in the nightly account extracts. Can you tell us where to look?",
    "The on-call analyst flagged that the 03:00 telemetry drop hasn't shown up.",
    "No export attachment came through for the weekly settlement. Do you see anything on your side?",
    "Yesterday we had a reconciliation file. Today the receiving job timed out waiting for it.",
    "The audit team expected a fresh snapshot after the maintenance window. Their bucket is empty.",
    "The upstream dashboard looks healthy, but the customer inventory snapshot is absent.",
    "The month rolled over and our compliance handoff is still missing the scheduled extract.",
    "We didn't get a new CSV for the overnight backfill. Please check the job, not just the notifications.",
    "The carrier integration has nothing to ingest this morning. It depends on your dispatch export.",
    "Can you trace the missing activity dump for the last shift? Nobody here received it.",
    "We're trying to close out the quarter and the final batch of line items isn't in the folder.",
    "A downstream alert says the attendance export is stale. Is there a record of today's delivery?",
    "We rotated the destination last night. The availability report didn't land where we expected.",
    "The delivery we're chasing is two weeks old. Can you still check whether it made it out?",
    "The bulk user list never reached the file watcher. It was supposed to arrive before standup.",
    "We've been watching the destination since six, but no updated claims extract has arrived.",
    "The automated import ran against an old file because a new reservations export wasn't present.",
    "Nothing turned up for the weekend meter export. We need to decide whether to retry it.",
    "The error feed usually lands on a schedule. There is a hole between Thursday and Saturday.",
    "Can you find the batch of renewal events that the receiving system couldn't locate?",
    "The team is ready to process today's contract report, but the location we checked has no file.",
    "Monday's archive didn't make it to the shared inbox. What evidence is available?",
    "After moving the integration, we can't find the scheduled line-items output.",
    "The nightly roster is missing, and we're not sure whether to change a setting or wait.",
    "Our downstream index isn't current. The hourly export it consumes hasn't arrived.",
    "We can't match the payout register to today's balance because the scheduled file is missing.",
    "The analyst asked for yesterday's shipment history, but the scheduled batch is absent.",
    "The operations handoff notes say a new file should be there. The receiving service found nothing.",
    "No export appeared at the new drop point. Please check what happened before we rerun it.",
]


def visible_prompt(context, turns):
    history = [{"action": turn["action"], "record": turn["result"]} for turn in turns]
    instructions = (
        'Investigate an export ticket. Give exactly one JSON action. Check runs first. '
        'Read a record with {"action":"inspect","source":"runs"}. Sources: runs, schedule, delivery, destination. '
        'If the expected destination is not recorded, ask with '
        '{"action":"ask","fact":"expected_destination"}. '
        'End with action recommend or escalate, and fields finding, next_step, evidence. '
        'Copy the evidence_id strings from relevant records. Never invent evidence. '
        'Allowed endings: schedule_disabled → review_and_enable_schedule; '
        'destination_mismatch → correct_destination; run_failed → inspect_run_error; '
        'delivery_blocked → review_delivery_access; undetermined → recover_diagnostic_records (escalate).'
    )
    return instructions + "\nTicket: " + context["customer_message"] + "\nHistory: " + json.dumps(history, ensure_ascii=False, separators=(",", ":"))


def scenarios(split, tickets, pairs=False):
    openings = []
    endings = tuple(DECISIONS)
    for index, ticket in enumerate(tickets):
        group = f"{split}-{index + 1:02d}"
        label = f"{split}-archive-{index + 18}"
        hour = f"{index % 8 + 1:02}:00"
        if split == "holdout":
            no_run = f"No invocation logged since the last scheduled window ({hour})."
            disabled = "Automatic dispatch is switched off for this account."
            completed = f"The output file was generated successfully at {hour}."
            failed = "A worker error interrupted generation before the file was written."
            denied = "The receiving service refused the write; permission denied."
            missing = "No retained records cover this delivery window."
        else:
            no_run = f"No execution entry for the {hour} slot."
            disabled = "Scheduler status: disabled."
            completed = f"Run finished successfully at {hour}."
            failed = "Export execution failed before producing output."
            denied = "Upload rejected because the writer lacks access."
            missing = "This interval is missing from retained logs."
        route = FACT if index % 4 == 1 else "destination"
        cases = {
            "schedule_disabled": ({"runs": ["no_run", no_run], "schedule": ["disabled", disabled]}, ["runs", "schedule"]),
            "destination_mismatch": ({"runs": ["completed", completed], "delivery": ["sent", f"{label}/old"], route: ["expected", f"{label}/new"]}, ["runs", "delivery", route]),
            "run_failed": ({"runs": ["failed", failed]}, ["runs"]),
            "delivery_blocked": ({"runs": ["completed", completed], "delivery": ["access_denied", denied]}, ["runs", "delivery"]),
            "undetermined": ({"runs": ["unavailable", missing], "delivery": ["unavailable", "No receipt is available."]}, ["runs", "delivery"]),
        }
        selected = (endings[index % 5], endings[(index + 2) % 5]) if pairs else endings
        worlds = []
        for position, finding in enumerate(selected):
            records, required = cases[finding]
            worlds.append({"id": f"{group}-{position + 1}", "finding": finding, "records": records, "paths": [required], "ask_useful": FACT in required})
        openings.append({"id": group, "ticket": ticket, "worlds": worlds})
    return openings


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_new(path, contents):
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(contents)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    partitions = {"train": TRAIN_TICKETS, "valid": VALIDATION_TICKETS, "holdout": HOLDOUT_TICKETS}
    normalized = [" ".join(ticket.casefold().split()) for tickets in partitions.values() for ticket in tickets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate opening in the training, validation, or holdout partitions")
    manifest = {"provenance": "Synthetic cases and deterministic reference decisions; no independent practitioner review.", "partitions": {}}
    for split, tickets in partitions.items():
        openings = scenarios(split, tickets, pairs=split == "holdout")
        info = {"groups": len(openings), "worlds": sum(len(opening["worlds"]) for opening in openings)}
        case_path = output / f"{split}-cases.json"
        write_new(case_path, encode_json({"split": split, "openings": openings}))
        info["fixture_sha256"] = digest(case_path.read_bytes())
        if split != "holdout":
            examples = []
            for opening in openings:
                for world in opening["worlds"]:
                    def teacher(context, history):
                        action = rules(context, history)
                        examples.append({"prompt": visible_prompt(context, history), "completion": action})
                        return action

                    trace = episode(opening, world, teacher)
                    if trace["verdict"]["outcome"] != "success":
                        raise ValueError(f"Reference failed to meet expected outcome in {world['id']}")
            rows_path = output / f"{split}.jsonl"
            write_new(rows_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
            info.update(rows=len(examples), row_sha256=digest(rows_path.read_bytes()))
        manifest["partitions"][split] = info
    write_new(output / "manifest.json", encode_json(manifest))
    return manifest
