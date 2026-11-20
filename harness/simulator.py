import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "pilot.json"
SOURCES = ("runs", "schedule", "delivery", "destination")
FACT = "expected_destination"
TURN_LIMIT = 6
DECISIONS = {
    "schedule_disabled": ("recommend", "review_and_enable_schedule"),
    "destination_mismatch": ("recommend", "correct_destination"),
    "run_failed": ("recommend", "inspect_run_error"),
    "delivery_blocked": ("recommend", "review_delivery_access"),
    "undetermined": ("escalate", "recover_diagnostic_records"),
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_pilot():
    contents = FIXTURES.read_bytes()
    pilot = json.loads(contents)
    openings = pilot["openings"]
    ids = set()
    if len(openings) != 10 or len({item["id"] for item in openings}) != 10:
        raise ValueError("Pilot must have ten unique openings")
    for opening in openings:
        worlds = opening["worlds"]
        if len(worlds) < 2 or len({world["finding"] for world in worlds}) != len(worlds):
            raise ValueError(f"{opening['id']}: expected distinct sibling findings")
        for world in worlds:
            records = world["records"]
            if world["id"] in ids or not records.keys() <= {*SOURCES, FACT}:
                raise ValueError(f"Invalid fixture {world['id']}")
            ids.add(world["id"])
            if world["finding"] not in DECISIONS or not world["paths"]:
                raise ValueError(f"Missing grading rules for {world['id']}")
            if any(not path or not set(path) <= records.keys() for path in world["paths"]):
                raise ValueError(f"Unknown evidence in fixture {world['id']}")
    return pilot, digest(contents)


def visible_opening(opening):
    return {
        "customer_message": opening["ticket"],
        "account": "fictional-export-account",
        "clock": "2026-11-19T10:00:00Z",
    }


def read_record(world, key):
    record = world["records"].get(key)
    evidence_id = "obs-" + digest(f"{world['id']}:{key}".encode())[:12]
    if record is None:
        return {"evidence_id": evidence_id, "status": "unavailable", "value": None}
    state, detail = record
    return {"evidence_id": evidence_id, "status": state, "value": detail}


def parse_action(text):
    def unique_fields(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("Duplicate JSON field")
        return fields

    try:
        action = json.loads(text, object_pairs_hook=unique_fields)
    except (ValueError, RecursionError):
        return None
    return action if isinstance(action, dict) else None


def valid_terminal(action, kind, seen):
    if action.keys() != {"action", "finding", "next_step", "evidence"}:
        return False
    finding, next_step, evidence = action["finding"], action["next_step"], action["evidence"]
    return (
        isinstance(finding, str)
        and DECISIONS.get(finding) == (kind, next_step)
        and isinstance(evidence, list)
        and bool(evidence)
        and all(isinstance(item, str) and item in seen for item in evidence)
        and len(evidence) == len(set(evidence))
    )


def episode(opening, world, choose):
    context = visible_opening(opening)
    turns = []
    seen = {}
    observed_keys = set()
    invalid_actions = 0
    unnecessary_questions = 0
    unsupported = False
    outcome = "timeout"
    reason = "No valid terminal action within six actions."
    for _ in range(TURN_LIMIT):
        raw = choose(context, tuple(turns))
        action = parse_action(raw)
        kind = action.get("action") if action is not None else None
        result = {"error": "Action violates the JSON contract or cites unseen evidence."}
        terminal = False
        if kind == "inspect" and action.keys() == {"action", "source"} and action["source"] in SOURCES:
            key = action["source"]
            result = read_record(world, key)
            seen[result["evidence_id"]] = key
            observed_keys.add(key)
        elif kind == "ask" and action == {"action": "ask", "fact": FACT}:
            if not world.get("ask_useful") or FACT in observed_keys:
                unnecessary_questions += 1
            result = read_record(world, FACT)
            seen[result["evidence_id"]] = FACT
            observed_keys.add(FACT)
        elif kind in ("recommend", "escalate"):
            finding = action.get("finding")
            evidence = action.get("evidence")
            cited = {
                seen[item] for item in evidence if isinstance(item, str) and item in seen
            } if isinstance(evidence, list) else set()
            grounded = finding == world["finding"] and any(set(path) <= cited for path in world["paths"])
            if kind == "recommend" and not grounded:
                unsupported = True
            if valid_terminal(action, kind, seen):
                terminal = True
                outcome = "success" if grounded else "failed"
                reason = (
                    "The finding, next step, and cited observations satisfy the fixture."
                    if grounded else "The finding or cited evidence does not support this decision."
                )
                result = {"accepted_terminal": True}
        invalid_actions += int("error" in result)
        turns.append({"raw": raw, "action": action, "result": result})
        if terminal:
            break
    return {
        "opening_id": opening["id"],
        "world_id": world["id"],
        "opening": context,
        "turns": turns,
        "verdict": {
            "outcome": outcome,
            "reason": reason,
            "expected_finding": world["finding"],
            "expected_next_step": DECISIONS[world["finding"]][1],
            "unsupported_claim": unsupported,
            "invalid_actions": invalid_actions,
            "unnecessary_questions": unnecessary_questions,
        },
    }


def score(episodes):
    groups = {}
    for entry in episodes:
        groups.setdefault(entry["opening_id"], []).append(entry["verdict"])
    verdicts = [entry["verdict"] for entry in episodes]
    return {
        "attempted": len(verdicts),
        "successes": sum(verdict["outcome"] == "success" for verdict in verdicts),
        "timeouts": sum(verdict["outcome"] == "timeout" for verdict in verdicts),
        "episodes_with_invalid_actions": sum(verdict["invalid_actions"] > 0 for verdict in verdicts),
        "unsupported_claims": sum(verdict["unsupported_claim"] for verdict in verdicts),
        "unnecessary_questions": sum(verdict["unnecessary_questions"] for verdict in verdicts),
        "matched_openings": len(groups),
        "matched_opening_successes": sum(
            all(verdict["outcome"] == "success" for verdict in group) for group in groups.values()
        ),
    }
