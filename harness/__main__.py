import argparse
import json
import platform
import sys
from pathlib import Path

from .backends import MODELS, load_backend, probe_falcon
from .simulator import episode, load_pilot, score


def run(args, pilot, fixture_hash):
    openings = pilot["openings"][:args.limit_openings]
    choose, backend = load_backend(args.backend, args.device, args.offline)
    episodes = []
    for opening in openings:
        for world in opening["worlds"]:
            entry = episode(opening, world, choose)
            episodes.append(entry)
            print(f"{world['id']}: {entry['verdict']['outcome']}", file=sys.stderr)
    report = {
        "schema_version": 1,
        "split": pilot["split"],
        "provenance": pilot["provenance"],
        "fixture_sha256": fixture_hash,
        "opening_ids": [opening["id"] for opening in openings],
        "backend": backend,
        "environment": {"python": platform.python_version(), "system": platform.system(), "arch": platform.machine()},
        "episodes": episodes,
        "metrics": score(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return report["metrics"]


def rescore(path, pilot, fixture_hash):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != 1 or report["fixture_sha256"] != fixture_hash:
        raise ValueError("Report schema or fixture hash does not match this checkout.")
    selected = report["opening_ids"]
    openings = [opening for opening in pilot["openings"] if opening["id"] in selected]
    if not openings or selected != [opening["id"] for opening in openings]:
        raise ValueError("Report has unknown, duplicate, or out-of-order openings.")
    expected = [(opening, world) for opening in openings for world in opening["worlds"]]
    if len(expected) != len(report["episodes"]):
        raise ValueError("Report is missing episodes or includes extra episodes.")

    rescored = []
    for (opening, world), saved in zip(expected, report["episodes"]):
        raw = iter(turn["raw"] for turn in saved["turns"])
        try:
            computed = episode(opening, world, lambda context, turns: next(raw))
        except StopIteration as error:
            raise ValueError(f"Incomplete action trace for {world['id']}.") from error
        if computed != saved:
            raise ValueError(f"Trace does not replay exactly for {world['id']}.")
        rescored.append(computed)
    metrics = score(rescored)
    if metrics != report["metrics"]:
        raise ValueError("Recomputed metrics do not match the report.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run or verify the QueueLab development pilot.")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Generate traces in the read-only simulator")
    runner.add_argument("--backend", choices=["rules", *MODELS], default="rules")
    runner.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    runner.add_argument("--offline", action="store_true", help="Use cached model files only")
    runner.add_argument("--limit-openings", type=int, choices=range(1, 11), default=10)
    runner.add_argument("--output", type=Path, required=True, help="New report path (will not overwrite)")
    audit = commands.add_parser("rescore", help="Verify a saved trace without loading a model")
    audit.add_argument("report", type=Path)
    probe = commands.add_parser("probe-falcon", help="Try one native Falcon tool call without executing it")
    probe.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    probe.add_argument("--offline", action="store_true", help="Use cached model files only")
    args = parser.parse_args()
    try:
        pilot, fixture_hash = load_pilot()
        if args.command == "run" and args.output.exists():
            raise FileExistsError(f"Choose a new output path: {args.output}")
        if args.command == "probe-falcon":
            result = probe_falcon(pilot["openings"][0]["ticket"], args.device, args.offline)
        elif args.command == "run":
            result = run(args, pilot, fixture_hash)
        else:
            result = rescore(args.report, pilot, fixture_hash)
    except KeyboardInterrupt:
        parser.exit(130, "queuelab: interrupted; no completed scorecard.\n")
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        parser.exit(1, f"queuelab: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
