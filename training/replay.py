import argparse
import json
from pathlib import Path

from training.__main__ import audit
from training.cases import write_new


def main():
    parser = argparse.ArgumentParser(description="Build a standalone browser replay from audited post-training traces")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = {}
    for backend, label in (("smol", "SmolLM2 1.7B"), ("falcon", "Falcon 90M")):
        results = args.run / f"{backend}-eval"
        if not (results / "comparison.json").exists():
            continue
        comparison = audit(args.run, results)
        comparisons[backend] = {"label": label, "scorecard": comparison}
        for arm in ("base", "adapted"):
            comparisons[backend][arm] = json.loads((results / f"{arm}.json").read_text())["episodes"]
    if not comparisons:
        parser.error("The run has no completed paired comparisons")
    source = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    payload = json.dumps(comparisons, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    write_new(args.output, source.replace("__QUEUE_LAB_TRACES__", payload, 1))


if __name__ == "__main__":
    main()
