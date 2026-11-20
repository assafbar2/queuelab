import argparse
from html import escape
from pathlib import Path

from training.__main__ import audit
from training.cases import write_new


MODELS = (("smol", "SmolLM2 1.7B"), ("falcon", "Falcon 90M"))


def points(value):
    return f"{value * 100:+.1f}".replace("-", "−")


def model_row(result, label, top):
    before = result["base"]
    after = result["adapted"]
    lower, upper = result["bootstrap_95_interval"]
    primary = f"{points(result['primary_delta'])} pp · 95% interval {points(lower)} to {points(upper)} pp"
    unsupported = f"Unsupported recommendations: {before['unsupported_claims']} → {after['unsupported_claims']} episodes"
    outcome = "Met measured gain rule" if result["met_measured_gain_rule"] else "Did not meet measured gain rule"
    parts = [
        f'<rect x="50" y="{top}" width="1020" height="145" rx="18" fill="#fffdf7" stroke="#d2dace"/>',
        f'<text x="75" y="{top+40}" class="heading">{escape(label)}</text>',
        f'<text x="75" y="{top+72}" class="detail">{escape(primary)}</text>',
        f'<text x="75" y="{top+103}" class="detail">{escape(unsupported)}</text>',
        f'<text x="75" y="{top+127}" class="status">{escape(outcome)}</text>',
    ]
    for arm, metrics, fill, row in (("Before", before, "#788980", top+48), ("After", after, "#2f7257", top+104)):
        filled = 307 * metrics["successes"] / metrics["attempted"]
        parts.extend([
            f'<text x="523" y="{row+5}" class="detail">{arm}</text>',
            f'<rect x="600" y="{row-11}" width="307" height="22" rx="11" fill="#e9eae1"/>',
            f'<rect x="600" y="{row-11}" width="{filled:.2f}" height="22" rx="{min(filled / 2, 11):.2f}" fill="{fill}"/>',
            f'<text x="1042" y="{row+7}" text-anchor="end" class="score">{metrics["successes"]} / {metrics["attempted"]}</text>',
        ])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Render the measured scorecard as SVG")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    alt = []
    for index, (backend, label) in enumerate(MODELS):
        result = audit(args.run, args.run / f"{backend}-eval")
        rows.append(model_row(result, label, 185 + index * 164))
        alt.append(f"{label}: {result['base']['successes']} before and {result['adapted']['successes']} after, out of {result['base']['attempted']}.")
    source = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="579" viewBox="0 0 1120 579" role="img" aria-labelledby="title desc">
<title id="title">QueueLab measured post-training results</title>
<desc id="desc">{escape(' '.join(alt))} Synthetic engineering support evaluation.</desc>
<style>
text {{ fill:#1e392d; font-family:Arial,Helvetica,sans-serif; }}
.eyebrow {{ font-size:14px; font-weight:700; letter-spacing:1.5px; fill:#446f5d; }}
.headline {{ font-family:Georgia,'Times New Roman',serif; font-size:49px; letter-spacing:-1.7px; }}
.heading {{ font-family:Georgia,'Times New Roman',serif; font-size:29px; }}
.detail {{ font-size:17px; fill:#42594e; }}
.status {{ font-size:15px; font-weight:700; fill:#355d4b; }}
.score {{ font-size:23px; font-weight:700; }}
</style>
<rect x="1" y="1" width="1118" height="577" rx="25" fill="#f6f2e8" stroke="#d4d9cc" stroke-width="2"/>
<text x="51" y="47" class="eyebrow">QUEUELAB / ACTUAL LOCAL RUN</text>
<text x="49" y="106" class="headline">The same model, with and without LoRA.</text>
<text x="51" y="145" class="detail">Evidence-backed decisions on 60 held-out engineering support scenarios per arm.</text>
{''.join(rows)}
<path d="M51 529h1018" stroke="#d4d9cc"/>
<text x="51" y="557" class="detail">Synthetic cases from known failure families · Independent practitioner review pending</text>
</svg>
'''
    write_new(args.output, source)


if __name__ == "__main__":
    main()
