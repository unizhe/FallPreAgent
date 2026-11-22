"""Write the P0 synthetic decision examples to a new ignored output folder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallpre.decision_config import DecisionConfigError, load_decision_config  # noqa: E402
from fallpre.decision_contracts import decision_view_from_result  # noqa: E402
from fallpre.decision_fixtures import synthetic_evidence  # noqa: E402
from fallpre.decision_provider import build_decision_provider  # noqa: E402


def _case(evidence, provider, *, provider_name: str) -> dict[str, object]:
    result = provider.assess(evidence, deadline=10**12)
    view = decision_view_from_result(result)
    return {
        "simulated": True,
        "model_called": False,
        "scenario_provider": provider_name,
        "input": evidence.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "decision_view": view.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("off", "mock", "aliyun"), default="mock")
    parser.add_argument("--output", type=Path, default=Path("runs/decision-stage01-demo"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        print("output directory already exists; choose a new path", file=sys.stderr)
        return 2

    config = load_decision_config() if args.provider == "aliyun" else None
    try:
        selected_provider = build_decision_provider(args.provider, config=config)
    except DecisionConfigError as error:
        print(f"configuration_error: {error}", file=sys.stderr)
        return 2

    off_provider = build_decision_provider("off")
    scenarios = {
        "normal_probability": _case(
            synthetic_evidence(sufficient=True, suffix="normal"),
            selected_provider,
            provider_name=args.provider,
        ),
        "insufficient_evidence": _case(
            synthetic_evidence(sufficient=False, suffix="insufficient"),
            selected_provider,
            provider_name=args.provider,
        ),
        "provider_off": _case(
            synthetic_evidence(sufficient=True, suffix="off"),
            off_provider,
            provider_name="off",
        ),
    }

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("output directory already exists; choose a new path", file=sys.stderr)
        return 2
    for name, payload in scenarios.items():
        (output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(f"output={output}")
    for name, payload in scenarios.items():
        result = payload["result"]
        print(f"{name}: status={result['status']} provider={result['provider']} kind={result['probability_kind']}")
    print("simulated=true model_called=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
