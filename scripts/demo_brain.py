"""Run the synthetic Pre-CAB reasoning flow without a real API key."""
from __future__ import annotations

import json
from pathlib import Path

from pre_cab.brain import build_reasoning_system_prompt, self_critique_questions
from pre_cab.bootstrap import demo_memory
from pre_cab.pipeline import run_pre_cab
from pre_cab.schemas import Strictness


class DemoModel:
    model_name = "demo-no-network"

    def generate(self, *, system: str, user: str, temperature: float = 0.1, **_: object):
        from pre_cab.models import ModelResponse

        return ModelResponse(
            text=(
                "DEMO MODEL: reasoning contract loaded.\n"
                "For production, inject Groq GPT-OSS 120B through the model adapter.\n"
                f"Self-critique checks: {len(self_critique_questions())}"
            ),
            model=self.model_name,
            raw={"system_prompt_length": len(system), "user_payload_length": len(user), "temperature": temperature},
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "data" / "demo" / "normal_crs.json").read_text(encoding="utf-8"))
    memory = demo_memory()
    print(build_reasoning_system_prompt())
    for cr in cases:
        result = run_pre_cab(cr, [], strictness=Strictness.BALANCED, model=DemoModel(), memory=memory)
        print("\n", cr["Number"], result.final_decision.value)
        for finding in result.stage1.findings:
            if finding.severity.value != "INFO":
                print(f"- {finding.severity.value}: {finding.message}")


if __name__ == "__main__":
    main()
