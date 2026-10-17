"""
Example: trajectory persistence with LociPolicyHook.

Shows how reward trajectories are written to JSONL after each LLM call,
structured for the DRPG loop to consume via mcp__loci__policy_evolve.

Run with:
    LOCI_POLICY_ENABLE=1 python examples/trajectory_example.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loci_litellm.policy_hook import LociPolicyHook


def _fake_response(content: str, prompt_tokens: int = 250) -> object:
    msg = Mock()
    msg.content = content
    choice = Mock()
    choice.message = msg
    usage = Mock()
    usage.prompt_tokens = prompt_tokens
    response = Mock()
    response.choices = [choice]
    response.usage = usage
    return response


async def main() -> None:
    trajectory_dir = Path("./trajectory_output")
    trajectory_dir.mkdir(exist_ok=True)

    os.environ["LOCI_POLICY_ENABLE"] = "1"
    os.environ["LOCI_TRAJECTORY_DIR"] = str(trajectory_dir)

    hook = LociPolicyHook()

    policy_id = "example-policy"
    kwargs = {"model": "gpt-4", "metadata": {"policy_id": policy_id}}

    print(f"Writing trajectories to: {trajectory_dir.resolve()}")

    await hook.async_log_success_event(
        kwargs,
        _fake_response("The capital of France is Paris.", prompt_tokens=120),
        start_time=0.0,
        end_time=0.45,
    )
    print("Logged success event (r_cor=1.0)")

    await hook.async_log_failure_event(
        kwargs,
        _fake_response("", prompt_tokens=80),
        start_time=0.0,
        end_time=1.8,
    )
    print("Logged failure event (r_cor=0.0)")

    trajectory_file = hook.get_trajectory_path(policy_id)
    steps = [json.loads(line) for line in trajectory_file.read_text(encoding="utf-8").strip().split("\n")]

    print(f"\nTrajectory file: {trajectory_file}")
    print(f"Steps recorded: {len(steps)}\n")

    for i, step in enumerate(steps):
        reward = step["reward"]
        print(f"Step {i + 1}:")
        print(f"  model={step['model']}  prompt_tokens={step['prompt_tokens']}  response_time_ms={step['response_time_ms']:.1f}")
        print(f"  r_ret={reward['r_ret']:.2f}  r_ovp={reward['r_ovp']:.3f}  r_step={reward['r_step']:.4f}  r_cor={reward['r_cor']:.2f}")

    trajectory_file.unlink()
    trajectory_dir.rmdir()


if __name__ == "__main__":
    asyncio.run(main())
