#!/usr/bin/env python3
"""
Example: Using LociPolicyHook for policy supervision in LiteLLM.

Demonstrates:
1. Configuring policy constraints via ENV
2. Monitoring policy status and reward signals
3. Reacting to DRPG refinement suggestions
4. Querying violation history
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path

# Add fork to path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from loci_litellm.policy_hook import (
    get_policy_status,
    get_violations_since,
    EvoRagRewardTensor,
)


def setup_env():
    """Configure policy environment for this example."""
    print("📋 Setting up policy environment...")
    
    os.environ["LOCI_POLICY_ENABLE"] = "1"
    os.environ["LOCI_POLICY_ALLOWED_MODELS"] = "gpt-4,claude-3-opus"
    os.environ["LOCI_POLICY_MAX_CHARS"] = "10000"
    os.environ["LOCI_POLICY_BLOCK_VIOLATIONS"] = "0"  # Warn, don't block
    os.environ["LOCI_REWARD_HORIZON_HOURS"] = "24"
    os.environ["LOCI_DRPG_REFINEMENT_INTERVAL"] = "10"  # Low for demo
    
    print(f"  ✓ LOCI_POLICY_ENABLE={os.environ['LOCI_POLICY_ENABLE']}")
    print(f"  ✓ LOCI_POLICY_ALLOWED_MODELS={os.environ['LOCI_POLICY_ALLOWED_MODELS']}")
    print(f"  ✓ LOCI_POLICY_MAX_CHARS={os.environ['LOCI_POLICY_MAX_CHARS']}")
    print()


def simulate_hook_behavior():
    """Simulate hook processing without a full LiteLLM proxy."""
    print("🔄 Simulating policy hook behavior...")
    
    from loci_litellm.policy_hook import (
        LociPolicyHook,
        PolicyViolation,
        PolicyViolationType,
    )
    
    hook = LociPolicyHook()
    
    # Example 1: Valid request (allowed model)
    print("\n  [1] Valid request (gpt-4):")
    asyncio.run(hook.async_pre_call_hook(
        "What is the capital of France?",
        {"model": "gpt-4", "request_id": "req-1"},
        "chat/completions"
    ))
    print("    ✓ Allowed")
    
    # Example 2: Disallowed model
    print("\n  [2] Disallowed model (gpt-5):")
    asyncio.run(hook.async_pre_call_hook(
        "What is the capital of Germany?",
        {"model": "gpt-5", "request_id": "req-2"},
        "chat/completions"
    ))
    print("    ✗ Violation recorded")
    
    # Example 3: Message too long
    print("\n  [3] Message exceeds max chars:")
    long_msg = "x" * 15000
    asyncio.run(hook.async_pre_call_hook(
        long_msg,
        {"model": "gpt-4", "request_id": "req-3"},
        "chat/completions"
    ))
    print("    ✗ Violation recorded")
    
    # Example 4: Simulate reward tracking
    print("\n  [4] Simulating successful completion with rewards:")
    response = {
        "content": "The capital of France is Paris.",
        "_metadata": {
            "grounding_score": 0.92,
            "memory_match_count": 3,
            "grounding_used": True,
            "refinement_delta": 0.15,
            "correction_signal": 1.0,
        }
    }
    
    start = time.time() - 0.8  # 0.8s latency
    end = time.time()
    
    asyncio.run(hook.async_post_call_success_hook(
        kwargs={"model": "gpt-4"},
        response=response,
        start_time=start,
        end_time=end,
    ))
    print("    ✓ Reward recorded (7D tensor)")
    
    # Record a few more to trigger DRPG
    print("\n  [5] Simulating multiple completions for DRPG refinement...")
    for i in range(12):
        r = EvoRagRewardTensor(
            r_ret=0.85,
            r_rem=0.80,
            r_ovp=0.75,
            r_bkp=0.70,
            r_ref=0.60,
            r_step=0.95,
            r_cor=0.88,
        )
        hook.state.record_reward(r)
    print(f"    ✓ Recorded 12 rewards (sample_count = {hook.state.drpg_state.sample_count})")


def display_policy_status():
    """Display current policy status."""
    print("\n📊 Policy Status Report:")
    print("=" * 60)
    
    status = get_policy_status()
    
    print(f"\nConfiguration:")
    print(f"  Enabled: {status['enabled']}")
    if status['max_chars_limit']:
        print(f"  Max chars limit: {status['max_chars_limit']}")
    if status['allowed_models']:
        print(f"  Allowed models: {', '.join(status['allowed_models'])}")
    print(f"  Block violations: {status['block_violations']}")
    
    print(f"\nViolations:")
    print(f"  Total: {status['total_violations']}")
    if status['recent_violations']:
        for vtype, count in status['recent_violations'].items():
            print(f"    • {vtype}: {count}")
    else:
        print("    (None)")
    
    print(f"\nModel Access:")
    if status['model_access_count']:
        for model, count in status['model_access_count'].items():
            print(f"    • {model}: {count} requests")
    else:
        print("    (None)")
    
    print(f"\nReward Statistics (7D):")
    if status['reward_stats']:
        stats = status['reward_stats']
        print(f"  Samples: {stats.get('count', 0)}")
        print(f"  r_ret (retrieval):        {stats.get('avg_r_ret', 0):.3f}")
        print(f"  r_rem (relevance):        {stats.get('avg_r_rem', 0):.3f}")
        print(f"  r_ovp (overfit prevent):  {stats.get('avg_r_ovp', 0):.3f}")
        print(f"  r_bkp (background knowl): {stats.get('avg_r_bkp', 0):.3f}")
        print(f"  r_ref (refinement):       {stats.get('avg_r_ref', 0):.3f}")
        print(f"  r_step (efficiency):      {stats.get('avg_r_step', 0):.3f}")
        print(f"  r_cor (correction):       {stats.get('avg_r_cor', 0):.3f}")
        print(f"  ──────────────────────────────────────")
        print(f"  avg_total:                {stats.get('avg_total', 0):.3f}")
    else:
        print("  (No rewards yet)")
    
    print(f"\nDRPG Refinement Suggestions:")
    if status['drpg_suggestions']:
        for i, suggestion in enumerate(status['drpg_suggestions'], 1):
            print(f"  [{i}] {suggestion}")
    else:
        print("  (None)")
    
    print("\n" + "=" * 60)


def display_violations():
    """Display recent violations."""
    print("\n⚠️  Recent Violations:")
    print("=" * 60)
    
    now = time.time()
    since = now - 3600  # Last hour
    
    violations = get_violations_since(since)
    
    if not violations:
        print("(No violations in the last hour)")
    else:
        print(f"({len(violations)} violation(s) in the last hour)\n")
        
        for v in violations:
            ts = time.strftime("%H:%M:%S", time.localtime(v['timestamp']))
            vtype = v['type']
            msg = v['message']
            model = v.get('model')
            
            print(f"  [{ts}] {vtype}")
            print(f"         Message: {msg}")
            if model:
                print(f"         Model: {model}")
            if v.get('request_id'):
                print(f"         Request ID: {v['request_id']}")
            print()
    
    print("=" * 60)


def main():
    """Run the example."""
    print("\n" + "=" * 60)
    print("  LociPolicyHook Example")
    print("=" * 60)
    
    setup_env()
    simulate_hook_behavior()
    display_policy_status()
    display_violations()
    
    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("  1. Register hook in your litellm config.yaml")
    print("  2. Set ENV variables as shown above")
    print("  3. Start LiteLLM proxy and make requests")
    print("  4. Monitor with: from loci_litellm.policy_hook import get_policy_status")
    print("  5. Review DRPG suggestions and refine policy as needed")


if __name__ == "__main__":
    main()
