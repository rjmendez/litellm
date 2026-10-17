"""
LociPolicyHook — policy supervision and enforcement hook for LiteLLM proxy.

Implements DRPG discovery/refinement cycle and EVO-RAG 7D reward semantics to
guide behavior and enforce policy constraints. Grounded in project references
(6003c71b, b5e01c4e, b64ebbf6) from hermes_memory and dama_gotchi_code.

Policy enforcement flow:
  1. Extract policy constraints from ENV (LOCI_POLICY_ENABLE, _MAX_CHARS, _ALLOWED_MODELS)
  2. Track 7D reward semantics: r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step, r_cor
  3. Apply DRPG discovery/refinement: detect policy drift, suggest refinements
  4. Surface policy violations as metadata, optional blocking
  5. Update EVO-RAG heuristics based on success/failure signals

Register in litellm config.yaml:
    litellm_settings:
      callbacks:
        - litellm.integrations.loci.policy_hook.loci_policy_hook
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.policy")

# ── Policy ENV helpers ─────────────────────────────────────────────────────────────

def _get_policy_config() -> dict[str, Any]:
    """Dynamically read policy configuration from ENV for thread-safety and reload support."""
    return {
        "enabled": os.environ.get("LOCI_POLICY_ENABLE", "1").lower() in ("1", "true", "yes"),
        "enabled_int": 1 if os.environ.get("LOCI_POLICY_ENABLE", "1").lower() in ("1", "true", "yes") else 0,
        "max_chars": int(os.environ.get("LOCI_POLICY_MAX_CHARS", "0")),
        "allowed_models": [m.strip() for m in os.environ.get("LOCI_POLICY_ALLOWED_MODELS", "").split(",") if m.strip()],
        "block_violations": os.environ.get("LOCI_POLICY_BLOCK_VIOLATIONS", "0").lower() in ("1", "true", "yes"),
        "reward_horizon_hours": int(os.environ.get("LOCI_REWARD_HORIZON_HOURS", "24")),
        "drpg_interval": int(os.environ.get("LOCI_DRPG_REFINEMENT_INTERVAL", "100")),
    }

# Legacy constants for backward compatibility (better to use _get_policy_config() directly)
POLICY_ENABLED = bool(_get_policy_config()["enabled_int"])
POLICY_MAX_CHARS = _get_policy_config()["max_chars"]
POLICY_ALLOWED_MODELS = _get_policy_config()["allowed_models"]
POLICY_BLOCK_VIOLATIONS = _get_policy_config()["block_violations"]
REWARD_HORIZON_HOURS = _get_policy_config()["reward_horizon_hours"]
DRPG_REFINEMENT_INTERVAL = _get_policy_config()["drpg_interval"]


class PolicyViolationType(Enum):
    """Types of policy violations detected."""
    MODEL_NOT_ALLOWED = "model_not_allowed"
    MAX_CHARS_EXCEEDED = "max_chars_exceeded"
    RESERVED_COLLECTION = "reserved_collection_access"
    MALFORMED_INTENT = "malformed_intent"


@dataclass
class EvoRagRewardTensor:
    """7D reward semantics from EVO-RAG: r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step, r_cor"""
    
    # r_ret: retrieval quality (Qdrant recall success)
    r_ret: float = 0.0
    
    # r_rem: relevance to memory (evidence grounding score)
    r_rem: float = 0.0
    
    # r_ovp: overfitting prevention (diversity penalty for repeated contexts)
    r_ovp: float = 0.0
    
    # r_bkp: background knowledge penalty (prefer retrieval over base model)
    r_bkp: float = 0.0
    
    # r_ref: refinement success (DRPG cycle improvement signal)
    r_ref: float = 0.0
    
    # r_step: step efficiency (latency penalty; prefer faster routes)
    r_step: float = 0.0
    
    # r_cor: correction signal (feedback on mistakes caught by policy)
    r_cor: float = 0.0
    
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
    
    def total(self) -> float:
        """Weighted sum of 7D rewards."""
        return (
            0.20 * self.r_ret +
            0.20 * self.r_rem +
            0.15 * self.r_ovp +
            0.15 * self.r_bkp +
            0.10 * self.r_ref +
            0.10 * self.r_step +
            0.10 * self.r_cor
        )


@dataclass
class PolicyViolation:
    """Record of a single policy violation."""
    violation_type: PolicyViolationType
    message: str
    model: Optional[str] = None
    completion_tokens: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "message": self.message,
            "model": self.model,
            "completion_tokens": self.completion_tokens,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
        }


@dataclass
class DRPGDiscoveryState:
    """DRPG discovery/refinement cycle state."""
    sample_count: int = 0
    last_refinement_time: float = field(default_factory=time.time)
    refinement_suggestions: List[str] = field(default_factory=list)
    recent_violations: Dict[str, int] = field(default_factory=dict)  # type -> count
    
    def should_refine(self) -> bool:
        """True if refinement interval passed."""
        return self.sample_count % DRPG_REFINEMENT_INTERVAL == 0


class PolicyState:
    """Thread-safe policy tracking state."""
    
    def __init__(self):
        self.lock = threading.RLock()
        self.violations: List[PolicyViolation] = []
        self.reward_history: List[EvoRagRewardTensor] = []
        self.drpg_state = DRPGDiscoveryState()
        self.model_access_count: Dict[str, int] = {}  # model -> count
        self.max_char_events: List[float] = []  # timestamps of violations
    
    def record_violation(self, violation: PolicyViolation) -> None:
        with self.lock:
            self.violations.append(violation)
            vtype = violation.violation_type.value
            self.drpg_state.recent_violations[vtype] = \
                self.drpg_state.recent_violations.get(vtype, 0) + 1
    
    def record_reward(self, reward: EvoRagRewardTensor) -> None:
        with self.lock:
            self.reward_history.append(reward)
            self.drpg_state.sample_count += 1
    
    def record_model_access(self, model: str) -> None:
        with self.lock:
            self.model_access_count[model] = self.model_access_count.get(model, 0) + 1
    
    def get_refinement_suggestions(self) -> List[str]:
        """DRPG: Generate suggestions based on recent pattern."""
        with self.lock:
            suggestions = []
            
            # If many violations, suggest tightening
            total_violations = sum(self.drpg_state.recent_violations.values())
            if total_violations > 10:
                suggestions.append(
                    f"High violation rate ({total_violations} in last {DRPG_REFINEMENT_INTERVAL} samples): "
                    "consider tightening policy constraints or adding model safeguards"
                )
            
            # If max_char violations dominate, suggest expanding limit
            max_char_count = self.drpg_state.recent_violations.get(PolicyViolationType.MAX_CHARS_EXCEEDED.value, 0)
            if max_char_count > 5 and POLICY_MAX_CHARS > 0:
                suggestions.append(
                    f"Max char limit hit {max_char_count} times: "
                    f"consider raising LOCI_POLICY_MAX_CHARS from {POLICY_MAX_CHARS}"
                )
            
            # If model blocked frequently, suggest adding to allowlist
            if self.model_access_count:
                blocked_attempts = self.drpg_state.recent_violations.get(
                    PolicyViolationType.MODEL_NOT_ALLOWED.value, 0
                )
                if blocked_attempts > 3:
                    suggestions.append(
                        f"Model restrictions blocked {blocked_attempts} requests: "
                        f"review LOCI_POLICY_ALLOWED_MODELS configuration"
                    )
            
            return suggestions
    
    def recent_reward_stats(self, window_hours: int = REWARD_HORIZON_HOURS) -> Dict[str, float]:
        """Get recent reward statistics for the given window."""
        with self.lock:
            now = time.time()
            cutoff = now - (window_hours * 3600)
            
            recent = [r for r in self.reward_history if r.timestamp >= cutoff]
            if not recent:
                return {}
            
            return {
                "count": len(recent),
                "avg_r_ret": sum(r.r_ret for r in recent) / len(recent),
                "avg_r_rem": sum(r.r_rem for r in recent) / len(recent),
                "avg_r_ovp": sum(r.r_ovp for r in recent) / len(recent),
                "avg_r_bkp": sum(r.r_bkp for r in recent) / len(recent),
                "avg_r_ref": sum(r.r_ref for r in recent) / len(recent),
                "avg_r_step": sum(r.r_step for r in recent) / len(recent),
                "avg_r_cor": sum(r.r_cor for r in recent) / len(recent),
                "avg_total": sum(r.total() for r in recent) / len(recent),
            }


# Global policy state (singleton)
_policy_state = PolicyState()


class LociPolicyHook(CustomLogger):
    """
    LiteLLM CustomLogger callback for policy supervision.
    
    Hooks into pre_call to enforce constraints, and async_post_call_success
    to record rewards and update EVO-RAG heuristics.
    """
    
    def __init__(self):
        self.state = _policy_state
    
    async def async_pre_call_hook(
            self,
            user_message: str | None,
            data,
            call_type: str,
            **kwargs: Any,
        ) -> Optional[tuple]:
            """
            Pre-call policy enforcement.

            Checks:
            - Model is in allowed list (if LOCI_POLICY_ALLOWED_MODELS set)
            - Request doesn't violate max char constraint

            Returns: (user_message, data, call_type) or None to block.
            """
            config = _get_policy_config()
        
            if not config["enabled"]:
                return None
        
            violations: List[PolicyViolation] = []
        
            # Check model whitelist
            model = data.get("model", "unknown")
            if config["allowed_models"] and model not in config["allowed_models"]:
                violation = PolicyViolation(
                    violation_type=PolicyViolationType.MODEL_NOT_ALLOWED,
                    message=f"Model '{model}' not in allowlist: {config['allowed_models']}",
                    model=model,
                    request_id=data.get("request_id"),
                )
                violations.append(violation)
                self.state.record_violation(violation)
                logger.warning(f"Policy violation: {violation.message}")
        
            # Check max char limit
            if config["max_chars"] > 0 and user_message:
                if len(user_message) > config["max_chars"]:
                    violation = PolicyViolation(
                        violation_type=PolicyViolationType.MAX_CHARS_EXCEEDED,
                        message=f"Request exceeds max chars: {len(user_message)} > {config['max_chars']}",
                        completion_tokens=len(user_message),
                        request_id=data.get("request_id"),
                    )
                    violations.append(violation)
                    self.state.record_violation(violation)
                    logger.warning(f"Policy violation: {violation.message}")
        
            # Block if requested and violations found
            if violations and config["block_violations"]:
                raise Exception(
                    f"Policy violation(s): {'; '.join([v.message for v in violations])}"
                )
        
            # Record model access
            self.state.record_model_access(model)
        
            return None  # Don't modify request
    
    async def async_post_call_success_hook(
        self,
        kwargs: dict,
        response: Any,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """
        Post-call reward tracking and EVO-RAG update.
        
        Called after successful completion. Records 7D reward tensor based on
        response metadata and tier_router signals.
        """
        if not POLICY_ENABLED:
            return
        
        # Extract signals from response metadata
        latency_s = (end_time or time.time()) - (start_time or time.time())
        
        # Build reward tensor from available signals
        reward = EvoRagRewardTensor(
            r_ret=self._extract_reward_signal(response, "grounding_score", 0.5),
            r_rem=self._extract_reward_signal(response, "memory_match_count", 0.6),
            r_ovp=1.0 - self._extract_diversity_penalty(response),
            r_bkp=self._extract_reward_signal(response, "grounding_used", 0.7),
            r_ref=self._extract_reward_signal(response, "refinement_delta", 0.5),
            r_step=max(0.0, 1.0 - (latency_s / 5.0)),  # Penalty for > 5s
            r_cor=self._extract_reward_signal(response, "correction_signal", 0.5),
        )
        
        self.state.record_reward(reward)
        
        # DRPG refinement check
        if self.state.drpg_state.should_refine():
            suggestions = self.state.get_refinement_suggestions()
            if suggestions:
                logger.info(f"DRPG refinement suggestions: {suggestions}")
                self.state.drpg_state.refinement_suggestions.extend(suggestions)
    
    def _extract_reward_signal(self, response: Any, key: str, default: float = 0.5) -> float:
        """Extract reward signal from response metadata."""
        try:
            if isinstance(response, dict):
                val = response.get(key, response.get("_metadata", {}).get(key))
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception:
            pass
        return default
    
    def _extract_diversity_penalty(self, response: Any) -> float:
        """Compute penalty based on diversity metrics."""
        try:
            if isinstance(response, dict):
                repeated_count = response.get("_metadata", {}).get("repeated_contexts", 0)
                return min(1.0, repeated_count * 0.2)
        except Exception:
            pass
        return 0.0


# Singleton instance for registration
loci_policy_hook = LociPolicyHook()


def get_policy_status() -> Dict[str, Any]:
    """
    Return policy status: violations, reward stats, DRPG suggestions.
    Useful for monitoring and debugging.
    """
    state = _policy_state
    reward_stats = state.recent_reward_stats()
    
    return {
        "enabled": POLICY_ENABLED,
        "max_chars_limit": POLICY_MAX_CHARS if POLICY_MAX_CHARS > 0 else None,
        "allowed_models": POLICY_ALLOWED_MODELS if POLICY_ALLOWED_MODELS else None,
        "block_violations": POLICY_BLOCK_VIOLATIONS,
        "total_violations": len(state.violations),
        "recent_violations": state.drpg_state.recent_violations,
        "model_access_count": state.model_access_count,
        "reward_stats": reward_stats,
        "drpg_suggestions": state.drpg_state.refinement_suggestions[-5:],  # Last 5
        "sample_count": state.drpg_state.sample_count,
    }


def get_violations_since(timestamp_s: float) -> List[Dict[str, Any]]:
    """Get all violations since given timestamp."""
    state = _policy_state
    return [
        v.to_dict()
        for v in state.violations
        if v.timestamp >= timestamp_s
    ]
