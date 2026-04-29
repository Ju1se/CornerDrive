"""
GLM Direct Policy Engine for FLPG Policy Agent.

This engine replaces the rule-based policy proposal with direct GLM control.
GLM analyzes telemetry and directly decides parameter values for the next round.

Key changes from llm_assistant.py:
- GLM directly outputs parameter values (not just explanations)
- No intermediate "proposal" stage - GLM's output IS the policy
- GLM operates within pre-defined bounds and step limits
- Removed advisory-only behavior - GLM is now the decision maker

IMPORTANT: theta_rare directionality
- L2判定: ΔL_corner ≤ theta_rare → RARITY
- theta_rare 是负值
- theta_rare 越负 (如 -0.05) → 条件越严格 → 更少梯度被判为 RARITY
- theta_rare 越接近 0 (如 -0.02) → 条件越宽松 → 更多梯度被判为 RARITY

Constraints:
- GLM can only modify: theta_tol, theta_rare, theta_drift, recheck_probability,
                      honest_reward_multiplier, slash_multiplier,
                      rarity_reward_multiplier, corner_weight
- GLM cannot directly judge if a gradient is RARITY/FRAUD (L2/L3 responsibility)
- GLM cannot write policy:current (only policy:next via round_close)
- GLM cannot modify thresholds mid-round (only at round boundary)
"""

import logging
import os
import json
import time
from typing import Optional
import httpx

from common.schemas import (
    Policy, RoundTelemetry, PolicyProposal,
    PolicyBounds, PolicyMaxStep
)
from common.utils.bounds import clamp_value, clamp_delta
from policy_agent.api.exceptions import GLMAPIException, GLMParseException
from policy_agent.engine.policy_modes import PolicyDecisionContext, build_policy_decision_context

logger = logging.getLogger(__name__)

_glm_backoff_until = 0.0


class GLMPolicyEngine:
    """
    GLM-powered direct policy engine.

    This engine:
    1. Analyzes telemetry to understand system state
    2. Calls GLM to get direct parameter recommendations
    3. Validates and clamps to bounds/step limits
    4. Returns a PolicyProposal with GLM's decisions

    Unlike the old llm_assistant.py, this engine is NOT advisory.
    GLM's parameter decisions ARE the policy (subject to safety constraints).
    """

    # Tunable parameters that GLM can control
    TUNABLE_PARAMS = [
        "theta_tol",
        "theta_rare",
        "theta_drift",
        "recheck_probability",
        "honest_reward_multiplier",
        "slash_multiplier",
        "rarity_reward_multiplier",
        "corner_weight",
    ]

    def __init__(self):
        """Initialize GLM policy engine from environment variables."""
        self.api_key = os.getenv("GLM_API_KEY")
        self.mode = os.getenv("GLM_MODE", "hybrid").lower()
        self.base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.model = os.getenv("GLM_MODEL", "glm-4.7")
        self.timeout = float(os.getenv("GLM_TIMEOUT", "45"))
        self.failure_backoff_seconds = float(os.getenv("GLM_FAILURE_BACKOFF_SECONDS", "300"))
        self.max_tokens = int(os.getenv("GLM_MAX_TOKENS", "1200"))
        self.retry_max_tokens = int(
            os.getenv("GLM_RETRY_MAX_TOKENS", str(max(self.max_tokens, 1800)))
        )
        self.thinking_mode = os.getenv("GLM_THINKING", "disabled").lower()
        if self.thinking_mode not in {"enabled", "disabled"}:
            logger.warning(
                "Unsupported GLM_THINKING=%s; falling back to disabled",
                self.thinking_mode,
            )
            self.thinking_mode = "disabled"

        # Policy bounds and step limits
        self.bounds = PolicyBounds()
        self.max_step = PolicyMaxStep()

        self.enabled = bool(self.api_key)

        if not self.enabled:
            logger.warning("GLM policy engine disabled: no API key provided")
        else:
            logger.info(f"GLM policy engine enabled (model: {self.model})")

    async def propose(
        self,
        current_policy: Policy,
        telemetry: RoundTelemetry
    ) -> PolicyProposal:
        """
        Generate next-round policy using GLM.

        Args:
            current_policy: Current frozen policy
            telemetry: Round telemetry data

        Returns:
            PolicyProposal with GLM's proposed policy
        """
        global _glm_backoff_until
        decision_context = build_policy_decision_context(telemetry)

        if self.mode == "rule_only":
            logger.info("GLM mode is rule_only; skipping remote call")
            return await self._fallback_proposal(current_policy, telemetry)

        if self.mode == "hybrid" and not decision_context.needs_glm_reasoning:
            logger.info(
                "GLM hybrid mode: deterministic scenario %s, using rule engine",
                list(decision_context.active_modes) or ["stable"],
            )
            return await self._fallback_proposal(current_policy, telemetry)

        if not self.enabled:
            return await self._fallback_proposal(current_policy, telemetry)

        if time.monotonic() < _glm_backoff_until:
            logger.info("GLM remote call is in backoff window; using fallback proposal")
            return await self._fallback_proposal(current_policy, telemetry)

        try:
            # Call GLM to get parameter recommendations
            glm_params = await self._call_glm_for_params(
                current_policy,
                telemetry,
                decision_context,
            )

            # Validate and clamp to bounds/step limits
            validated_params, validator_messages = self._validate_params(
                current_policy=current_policy,
                glm_params=glm_params,
                decision_context=decision_context,
            )

            # Create proposed policy
            proposed_policy = Policy(
                round_id=telemetry.round_id + 1,
                effective_from_round=telemetry.round_id + 1,
                **validated_params
            )

            # Create proposal
            proposal = PolicyProposal(
                current_policy=current_policy,
                proposed_policy=proposed_policy,
                reasons=glm_params.get("reasons", ["GLM-based policy adjustment"]),
                validator_passed=True,
                safety_guard_passed=False,  # Will be checked by safety guard
                blocked_reasons=[],
                validator_messages=validator_messages,
                round_id=telemetry.round_id + 1,
                source_engine="glm_policy_engine",
                llm_used=True
            )

            logger.info(
                f"GLM policy engine: proposed {len(proposal.get_diff())} parameter changes"
            )

            return proposal

        except Exception as e:
            logger.error(f"GLM policy engine failed: {e}")
            _glm_backoff_until = time.monotonic() + self.failure_backoff_seconds
            return await self._fallback_proposal(current_policy, telemetry)

    async def _fallback_proposal(
        self,
        current_policy: Policy,
        telemetry: RoundTelemetry
    ) -> PolicyProposal:
        """Fallback to the deterministic rule engine when GLM is unavailable."""
        from policy_agent.engine.rule_engine import RuleEngine

        logger.info("Using deterministic rule engine fallback for policy proposal")
        proposal = await RuleEngine().propose(current_policy, telemetry)
        proposal.llm_used = False
        proposal.source_engine = "rule_engine_fallback"
        return proposal

    async def _call_glm_for_params(
        self,
        current_policy: Policy,
        telemetry: RoundTelemetry,
        decision_context: PolicyDecisionContext,
    ) -> dict:
        """
        Call GLM API to get parameter recommendations.

        Returns dict with parameter values and reasons.

        Raises:
            GLMAPIException: If API call fails
            GLMParseException: If response parsing fails
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt = self._build_policy_prompt(current_policy, telemetry, decision_context)
        system_prompt = self._build_system_prompt()

        attempts = [
            (self.thinking_mode, self.max_tokens, "primary"),
        ]
        retry_tokens = max(self.max_tokens, self.retry_max_tokens)
        if self.thinking_mode != "disabled":
            attempts.append(("disabled", retry_tokens, "retry_with_thinking_disabled"))
        elif retry_tokens > self.max_tokens:
            attempts.append(("disabled", retry_tokens, "retry_with_larger_budget"))

        last_exception: Exception | None = None
        for thinking_mode, max_tokens, attempt_name in attempts:
            try:
                content, finish_reason, reasoning_present = await self._request_glm_completion(
                    url=url,
                    headers=headers,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    thinking_mode=thinking_mode,
                    max_tokens=max_tokens,
                )
                return self._parse_glm_response(content)
            except GLMParseException as exc:
                last_exception = exc
                if attempt_name != attempts[-1][2]:
                    logger.warning(
                        "GLM response was not valid JSON on %s; retrying with safer settings",
                        attempt_name,
                    )
                    continue
                raise
            except GLMAPIException as exc:
                last_exception = exc
                if attempt_name != attempts[-1][2]:
                    logger.warning(
                        "GLM response on %s was unusable (finish/content issue); retrying with safer settings",
                        attempt_name,
                    )
                    continue
                raise GLMAPIException(
                    f"{exc} (finish_reason={finish_reason if 'finish_reason' in locals() else 'unknown'}, "
                    f"thinking={thinking_mode}, reasoning_present={reasoning_present if 'reasoning_present' in locals() else 'unknown'})"
                ) from exc

        if last_exception is not None:
            raise last_exception

        raise GLMAPIException("GLM call finished without a usable response")

    async def _request_glm_completion(
        self,
        *,
        url: str,
        headers: dict[str, str],
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        max_tokens: int,
    ) -> tuple[str, str | None, bool]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "do_sample": False,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": thinking_mode,
                "clear_thinking": True,
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=min(self.timeout, 3.0),
                    read=self.timeout,
                    write=min(self.timeout, 3.0),
                    pool=min(self.timeout, 3.0),
                ),
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise GLMAPIException(
                f"GLM API request failed: {e.response.status_code}",
                status_code=e.response.status_code
            ) from e
        except httpx.ReadTimeout as e:
            raise GLMAPIException(
                f"GLM API read timeout after {self.timeout:.1f}s ({type(e).__name__})"
            ) from e
        except httpx.ConnectTimeout as e:
            raise GLMAPIException(
                f"GLM API connect timeout after {min(self.timeout, 3.0):.1f}s ({type(e).__name__})"
            ) from e
        except httpx.RequestError as e:
            raise GLMAPIException(
                f"GLM API request error ({type(e).__name__}): {repr(e)}"
            ) from e
        except KeyError as e:
            raise GLMAPIException(f"Invalid GLM response structure: missing {str(e)}") from e

        if "choices" not in data or not data["choices"]:
            raise GLMAPIException("Invalid GLM response: missing choices")

        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        content = self._coerce_message_text(message.get("content"))
        reasoning_content = self._coerce_message_text(message.get("reasoning_content"))

        if not content:
            if reasoning_content and self._looks_like_json_payload(reasoning_content):
                content = reasoning_content
            else:
                raise GLMAPIException(
                    f"Invalid GLM response: empty content (finish_reason={finish_reason})"
                )

        return content, finish_reason, bool(reasoning_content)

    @staticmethod
    def _coerce_message_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            fragments: list[str] = []
            for item in value:
                if isinstance(item, str):
                    fragments.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        fragments.append(str(text))
            return "\n".join(fragment for fragment in fragments if fragment).strip()
        return str(value).strip()

    @staticmethod
    def _looks_like_json_payload(value: str) -> bool:
        stripped = value.lstrip()
        return stripped.startswith("{") or stripped.startswith("```")

    def _build_system_prompt(self) -> str:
        """Build system prompt for GLM."""
        bounds_dict = self.bounds.model_dump()
        max_step_dict = self.max_step.model_dump()

        param_specs = []
        for param in self.TUNABLE_PARAMS:
            bounds = bounds_dict.get(param, (None, None))
            step = max_step_dict.get(param, None)

            spec = f"- {param}: range [{bounds[0]}, {bounds[1]}]"
            if step is not None:
                spec += f", max step ±{step}"
            param_specs.append(spec)

        return f"""You are a federated learning policy controller. Your task is to adjust parameters based on telemetry.

You must reason in four policy modes:
1. fraud-suppression mode
2. rarity-preservation mode
3. uncertainty-protection mode
4. drift-warning mode

MODE RULES:
- Fraud pressure high:
  - tighten theta_tol
  - moderately raise slash_multiplier
  - raise recheck_probability
  - do NOT raise honest_reward_multiplier
  - do NOT tighten theta_rare at the same time
- Rarity under-recall:
  - relax theta_rare
  - raise rarity_reward_multiplier
  - slightly raise recheck_probability
  - do NOT raise slash_multiplier
- False-slash risk high:
  - lower slash_multiplier
  - raise recheck_probability
  - prefer extra review over punishment
- Healthy but thin honest volume:
  - modestly raise honest_reward_multiplier
  - keep changes small and reversible
- Drift warning:
  - treat drift as warning, not verdict
  - raise recheck_probability first
  - use corner_loss_delta_avg and main_loss_delta_avg to classify drift
  - if corner loss improves and main loss does not worsen, interpret drift as novel rarity:
    keep theta_drift unchanged or only minimally tighten it, preserve rarity, avoid stronger slashing
  - if main loss worsens and corner benefit is weak, interpret drift as harmful shift:
    tighten theta_drift, raise recheck_probability, only raise slash if fraud pressure is also high
  - if drift evidence conflicts:
    favor recheck and minimal punishment

PARAMETERS YOU CONTROL:
{chr(10).join(param_specs)}

IMPORTANT DIRECTIONALITY:
- theta_tol is the fraud threshold. L2判定: ΔL_main > theta_tol → FRAUD
  - theta_tol lower = stricter fraud detection
  - theta_tol higher = more permissive fraud detection
  - tighten theta_tol = decrease it (for example 0.05 -> 0.048)
  - relax theta_tol = increase it (for example 0.05 -> 0.055)
- theta_rare is NEGATIVE. L2判定: ΔL_corner ≤ theta_rare → RARITY
  - theta_rare = -0.05 (more negative): stricter, fewer gradients become RARITY
  - theta_rare = -0.02 (closer to zero): more permissive, more gradients become RARITY
  - relax rarity threshold = increase theta_rare (for example -0.05 -> -0.03)
  - tighten rarity threshold = decrease theta_rare (for example -0.03 -> -0.05)
- theta_drift lower = stricter drift detection
- theta_drift higher = more permissive drift detection
- tighten theta_drift = decrease it (for example 0.05 -> 0.048)
- relax theta_drift = increase it (for example 0.05 -> 0.055)
- honest_reward_multiplier higher = stronger baseline honest participation incentive
- slash_multiplier higher = stronger punishment
- recheck_probability higher = more review before final settlement

RESPONSE FORMAT:
Respond ONLY with valid JSON in this exact format:
{{
    "theta_tol": <value>,
    "theta_rare": <value>,
    "theta_drift": <value>,
    "recheck_probability": <value>,
    "honest_reward_multiplier": <value>,
    "slash_multiplier": <value>,
    "rarity_reward_multiplier": <value>,
    "corner_weight": <value>,
    "reasons": ["<reason 1>", "<reason 2>"]
}}

GUIDELINES:
1. Make small, incremental changes and respect max step limits.
2. Protect honest rarity contributors whenever signals are uncertain.
3. Use stronger punishment only when fraud evidence is coherent, not merely novel.
4. If telemetry is healthy, keep changes minimal.
5. Never punish drift by default; classify it first.
6. If `audit_sample_size` is small, treat classification-rate signals as low confidence and keep changes especially conservative.
7. Only raise `honest_reward_multiplier` modestly when audit volume is thin but honesty remains healthy.
8. If fraud pressure or uncertainty is elevated, avoid increasing `honest_reward_multiplier`.

IMPORTANT: You control parameters ONLY. You do NOT:
- Judge individual gradients (that's L2/L3's job)
- Write policy:current directly
- Change parameters mid-round"""

    def _build_policy_prompt(
        self,
        current_policy: Policy,
        telemetry: RoundTelemetry,
        decision_context: PolicyDecisionContext,
    ) -> str:
        """Build user prompt for GLM."""
        current_params = {param: getattr(current_policy, param) for param in self.TUNABLE_PARAMS}

        telemetry_dict = {
            "round_id": telemetry.round_id,
            "fraud_rate": telemetry.fraud_rate,
            "rarity_rate": telemetry.rarity_rate,
            "honest_rate": telemetry.honest_rate,
            "noise_rate": telemetry.noise_rate,
            "audit_sample_size": telemetry.audit_sample_size,
            "main_accuracy": telemetry.main_accuracy,
            "corner_accuracy": telemetry.corner_accuracy,
            "main_loss_delta_avg": telemetry.main_loss_delta_avg,
            "corner_loss_delta_avg": telemetry.corner_loss_delta_avg,
            "false_slash_estimate": telemetry.false_slash_estimate,
            "rarity_retention_rate": telemetry.rarity_retention_rate,
            "golden_drift_score": telemetry.golden_drift_score,
            "suspect_queue_length": telemetry.suspect_queue_length,
            "hash_mismatch_rate": telemetry.hash_mismatch_rate,
            "recent_attack_pressure": telemetry.recent_attack_pressure,
        }
        decision_context_dict = decision_context.to_prompt_dict()

        return f"""Based on the following telemetry and current policy, recommend parameter values for the next round.

CURRENT POLICY (Round {current_policy.round_id}):
{json.dumps(current_params, indent=2)}

TELEMETRY:
{json.dumps(telemetry_dict, indent=2)}

DERIVED DECISION CONTEXT:
{json.dumps(decision_context_dict, indent=2)}

Prioritize these behaviors:
- If fraud pressure is high, enter fraud-suppression mode without punishing rarity at the same time.
- If rarity is under-recalled, enter rarity-preservation mode and do not increase slashing.
- If false-slash risk is high, prefer uncertainty-protection mode: lower punishment and increase recheck.
- If drift warning is active, classify it as novel rarity, harmful shift, or ambiguous before changing theta_drift or slashing.
- If `audit_sample_size` is small, lower your confidence in rate-based spikes and keep parameter moves closer to the current policy.

Respond with JSON containing the 8 parameter values and an array of reasons."""

    def _parse_glm_response(self, content: str) -> dict:
        """
        Parse GLM response and extract parameters.

        Handles various JSON formats GLM might return.

        Args:
            content: Raw response content from GLM

        Returns:
            Dict with parameter values and reasons

        Raises:
            GLMParseException: If JSON parsing fails or required params missing
        """
        # Try to extract JSON from response
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Find the code block content
            if len(lines) > 2:
                # Skip first line (```json or ```) and last line (```)
                content = "\n".join(lines[1:-1])
            else:
                # Malformed code block, try to extract anyway
                content = content.strip("`").strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GLM response as JSON: {e}")
            logger.debug(f"GLM response content: {content[:500]}")
            raise GLMParseException(
                f"GLM returned invalid JSON: {e}",
                raw_response=content
            )

        # Extract only the tunable parameters
        result = {}
        missing_params = []

        for param in self.TUNABLE_PARAMS:
            if param in data:
                try:
                    result[param] = float(data[param])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {param}: {data[param]}, using current")
                    missing_params.append(param)
            else:
                logger.warning(f"GLM response missing parameter: {param}")
                missing_params.append(param)

        # Ensure at least some parameters were returned
        if not result:
            raise GLMParseException(
                "GLM response contained no valid parameters",
                raw_response=content
            )

        # Add reasons if provided
        if "reasons" in data:
            reasons = data["reasons"]
            if isinstance(reasons, list):
                result["reasons"] = reasons
            elif isinstance(reasons, str):
                result["reasons"] = [reasons]
            else:
                result["reasons"] = ["GLM-based policy adjustment"]
        else:
            result["reasons"] = ["GLM-based policy adjustment"]

        # Log missing parameters
        if missing_params:
            result["reasons"].append(
                f"Note: GLM did not provide values for: {', '.join(missing_params)}"
            )

        return result

    def _validate_params(
        self,
        current_policy: Policy,
        glm_params: dict,
        decision_context: PolicyDecisionContext | None = None,
    ) -> tuple[dict, list[str]]:
        """
        Validate and clamp GLM parameters to bounds and step limits.

        Args:
            current_policy: Current policy for step limit calculation
            glm_params: Raw GLM output parameters

        Returns:
            Tuple of (validated_params, validator_messages)
        """
        validated = {}
        messages = []

        current_dict = current_policy.model_dump()
        bounds_dict = self.bounds.model_dump()
        max_step_dict = self.max_step.model_dump()

        for param in self.TUNABLE_PARAMS:
            if param not in glm_params:
                # Keep current value if GLM didn't specify
                validated[param] = current_dict.get(param)
                continue

            current_value = current_dict.get(param)
            proposed_value = glm_params[param]

            # Apply step limit first
            if param in max_step_dict:
                stepped = clamp_delta(
                    current=current_value,
                    proposed=proposed_value,
                    max_delta=max_step_dict[param]
                )
                if stepped != proposed_value:
                    messages.append(
                        f"{param}: {proposed_value} → {stepped} (max step: {max_step_dict[param]})"
                    )
                proposed_value = stepped

            # Apply bounds
            if param in bounds_dict:
                bounded = clamp_value(
                    value=proposed_value,
                    bounds=bounds_dict[param]
                )
                if bounded != proposed_value:
                    messages.append(
                        f"{param}: {proposed_value} → {bounded} (bounds: {bounds_dict[param]})"
                    )
                validated[param] = bounded
            else:
                validated[param] = proposed_value

        if decision_context is not None:
            validated, alignment_messages = self._apply_context_alignment(
                current_policy=current_policy,
                validated_params=validated,
                decision_context=decision_context,
            )
            messages.extend(alignment_messages)

        # Copy metadata fields from current policy, except the round-specific
        # values that will be set explicitly when constructing the next policy.
        excluded_fields = {"round_id", "effective_from_round", "created_at"}
        for key, value in current_dict.items():
            if key not in validated and key not in excluded_fields:
                validated[key] = value

        # Add reasons from GLM
        validated["reasons"] = glm_params.get("reasons", ["GLM-based policy adjustment"])

        return validated, messages

    def _apply_context_alignment(
        self,
        *,
        current_policy: Policy,
        validated_params: dict,
        decision_context: PolicyDecisionContext,
    ) -> tuple[dict, list[str]]:
        """
        Apply context-aware directional corrections to GLM outputs.

        This protects against semantic mistakes such as "tighten theta_tol"
        being translated into a larger numeric value, which would actually make
        fraud detection more permissive.
        """
        adjusted = dict(validated_params)
        messages: list[str] = []

        def _recommended_value(param: str, direction: str, step: float) -> float:
            current_value = getattr(current_policy, param)
            if direction == "decrease":
                target = current_value - step
            elif direction == "increase":
                target = current_value + step
            else:
                raise ValueError(f"Unsupported direction: {direction}")

            bounds = self.bounds.model_dump()[param]
            return clamp_value(target, bounds)

        def _correct_if_wrong_direction(
            param: str,
            direction: str,
            step: float,
            rationale: str,
        ) -> None:
            current_value = getattr(current_policy, param)
            proposed_value = adjusted.get(param, current_value)

            wrong_direction = (
                proposed_value > current_value
                if direction == "decrease"
                else proposed_value < current_value
            )
            if not wrong_direction:
                return

            corrected = _recommended_value(param, direction, step)
            adjusted[param] = corrected
            messages.append(
                f"Corrected {param} from {proposed_value} to {corrected} "
                f"to match {rationale}"
            )

        def _cap_at_current_if_higher(
            param: str,
            rationale: str,
        ) -> None:
            current_value = getattr(current_policy, param)
            proposed_value = adjusted.get(param, current_value)
            if proposed_value <= current_value:
                return
            adjusted[param] = current_value
            messages.append(
                f"Reset {param} from {proposed_value} to {current_value} "
                f"to avoid violating {rationale}"
            )

        if decision_context.fraud_pressure_high:
            _correct_if_wrong_direction(
                "theta_tol",
                "decrease",
                0.002,
                "fraud-suppression mode",
            )
            _cap_at_current_if_higher(
                "honest_reward_multiplier",
                "fraud-suppression mode",
            )

        if decision_context.rarity_under_recall:
            _correct_if_wrong_direction(
                "theta_rare",
                "increase",
                0.002,
                "beneficial-rarity preservation mode",
            )
            _cap_at_current_if_higher(
                "slash_multiplier",
                "beneficial-rarity preservation mode",
            )

        if decision_context.false_slash_risk_high:
            _correct_if_wrong_direction(
                "slash_multiplier",
                "decrease",
                0.10,
                "uncertainty-protection mode",
            )
            _correct_if_wrong_direction(
                "recheck_probability",
                "increase",
                0.05,
                "uncertainty-protection mode",
            )
            _cap_at_current_if_higher(
                "honest_reward_multiplier",
                "uncertainty-protection mode",
            )

        if decision_context.drift_warning:
            _correct_if_wrong_direction(
                "recheck_probability",
                "increase",
                0.05,
                "drift-warning mode",
            )
            if decision_context.novel_rarity_like_drift:
                _cap_at_current_if_higher(
                    "slash_multiplier",
                    "novel-beneficial-rarity-like drift handling",
                )
                current_drift = current_policy.theta_drift
                if adjusted.get("theta_drift", current_drift) < current_drift:
                    adjusted["theta_drift"] = current_drift
                    messages.append(
                        f"Reset theta_drift to {current_drift} to avoid punitive tightening "
                        "under novelty-like drift"
                    )
            elif decision_context.harmful_shift_like_drift:
                _correct_if_wrong_direction(
                    "theta_drift",
                    "decrease",
                    0.002,
                    "harmful-shift drift handling",
                )

        return adjusted, messages
