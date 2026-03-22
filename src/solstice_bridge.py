"""
Solstice-EIM Service Bridge
============================

Lazy-loads Solstice-EIM services and provides async wrappers for use in
A2A agents. Falls back to litellm when Solstice imports are unavailable
(e.g., in a minimal Docker container without the full backend).

Usage:
    from solstice_bridge import get_bridge
    bridge = get_bridge()
    text, meta = await bridge.generate("Analyze AAPL", agent_name="plutus")
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("agentbeats.bridge")

# ── Solstice-EIM path resolution ──────────────────────────────────────
SOLSTICE_ROOT = os.getenv(
    "SOLSTICE_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

if os.path.isdir(SOLSTICE_ROOT) and SOLSTICE_ROOT not in sys.path:
    sys.path.insert(0, SOLSTICE_ROOT)
    log.info(f"Added Solstice-EIM to sys.path: {SOLSTICE_ROOT}")


class SolsticeBridge:
    """
    Lazy-loading bridge to Solstice-EIM services.

    Services are loaded on first property access. If any import fails,
    that service is silently disabled and the bridge falls back to litellm
    for LLM generation.
    """

    def __init__(self):
        self._llm = None
        self._retrieval = None
        self._orchestrator = None
        self._plutus = None
        self._initialized = False

    # ── Lazy initialization ───────────────────────────────────────────

    def _init_services(self):
        if self._initialized:
            return
        self._initialized = True

        # LLM Service
        try:
            from services.llm_service import LLMService

            self._llm = LLMService(cache=None)
            log.info("LLMService loaded")
        except Exception as e:
            log.warning(f"LLMService unavailable ({e.__class__.__name__}): {e}")

        # Retrieval Service (Pinecone RAG)
        try:
            from services.retrieval_service import RetrievalService

            self._retrieval = RetrievalService()
            log.info("RetrievalService loaded")
        except Exception as e:
            log.warning(f"RetrievalService unavailable ({e.__class__.__name__}): {e}")

        # Agent Orchestrator (multi-agent delegation)
        try:
            from services.agent_orchestrator import AgentOrchestrator

            self._orchestrator = AgentOrchestrator(
                llm_service=self._llm,
                retrieval_service=self._retrieval,
            )
            log.info("AgentOrchestrator loaded")
        except Exception as e:
            log.warning(f"AgentOrchestrator unavailable ({e.__class__.__name__}): {e}")

        # Plutus (Finance agent)
        try:
            from agents.plutus import PlutusAgent

            self._plutus = PlutusAgent(llm=self._llm, retrieval=self._retrieval)
            log.info("PlutusAgent loaded")
        except Exception as e:
            log.warning(f"PlutusAgent unavailable ({e.__class__.__name__}): {e}")

    @property
    def llm(self):
        self._init_services()
        return self._llm

    @property
    def retrieval(self):
        self._init_services()
        return self._retrieval

    @property
    def orchestrator(self):
        self._init_services()
        return self._orchestrator

    @property
    def plutus(self):
        self._init_services()
        return self._plutus

    # ── Async wrappers ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        agent_name: str = "sol",
        history: Optional[List[Dict]] = None,
        use_nexus: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Async wrapper around LLMService.generate() with litellm fallback."""
        if self.llm:
            try:
                return await asyncio.to_thread(
                    self.llm.generate,
                    prompt=prompt,
                    agent_name=agent_name,
                    history=history or [],
                    use_nexus=use_nexus,
                )
            except Exception as e:
                log.warning(f"LLMService.generate failed, falling back to litellm: {e}")

        return await self._litellm_generate(prompt, history)

    async def retrieve(
        self,
        query: str,
        agent: str = "sol",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Async wrapper around RetrievalService.retrieve()."""
        if self.retrieval:
            try:
                results = await asyncio.to_thread(
                    self.retrieval.retrieve,
                    query=query,
                    agent=agent,
                    top_k=top_k,
                )
                return results if isinstance(results, list) else []
            except Exception as e:
                log.warning(f"RetrievalService.retrieve failed: {e}")
        return []

    async def delegate(
        self,
        query: str,
        agent_name: str,
        session_id: str = "agentbeats",
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper around AgentOrchestrator.delegate()."""
        if self.orchestrator:
            try:
                result = await asyncio.to_thread(
                    self.orchestrator.delegate,
                    query=query,
                    agent_name=agent_name,
                    session_id=session_id,
                    history=history,
                )
                return {
                    "text": result.response_text,
                    "agent": result.agent_name,
                    "confidence": result.confidence,
                    "success": result.success,
                    "time_ms": result.delegation_time_ms,
                }
            except Exception as e:
                log.warning(f"Delegation to {agent_name} failed: {e}")

        # Fallback: direct LLM call
        text, meta = await self.generate(prompt=query, agent_name=agent_name)
        return {
            "text": text,
            "agent": "litellm_fallback",
            "confidence": 0.5,
            "success": True,
        }

    async def auto_delegate(
        self,
        query: str,
        session_id: str = "agentbeats",
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper around AgentOrchestrator.auto_delegate()."""
        if self.orchestrator:
            try:
                result = await asyncio.to_thread(
                    self.orchestrator.auto_delegate,
                    query=query,
                    session_id=session_id,
                    history=history,
                )
                if result:
                    return {
                        "text": result.response_text,
                        "agent": result.agent_name,
                        "confidence": result.confidence,
                        "success": result.success,
                        "time_ms": result.delegation_time_ms,
                    }
            except Exception as e:
                log.warning(f"Auto-delegation failed: {e}")

        # Fallback: direct LLM call
        text, meta = await self.generate(prompt=query, agent_name="sol")
        return {
            "text": text,
            "agent": "litellm_fallback",
            "confidence": 0.5,
            "success": True,
        }

    # ── Direct Plutus math bridges (real computation, zero LLM cost) ──────

    async def plutus_dcf(
        self,
        free_cash_flows: List[float],
        terminal_growth_rate: float = 0.03,
        discount_rate: float = 0.10,
        shares_outstanding: int = 1,
    ) -> Dict[str, Any]:
        """Discounted Cash Flow valuation via Plutus — real math, no LLM."""
        if self.plutus:
            try:
                return await asyncio.to_thread(
                    self.plutus.calculate_dcf_valuation,
                    free_cash_flows,
                    terminal_growth_rate,
                    discount_rate,
                    shares_outstanding,
                )
            except Exception as e:
                log.warning(f"Plutus DCF failed, using fallback: {e}")
        # Pure Python fallback (Gordon Growth model)
        pv_fcfs = [cf / (1 + discount_rate) ** i for i, cf in enumerate(free_cash_flows, 1)]
        last_fcf = free_cash_flows[-1]
        terminal_value = last_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / (1 + discount_rate) ** len(free_cash_flows)
        ev = sum(pv_fcfs) + pv_terminal
        return {
            "enterprise_value": round(ev, 2),
            "pv_projected_fcfs": round(sum(pv_fcfs), 2),
            "pv_terminal_value": round(pv_terminal, 2),
            "value_per_share": round(ev / shares_outstanding, 2) if shares_outstanding else 0,
            "terminal_growth_rate": terminal_growth_rate,
            "discount_rate": discount_rate,
        }

    async def plutus_monte_carlo(
        self,
        initial_price: float,
        expected_return: float = 0.08,
        volatility: float = 0.20,
        time_horizon_days: int = 252,
        num_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """Geometric Brownian Motion simulation via Plutus — real numpy, no LLM."""
        if self.plutus:
            try:
                return await asyncio.to_thread(
                    self.plutus.monte_carlo_simulation,
                    initial_price,
                    expected_return,
                    volatility,
                    time_horizon_days,
                    num_simulations,
                )
            except Exception as e:
                log.warning(f"Plutus Monte Carlo failed, using fallback: {e}")
        # Pure numpy fallback (vectorized GBM)
        import numpy as np
        dt = 1 / 252
        daily_drift = (expected_return - 0.5 * volatility ** 2) * dt
        daily_vol = volatility * (dt ** 0.5)
        shocks = np.random.normal(0, 1, (num_simulations, time_horizon_days))
        log_returns = daily_drift + daily_vol * shocks
        final_prices = initial_price * np.exp(np.sum(log_returns, axis=1))
        returns_pct = (final_prices - initial_price) / initial_price
        return {
            "mean_price": round(float(np.mean(final_prices)), 4),
            "median_price": round(float(np.median(final_prices)), 4),
            "std_dev": round(float(np.std(final_prices)), 4),
            "percentile_5": round(float(np.percentile(final_prices, 5)), 4),
            "percentile_25": round(float(np.percentile(final_prices, 25)), 4),
            "percentile_75": round(float(np.percentile(final_prices, 75)), 4),
            "percentile_95": round(float(np.percentile(final_prices, 95)), 4),
            "var_95": round(float(np.percentile(returns_pct, 5)) * 100, 2),
            "var_99": round(float(np.percentile(returns_pct, 1)) * 100, 2),
            "probability_above_initial": round(float(np.mean(final_prices > initial_price)), 4),
            "expected_return_pct": round(float(np.mean(returns_pct)) * 100, 2),
            "num_simulations": num_simulations,
            "time_horizon_days": time_horizon_days,
        }

    async def plutus_npv_irr(
        self,
        cash_flows: List[float],
        discount_rate: float = 0.10,
    ) -> Dict[str, Any]:
        """NPV + IRR via Plutus scipy solver — real math, no LLM."""
        if self.plutus:
            try:
                return await asyncio.to_thread(
                    self.plutus.calculate_npv_irr,
                    cash_flows,
                    discount_rate,
                )
            except Exception as e:
                log.warning(f"Plutus NPV/IRR failed, using fallback: {e}")
        # Pure Python fallback
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        irr = None
        try:
            from scipy.optimize import brentq, fsolve

            def npv_func(r: float) -> float:
                return sum(cf / (1 + r) ** i for i, cf in enumerate(cash_flows))

            try:
                irr = float(brentq(npv_func, -0.9999, 10.0))
            except ValueError:
                irr = float(fsolve(npv_func, 0.1)[0])
        except Exception:
            pass
        return {
            "npv": round(npv, 2),
            "irr": round(irr * 100, 2) if irr is not None else None,
            "discount_rate": discount_rate,
            "recommendation": "accept" if npv > 0 else "reject",
        }

    async def plutus_budget_variance(
        self,
        budgeted: Dict[str, float],
        actual: Dict[str, float],
    ) -> Dict[str, Any]:
        """Budget variance analysis via Plutus — pure math, no LLM."""
        if self.plutus:
            try:
                return await asyncio.to_thread(
                    self.plutus.analyze_budget_variance,
                    budgeted,
                    actual,
                )
            except Exception as e:
                log.warning(f"Plutus budget variance failed, using fallback: {e}")
        # Pure Python fallback
        categories = set(list(budgeted.keys()) + list(actual.keys()))
        variances: Dict[str, Any] = {}
        total_b, total_a = 0.0, 0.0
        for cat in sorted(categories):
            b = budgeted.get(cat, 0.0)
            a = actual.get(cat, 0.0)
            v = a - b
            variances[cat] = {
                "budgeted": round(b, 2),
                "actual": round(a, 2),
                "variance": round(v, 2),
                "variance_percent": round(v / b * 100, 2) if b != 0 else 0,
                "status": "over_budget" if v > 0 else "under_budget" if v < 0 else "on_budget",
            }
            total_b += b
            total_a += a
        return {
            "category_variances": variances,
            "total_budgeted": round(total_b, 2),
            "total_actual": round(total_a, 2),
            "total_variance": round(total_a - total_b, 2),
            "overall_variance_percent": round((total_a - total_b) / total_b * 100, 2) if total_b else 0,
        }

    # ── Plutus narrative query ─────────────────────────────────────────

    async def plutus_query(
        self,
        query: str,
        session_id: str = "agentbeats",
    ) -> Dict[str, Any]:
        """Route directly to PlutusAgent for financial analysis."""
        if self.plutus:
            try:
                result = await asyncio.to_thread(
                    self.plutus.process_query,
                    query=query,
                    session_id=session_id,
                )
                return result if isinstance(result, dict) else {"text": str(result)}
            except Exception as e:
                log.warning(f"PlutusAgent.process_query failed: {e}")

        # Fallback
        text, meta = await self.generate(prompt=query, agent_name="plutus")
        return {"text": text, "sources": [], "confidence": 0.5}

    # ── Litellm fallback ──────────────────────────────────────────────

    async def _litellm_generate(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Fallback LLM generation using litellm (supports OpenAI, Anthropic, etc.)."""
        import litellm

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        model = os.getenv("AGENTBEATS_LLM_MODEL", "openai/gpt-4.1")
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=model,
                messages=messages,
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            return text, {"model": model, "source": "litellm"}
        except Exception as e:
            log.error(f"litellm fallback failed: {e}")
            return f"Error: {e}", {"model": model, "source": "error"}


# ── Singleton ─────────────────────────────────────────────────────────

_bridge: Optional[SolsticeBridge] = None


def get_bridge() -> SolsticeBridge:
    """Get or create the singleton bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = SolsticeBridge()
    return _bridge
