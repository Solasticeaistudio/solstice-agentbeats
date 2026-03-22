"""
Mock Green Agent — simulates AgentBeats evaluation flow.

Starts the Finance and BizProcess A2A servers, then sends tasks
exactly like a green agent evaluator would:
1. Discover agent via /.well-known/agent.json
2. Send tasks via JSON-RPC message/send
3. Validate response format and artifacts

Usage:
    python tests/test_a2a_live.py
    python tests/test_a2a_live.py --track finance
    python tests/test_a2a_live.py --track bizprocess
    python tests/test_a2a_live.py --track both
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import os

import httpx

FINANCE_PORT = 9009
BIZPROCESS_PORT = 9019
HOST = "127.0.0.1"

# ── Test scenarios that mirror green agent evaluation ─────────────────

FINANCE_TASKS = [
    {
        "name": "NPV Calculation",
        "message": (
            "Calculate the Net Present Value of an investment with initial cost "
            "$50,000 and expected cash flows of $12,000, $15,000, $18,000, $20,000, "
            "and $25,000 over the next 5 years. Use a discount rate of 10%."
        ),
    },
    {
        "name": "Portfolio Risk Assessment",
        "message": (
            "Analyze the risk profile of a portfolio containing AAPL (40%), "
            "MSFT (30%), and GOOGL (30%). Provide Value at Risk, Sharpe ratio, "
            "and maximum drawdown estimates."
        ),
    },
    {
        "name": "Multi-Step Financial Reasoning",
        "message": (
            "A company is considering two expansion options:\n"
            "Option A: $2M investment, projected 15% annual return, 3-year payback\n"
            "Option B: $5M investment, projected 22% annual return, 5-year payback\n"
            "The company has $3M in cash reserves and can borrow at 7% interest.\n"
            "Which option maximizes shareholder value? Show your reasoning."
        ),
    },
]

BIZPROCESS_TASKS = [
    {
        "name": "Customer Refund Workflow",
        "message": (
            "A customer (ID: C-4521) has requested a refund for order #ORD-98765 "
            "totaling $1,247.50. The order was delivered 18 days ago. Our refund "
            "policy allows returns within 30 days. The customer reports the product "
            "arrived damaged. Process this refund request following standard procedures."
        ),
    },
    {
        "name": "Vendor Evaluation",
        "message": (
            "Evaluate the following vendor proposal for our cloud infrastructure:\n"
            "- Vendor: CloudScale Inc.\n"
            "- Contract: 3-year commitment, $45,000/month\n"
            "- SLA: 99.95% uptime guarantee\n"
            "- Data centers: US-East, US-West, EU-West\n"
            "- Current provider: $52,000/month, 99.9% SLA, same regions\n"
            "Provide a recommendation with risk analysis."
        ),
    },
    {
        "name": "Strategic Business Decision",
        "message": (
            "Our SaaS company (ARR: $12M, 200 employees) has received an acquisition "
            "offer of $80M from a larger competitor. We're currently growing at 45% YoY "
            "and have 18 months of runway at current burn rate. The board needs a "
            "recommendation. Analyze the strategic options."
        ),
    },
]


def build_jsonrpc_request(text: str, task_id: str = None) -> dict:
    """Build an A2A JSON-RPC message/send request."""
    import uuid

    msg_id = uuid.uuid4().hex
    params = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": msg_id,
        }
    }
    if task_id:
        params["message"]["taskId"] = task_id

    return {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": uuid.uuid4().hex,
        "params": params,
    }


async def discover_agent(base_url: str) -> dict:
    """Step 1: Discover agent via Agent Card."""
    async with httpx.AsyncClient(timeout=10) as client:
        # Try the standard A2A discovery paths
        for path in ["/.well-known/agent.json", "/agent"]:
            try:
                resp = await client.get(f"{base_url}{path}")
                if resp.status_code == 200:
                    card = resp.json()
                    return card
            except Exception:
                continue
    return None


async def send_task(base_url: str, message: str, timeout: float = 300) -> dict:
    """Step 2: Send a task via JSON-RPC message/send."""
    payload = build_jsonrpc_request(message)
    # Use explicit timeout config — read timeout is the critical one
    # (server may take minutes to process complex multi-step reasoning)
    timeout_config = httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0)
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        resp = await client.post(f"{base_url}/", json=payload)
        return resp.json()


def print_separator(char="-", width=70):
    print(char * width)


def print_header(text: str):
    print_separator("=")
    print(f"  {text}")
    print_separator("=")


def print_result(name: str, response: dict, elapsed: float):
    """Pretty-print a task result like a green agent scorer would see it."""
    print(f"\n  Task: {name}")
    print(f"  Time: {elapsed:.1f}s")
    print_separator("─")

    result = response.get("result", {})
    error = response.get("error")

    if error:
        print(f"  ERROR: {json.dumps(error, indent=2)}")
        return False

    # Extract task status
    status = result.get("status", {})
    state = status.get("state", "unknown") if isinstance(status, dict) else str(status)
    print(f"  Status: {state}")

    # Extract artifacts
    artifacts = result.get("artifacts", [])
    if artifacts:
        print(f"  Artifacts: {len(artifacts)}")
        for i, art in enumerate(artifacts):
            art_name = art.get("name", f"artifact_{i}")
            parts = art.get("parts", [])
            print(f"    [{art_name}]")
            for part in parts:
                if "text" in part:
                    text = part["text"]
                    # Truncate long responses for readability
                    if len(text) > 500:
                        print(f"      {text[:500]}...")
                        print(f"      ... ({len(text)} chars total)")
                    else:
                        print(f"      {text}")
                elif "data" in part:
                    data = part["data"]
                    print(f"      [JSON] {json.dumps(data, indent=2, default=str)[:500]}")
    else:
        # Check for message in status
        msg = status.get("message", {}) if isinstance(status, dict) else {}
        if msg:
            msg_parts = msg.get("parts", [])
            for p in msg_parts:
                if "text" in p:
                    print(f"  Message: {p['text'][:300]}")

    # Validate response structure
    checks = {
        "has_result": result is not None,
        "has_status": "status" in result,
        "has_artifacts": len(artifacts) > 0,
        "state_completed": state in ("completed", "working"),
    }
    all_pass = all(checks.values())
    print(f"\n  Validation:")
    for check, passed in checks.items():
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {check}")

    return all_pass


async def run_track(
    track_name: str, port: int, tasks: list, server_entry: str
):
    """Run a full green-agent-style evaluation against one track."""
    base_url = f"http://{HOST}:{port}"

    print_header(f"GREEN AGENT EVALUATION: {track_name.upper()}")

    # Start the server
    print(f"\n  Starting {track_name} agent on port {port}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(
        os.path.dirname(__file__), "..", "src"
    ) + os.pathsep + env.get("PYTHONPATH", "")

    # Use DEVNULL to prevent pipe buffer deadlock — server may produce
    # lots of debug output that fills the pipe buffer and blocks.
    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "src", server_entry),
            "--host", HOST,
            "--port", str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    ready = False
    for _ in range(30):
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{base_url}/")
                if resp.status_code in (200, 404, 405):
                    ready = True
                    break
        except Exception:
            pass
        await asyncio.sleep(1)

    if not ready:
        print("  FAILED: Server did not start within 30s")
        proc.terminate()
        return False

    print(f"  Server ready at {base_url}")

    # Step 1: Agent Discovery
    print(f"\n  [Discovery] Fetching Agent Card...")
    card = await discover_agent(base_url)
    if card:
        print(f"    Name: {card.get('name', 'unknown')}")
        print(f"    Version: {card.get('version', 'unknown')}")
        skills = card.get("skills", [])
        print(f"    Skills: {len(skills)}")
        for s in skills:
            print(f"      - {s.get('name', s.get('id', '?'))}")
    else:
        print("    WARNING: Could not fetch agent card (non-fatal)")

    # Step 2: Run tasks
    results = []
    for task in tasks:
        print_separator()
        print(f"  Sending: {task['name']}...")
        t0 = time.time()
        try:
            response = await send_task(base_url, task["message"], timeout=300)
            elapsed = time.time() - t0
            passed = print_result(task["name"], response, elapsed)
            results.append({"name": task["name"], "passed": passed, "time": elapsed})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {e}")
            results.append({"name": task["name"], "passed": False, "time": elapsed})

    # Summary
    print_separator("═")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_time = sum(r["time"] for r in results) / total if total else 0
    print(f"  {track_name.upper()} RESULTS: {passed}/{total} passed, avg {avg_time:.1f}s/task")
    print_separator("═")

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    return passed == total


async def main():
    parser = argparse.ArgumentParser(description="Mock Green Agent Evaluator")
    parser.add_argument(
        "--track",
        choices=["finance", "bizprocess", "both"],
        default="both",
        help="Which track to evaluate",
    )
    args = parser.parse_args()

    print_header("AGENTBEATS MOCK GREEN AGENT EVALUATOR")
    print("  Simulating Phase 2 evaluation flow")
    print(f"  Track: {args.track}")
    print()

    results = {}

    if args.track in ("finance", "both"):
        results["finance"] = await run_track(
            "Finance Agent", FINANCE_PORT, FINANCE_TASKS, "server_finance.py"
        )

    if args.track in ("bizprocess", "both"):
        results["bizprocess"] = await run_track(
            "Business Process Agent",
            BIZPROCESS_PORT,
            BIZPROCESS_TASKS,
            "server_bizprocess.py",
        )

    # Final summary
    print()
    print_header("FINAL SUMMARY")
    for track, passed in results.items():
        status = "ALL PASSED" if passed else "SOME FAILED"
        print(f"  {track}: {status}")
    print()

    all_passed = all(results.values())
    if all_passed:
        print("  Ready for AgentBeats submission.")
    else:
        print("  Fix failures before submitting.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
