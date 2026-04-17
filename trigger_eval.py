"""
trigger_eval.py — Manual evaluation runner for VCAI.

Usage:
    python trigger_eval.py                         # Evaluate most recent session
    python trigger_eval.py <session_id>            # Evaluate specific session
    python trigger_eval.py <session_id> testing    # Use testing mode (default: training)

Output:
    reports/eval_{session_id[:8]}_{YYYY-MM-DD_HH-MM}.json
"""

import sys
import json
import requests
from datetime import datetime
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "http://localhost:8000/api"
USERNAME  = "ali@test.com"
PASSWORD  = "123456"
REPORTS_DIR = Path("reports")


# ── Auth ──────────────────────────────────────────────────────────────────────
def login() -> str:
    resp = requests.post(f"{BASE_URL}/auth/login", data={"username": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Session helpers ────────────────────────────────────────────────────────────
def get_most_recent_session(headers: dict) -> str:
    """Fetch the most recent completed session for the current user."""
    resp = requests.get(f"{BASE_URL}/sessions", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # Handle both list and paginated-dict response formats
    sessions = data if isinstance(data, list) else (data.get("sessions") or data.get("items") or [])
    if not sessions:
        raise SystemExit("No sessions found.")
    # Sort by started_at descending, pick first
    sessions = sorted(sessions, key=lambda s: s.get("started_at") or "", reverse=True)
    session_id = sessions[0]["id"]
    print(f"[AUTO] Most recent session: {session_id}")
    return session_id


def get_full_report(session_id: str, headers: dict) -> dict:
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/report", headers=headers)
    resp.raise_for_status()
    return resp.json()


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(report: dict) -> None:
    r = report.get("report", {})
    if not r:
        print("[SUMMARY] No report content yet.")
        return

    overall = report.get("overall_score") or r.get("scores", {}).get("overall_score", "?")
    mode    = r.get("mode", "?")
    passed  = report.get("passed")

    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Session  : {r.get('session_id', '?')}")
    print(f"  Persona  : {r.get('persona_name', '?')}")
    print(f"  Mode     : {mode}")
    print(f"  Score    : {overall}/100", end="")
    if passed is not None:
        print(f"  ({'PASSED' if passed else 'FAILED'})", end="")
    print()

    scores = r.get("scores", {})
    threshold = scores.get("pass_threshold", 75)
    print(f"  Threshold: {threshold}")
    print(f"  Status   : {scores.get('status', '?')}")

    # Skills table
    skills = scores.get("skills", [])
    if skills:
        print("\n  SKILL SCORES:")
        print(f"  {'Skill':<25} {'Score':>6}  {'Weight':>7}  {'Tested':>7}")
        print(f"  {'-'*25} {'-'*6}  {'-'*7}  {'-'*7}")
        for sk in sorted(skills, key=lambda s: s.get("score", 0), reverse=True):
            tested = "yes" if sk.get("was_tested") else "no"
            print(f"  {sk.get('skill_name','?'):<25} {sk.get('score',0):>6}  {sk.get('weight',0):>7.2f}  {tested:>7}")

    # Checkpoints
    checkpoints = r.get("checkpoints", [])
    if checkpoints:
        print("\n  CHECKPOINTS:")
        for cp in checkpoints:
            icon = "[OK]" if cp.get("achieved") else "[X] "
            print(f"    {icon} {cp.get('name', '?')}")

    # Quick stats
    qs = r.get("quick_stats", {})
    if qs:
        print(f"\n  QUICK STATS:")
        print(f"    Duration  : {qs.get('duration_formatted', '?')}")
        print(f"    Turns     : {qs.get('total_turns', '?')} total "
              f"({qs.get('salesperson_turns', '?')} sales / {qs.get('customer_turns', '?')} customer)")
        print(f"    Emotion   : final={qs.get('final_customer_emotion', '?')} "
              f"improved={qs.get('emotion_improved', '?')}")
        print(f"    Outcome   : {qs.get('call_outcome', '?')}")

    # Top strengths / improvements
    strengths = r.get("top_strengths", [])
    if strengths:
        print(f"\n  TOP STRENGTHS:")
        for s in strengths:
            print(f"    + {s}")

    improvements = r.get("top_improvements", [])
    if improvements:
        print(f"\n  TOP IMPROVEMENTS:")
        for i in improvements:
            print(f"    > {i}")

    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]
    session_id = args[0] if args else None
    mode       = args[1] if len(args) > 1 else "training"

    if mode not in ("training", "testing"):
        print(f"[ERROR] Invalid mode '{mode}'. Use 'training' or 'testing'.")
        sys.exit(1)

    # Auth
    print("[AUTH] Logging in...")
    token   = login()
    headers = {"Authorization": f"Bearer {token}"}

    # Resolve session
    if not session_id:
        session_id = get_most_recent_session(headers)

    print(f"[EVAL] Fetching report for session: {session_id} (mode={mode})")

    # Fetch report
    report = get_full_report(session_id, headers)
    status = report.get("status", "unknown")

    print(f"[EVAL] Report status: {status}")

    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    short_id   = session_id.replace("-", "")[:8]
    filename   = REPORTS_DIR / f"eval_{short_id}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {filename}")

    # Print summary
    if status == "completed":
        print_summary(report)
    elif status in ("pending", "processing"):
        print("[INFO] Evaluation is still running. Re-run this script to check later.")
    elif status == "not_started":
        print("[INFO] No evaluation exists for this session yet. Trigger one via the UI or API.")
    elif status == "failed":
        print(f"[FAILED] {report.get('error', 'Unknown error')}")
    else:
        print(f"[INFO] Status: {status}")


if __name__ == "__main__":
    main()
