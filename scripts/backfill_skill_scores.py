"""One-time backfill: write eval skill scores to session table."""
import sys
sys.path.insert(0, "C:/VCAI")

from backend.database import get_db_context
from backend.models.session import Session as TrainingSession
from backend.models.evaluation import EvaluationReport

EVAL_TO_SESSION = {
    "communication_clarity": "communication_score",
    "product_knowledge":     "product_knowledge_score",
    "objection_handling":    "objection_handling_score",
    "rapport_building":      "rapport_score",
    "closing_skills":        "closing_score",
}

with get_db_context() as db:
    reports = db.query(EvaluationReport).filter(
        EvaluationReport.status == "completed",
        EvaluationReport.report_json.isnot(None),
    ).all()
    updated = 0
    for r in reports:
        s = db.query(TrainingSession).filter(TrainingSession.id == r.session_id).first()
        if not s:
            continue
        skill_rows = {
            sk.get("skill_key"): sk.get("score")
            for sk in r.report_json.get("scores", {}).get("skills", [])
            if sk.get("skill_key") and sk.get("score") is not None
        }
        changed = False
        for eval_key, col in EVAL_TO_SESSION.items():
            if eval_key in skill_rows:
                setattr(s, col, skill_rows[eval_key])
                changed = True
        if changed:
            updated += 1
    db.commit()
    print(f"Backfilled {updated} / {len(reports)} sessions")
