"""
Clean up audio files from sessions older than N days.

A session's audio lives under data/audio_sessions/<session_id>/. This script
finds those folders whose corresponding DB session ended more than N days
ago and deletes them. The DB rows and Message.audio_path values are kept
(harmlessly broken) so the rest of the report stays intact.

Usage:
    # Dry run (default): list what WOULD be deleted, no changes
    python scripts/cleanup_old_audio.py

    # Use a custom retention window
    python scripts/cleanup_old_audio.py --days 14

    # Actually delete
    python scripts/cleanup_old_audio.py --apply

Cron example (Linux):
    0 3 * * *  cd /path/to/VCAI && python scripts/cleanup_old_audio.py --apply --days 30
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bootstrap project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import SessionLocal
from backend.models import Session as TrainingSession


AUDIO_ROOT = Path("data/audio_sessions")


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _folder_size(folder: Path) -> int:
    total = 0
    for f in folder.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def cleanup(days: int, apply: bool) -> None:
    if not AUDIO_ROOT.exists():
        print(f"[cleanup] {AUDIO_ROOT} does not exist — nothing to do.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"[cleanup] Looking for audio folders for sessions ended before {cutoff.isoformat()}")
    print(f"[cleanup] Audio root: {AUDIO_ROOT.resolve()}")
    print(f"[cleanup] Mode: {'APPLY (will delete)' if apply else 'DRY RUN (no changes)'}")
    print("-" * 70)

    db = SessionLocal()
    try:
        # Get every session whose audio folder is older than the cutoff.
        # Sessions that are active or recently ended are protected.
        candidates = (
            db.query(TrainingSession)
            .filter(
                TrainingSession.ended_at.isnot(None),
                TrainingSession.ended_at < cutoff,
            )
            .all()
        )
        print(f"[cleanup] Sessions older than {days} days with ended_at set: {len(candidates)}")

        deleted_count = 0
        freed_bytes = 0
        skipped_no_folder = 0

        for sess in candidates:
            folder = AUDIO_ROOT / str(sess.id)
            if not folder.exists():
                skipped_no_folder += 1
                continue

            size = _folder_size(folder)
            ended = sess.ended_at.isoformat() if sess.ended_at else "n/a"
            tag = "WOULD DELETE" if not apply else "DELETING"
            print(f"  {tag}  {folder.name}  ended={ended}  size={_human_size(size)}")

            if apply:
                try:
                    shutil.rmtree(folder)
                    deleted_count += 1
                    freed_bytes += size
                except Exception as e:
                    print(f"    FAILED: {e}")
            else:
                deleted_count += 1
                freed_bytes += size

        # Bonus: orphan folders (audio with no matching DB row) — also old.
        all_session_ids = {str(s.id) for s in db.query(TrainingSession.id).all()}
        orphans = [
            d for d in AUDIO_ROOT.iterdir()
            if d.is_dir() and d.name not in all_session_ids
        ]
        if orphans:
            print()
            print(f"[cleanup] Orphan folders (no DB row): {len(orphans)}")
            for d in orphans:
                size = _folder_size(d)
                tag = "WOULD DELETE" if not apply else "DELETING"
                print(f"  {tag}  {d.name}  (orphan)  size={_human_size(size)}")
                if apply:
                    try:
                        shutil.rmtree(d)
                        deleted_count += 1
                        freed_bytes += size
                    except Exception as e:
                        print(f"    FAILED: {e}")
                else:
                    deleted_count += 1
                    freed_bytes += size

        print("-" * 70)
        verb = "Deleted" if apply else "Would delete"
        print(f"[cleanup] {verb}: {deleted_count} folder(s) — frees {_human_size(freed_bytes)}")
        if skipped_no_folder:
            print(f"[cleanup] Skipped (no audio folder on disk): {skipped_no_folder}")
        if not apply:
            print("[cleanup] Re-run with --apply to actually delete.")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30,
                        help="Retention window in days (default: 30)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete the folders (default: dry run)")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be >= 1")

    cleanup(days=args.days, apply=args.apply)


if __name__ == "__main__":
    main()
