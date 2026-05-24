import csv
import subprocess
from collections import defaultdict

from .types import Event, OutcomeRow


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def export_parser_csv(db_path: str, parser_script: str, out_csv: str) -> None:
    cmd = [
        "python3",
        parser_script,
        "--db",
        db_path,
        "--resolved-only",
        "--output",
        out_csv,
        "--format",
        "csv",
    ]
    subprocess.run(cmd, check=True)


def load_events_from_parser_csv(path: str) -> tuple[list[Event], dict[str, int]]:
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            event_id = str(r.get("event_id") or "")
            if not event_id:
                continue

            outcome_id = str(r.get("outcome_id") or "")
            odds = max(1.01, _f(r.get("outcome_odds"), 1.01))
            row = OutcomeRow(
                timestamp=r.get("created_at", ""),
                event_id=event_id,
                streamer=r.get("streamer", ""),
                main_title=r.get("main_title", ""),
                outcome_id=outcome_id,
                odds=odds,
                users_pct=_f(r.get("outcome_users_percent"), 0.0),
                points_pct=_f(r.get("outcome_points_percent"), 0.0),
                users=_f(r.get("outcome_users"), 0.0),
                points=_f(r.get("outcome_points"), 0.0),
                total_users=_f(r.get("total_users"), 0.0),
                total_points=_f(r.get("total_points"), 0.0),
                hw=_f(r.get("history_wins_weighted"), 0.0),
                ht=_f(r.get("history_total_weighted"), 0.0),
                rw=_f(r.get("history_wins_raw"), 0.0),
                rt=_f(r.get("history_total_raw"), 0.0),
            )
            grouped[event_id].append(
                (
                    row,
                    str(r.get("selected_outcome_id") or ""),
                    str(r.get("winning_outcome_id") or "").strip(),
                )
            )

    events = []
    stats = {
        "events_total": 0,
        "events_label_known": 0,
        "events_label_unknown": 0,
        "events_inferred_from_win": 0,
        "events_inferred_from_binary_lose": 0,
    }

    for event_id, pairs in grouped.items():
        outcomes = [p[0] for p in pairs]
        selected_outcome_id = pairs[0][1]
        winner_outcome_id = pairs[0][2]
        label_known = bool(winner_outcome_id)

        stats["events_total"] += 1
        if label_known:
            stats["events_label_known"] += 1
        else:
            stats["events_label_unknown"] += 1

        events.append(
            Event(
                timestamp=outcomes[0].timestamp,
                event_id=event_id,
                streamer=outcomes[0].streamer,
                main_title=outcomes[0].main_title,
                outcomes=outcomes,
                winner_outcome_id=winner_outcome_id,
                selected_outcome_id=selected_outcome_id,
                result_type="",
                label_known=label_known,
            )
        )

    events.sort(key=lambda e: (e.timestamp, e.event_id))
    return events, stats
