import argparse
import csv
import json
import sqlite3


BASE_QUERY = """
SELECT
  e.created_at,
  e.event_id,
  e.streamer,
  e.main_title,
  e.placed_amount,
  e.selected_outcome_id,
  e.winning_outcome_id,
  e.result_type,
  e.points_gained,
  o.outcome_idx,
  o.outcome_id,
  o.outcome_title,
  o.outcome_users,
  o.outcome_points,
  o.outcome_odds,
  o.outcome_users_percent,
  o.outcome_points_percent,
  o.history_wins_weighted,
  o.history_total_weighted,
  o.history_wins_raw,
  o.history_total_raw,
  o.total_users,
  o.total_points
FROM bet_events e
JOIN bet_outcomes o ON o.event_id = e.event_id
"""


def build_where(args):
    clauses = []
    params = []
    if args.streamer:
        clauses.append("e.streamer = ?")
        params.append(args.streamer)
    if args.date_from:
        clauses.append("substr(e.created_at, 1, 10) >= ?")
        params.append(args.date_from)
    if args.date_to:
        clauses.append("substr(e.created_at, 1, 10) <= ?")
        params.append(args.date_to)
    if args.resolved_only:
        clauses.append("e.result_type IN ('WIN','LOSE','REFUND')")
    if clauses:
        return " WHERE " + " AND ".join(clauses), params
    return "", params


def _drop_non_output_fields(rows):
    for r in rows:
        r.pop("result_type", None)


def fetch_rows(conn, args):
    where, params = build_where(args)
    q = BASE_QUERY + where + " ORDER BY e.created_at, e.event_id, o.outcome_idx"
    cur = conn.cursor()
    cols = [c[0] for c in cur.execute(q, params).description]
    rows = [dict(zip(cols, r)) for r in cur.execute(q, params).fetchall()]
    _drop_non_output_fields(rows)
    return rows


def output_rows(rows, args):
    if args.format == "jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif args.format == "json":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        if not rows:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                pass
            return
        with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_basic_stats(rows):
    if not rows:
        print("No rows.")
        return

    event_ids = sorted({r["event_id"] for r in rows})
    selected_rows = [r for r in rows if r["outcome_id"] == r["selected_outcome_id"]]
    winner_known = sum(1 for r in selected_rows if str(r.get("winning_outcome_id") or "").strip())

    print(f"rows={len(rows)}")
    print(f"events={len(event_ids)}")
    print(f"selected_events={len(selected_rows)} winner_known={winner_known} winner_unknown={len(selected_rows)-winner_known}")


def main():
    parser = argparse.ArgumentParser(
        description="Parser for bet_logs.sqlite using raw winning_outcome_id only (no inference)."
    )
    parser.add_argument("--db", default="bet_logs.sqlite")
    parser.add_argument("--streamer", default=None)
    parser.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--resolved-only", action="store_true")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--format", choices=["csv", "json", "jsonl"], default="csv")
    parser.add_argument("--stats", action="store_true", help="Print aggregate winner-id coverage")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    rows = fetch_rows(conn, args)
    conn.close()

    if args.output:
        output_rows(rows, args)
        print(f"wrote_rows={len(rows)} output={args.output} format={args.format}")
    elif not args.stats:
        for r in rows[:5]:
            print(json.dumps(r, ensure_ascii=False))
        print(f"preview_rows=5/{len(rows)}")

    if args.stats:
        print_basic_stats(rows)


if __name__ == "__main__":
    main()
