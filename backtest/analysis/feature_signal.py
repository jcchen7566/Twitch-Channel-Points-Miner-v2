import math

from backtest.core.features import build_features


def _corr(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _auc_score(y_true, scores):
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _fit_logit(X, y, lr=0.05, steps=3000, l2=1e-3):
    m = len(X)
    d = len(X[0]) if X else 0
    w = [0.0] * d
    for _ in range(steps):
        grad = [0.0] * d
        for i in range(m):
            z = sum(w[j] * X[i][j] for j in range(d))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            e = p - y[i]
            for j in range(d):
                grad[j] += e * X[i][j]
        for j in range(d):
            grad[j] = (grad[j] / max(1, m)) + l2 * w[j]
            w[j] -= lr * grad[j]
    return w


def _predict_logit(X, w):
    out = []
    for row in X:
        z = sum(w[j] * row[j] for j in range(len(w)))
        z = max(-30.0, min(30.0, z))
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def analyze_feature_signal(events, streamer=None):
    pool = [e for e in events if e.label_known and (streamer is None or e.streamer == streamer)]
    if not pool:
        return {"samples": 0, "events": 0, "corr": {}, "auc": 0.5, "coeff": {}, "perm": {}}

    rows = []
    X = []
    y = []
    coef_names = ["bias", "d_log_odds", "d_ratio", "d_ev", "conf", "users_minus_points"]

    for e in pool:
        vals = []
        for o in e.outcomes:
            f = build_features(o, 0.5)
            vals.append((o, f))
            rows.append(
                {
                    "y": 1 if o.outcome_id == e.winner_outcome_id else 0,
                    "odds": o.odds,
                    "log_odds": math.log(max(1.01, o.odds)),
                    "users_pct": o.users_pct,
                    "points_pct": o.points_pct,
                    "ratio": f["ratio"],
                    "ev": f["ev"],
                    "p": f["p"],
                    "conf": f["conf"],
                    "ht": o.ht,
                    "hw": o.hw,
                }
            )

        mean_log_odds = sum(math.log(max(1.01, o.odds)) for o, _ in vals) / len(vals)
        mean_ratio = sum(f["ratio"] for _, f in vals) / len(vals)
        mean_ev = sum(f["ev"] for _, f in vals) / len(vals)

        for o, f in vals:
            X.append(
                [
                    1.0,
                    math.log(max(1.01, o.odds)) - mean_log_odds,
                    f["ratio"] - mean_ratio,
                    f["ev"] - mean_ev,
                    f["conf"],
                    o.users_pct - o.points_pct,
                ]
            )
            y.append(1 if o.outcome_id == e.winner_outcome_id else 0)

    corr_features = ["odds", "log_odds", "users_pct", "points_pct", "ratio", "ev", "p", "conf", "ht", "hw"]
    ys = [r["y"] for r in rows]
    corr = {}
    for k in corr_features:
        corr[k] = _corr([r[k] for r in rows], ys)

    w = _fit_logit(X, y)
    pred = _predict_logit(X, w)
    auc = _auc_score(y, pred)

    coeff = {coef_names[i]: w[i] for i in range(len(coef_names))}

    perm = {}
    base_auc = auc
    for j in range(1, len(coef_names)):
        Xp = [row[:] for row in X]
        col = [r[j] for r in Xp]
        if col:
            col = col[1:] + col[:1]
        for i in range(len(Xp)):
            Xp[i][j] = col[i]
        a = _auc_score(y, _predict_logit(Xp, w))
        perm[coef_names[j]] = base_auc - a

    return {
        "samples": len(rows),
        "events": len(pool),
        "corr": corr,
        "auc": auc,
        "coeff": coeff,
        "perm": perm,
    }


def print_feature_signal(report, label):
    print(f"feature_signal scope={label} events={report['events']} samples={report['samples']} auc={report['auc']:.4f}")
    print("feature_corr")
    for k, v in sorted(report["corr"].items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"{k},{v:+.4f}")
    print("logit_coeff")
    for k, v in report["coeff"].items():
        print(f"{k},{v:+.4f}")
    print("perm_auc_drop")
    for k, v in sorted(report["perm"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k},{v:+.4f}")
