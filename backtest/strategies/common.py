from ..core.features import build_features


def best_by_ev(event, state):
    best = None
    best_ev = -10**9
    for o in event.outcomes:
        # Passing event.outcomes for event-relative normalization (Z-scores)
        feat = build_features(o, state.prior_win_rate(event.streamer, o.odds), event_outcomes=event.outcomes)
        if feat["ev"] > best_ev:
            best_ev = feat["ev"]
            best = o
    return best
