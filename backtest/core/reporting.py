import csv

def _slice_roi(s):
    return (s.gain / s.bet) if s.bet > 0 else 0.0

def _slice_wr(s):
    return (s.wins / (s.wins + s.losses)) if (s.wins + s.losses) > 0 else 0.0

def print_per_streamer_leaderboards(results):
    """
    Groups results by streamer and prints a leaderboard for each.
    This is the primary output as global stats are irrelevant.
    """
    streamers = set()
    for r in results:
        streamers.update(r.by_streamer.keys())
    
    # Sort streamers by total bets to show most active ones first
    streamer_activity = []
    for s in streamers:
        # Find any result to get the bet count for this streamer
        bets = 0
        for r in results:
            if s in r.by_streamer:
                bets = r.by_streamer[s].bets
                break
        streamer_activity.append((s, bets))
    
    streamer_activity.sort(key=lambda x: x[1], reverse=True)
    
    for streamer, total_bets in streamer_activity:
        print(f"\n>>> STREAMER: {streamer:<15} (Total Events: {total_bets})")
        print(f"{'RANK':<4} | {'STRATEGY':<30} | {'ROI':<8} | {'GAIN':<12} | {'WR':<6} | {'AVG_BET':<8}")
        print("-" * 85)
        
        strat_stats = []
        for r in results:
            if streamer in r.by_streamer:
                s = r.by_streamer[streamer]
                strat_stats.append((r.name, _slice_roi(s), s.gain, _slice_wr(s), (s.bet/s.bets if s.bets > 0 else 0)))
        
        # Rank by ROI for this specific streamer
        strat_stats.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, roi, gain, wr, avg_bet) in enumerate(strat_stats):
            print(f"{i+1:<4} | {name:<30} | {roi:<8.4f} | {gain:<12.0f} | {wr:<6.4f} | {avg_bet:<8.0f}")

def print_walkforward_summary(name: str, folds: list):
    # For walk-forward, we still show per-fold but focused on localized metrics if possible.
    # But usually WF is a summary of the strategy's robustness. 
    # To keep with the "per-streamer only" rule, we'll keep this minimal or omit global averages.
    if not folds:
        return
    total_gain = sum(f.total_gain for f in folds)
    total_bets = sum(f.bets for f in folds)
    # print(f"WF-DEBUG | {name:<25} | bets={total_bets:<4} | gain={total_gain:>10.0f}")
    pass
