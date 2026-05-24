import os
import subprocess
import tempfile
import sys
from pathlib import Path

# Ensure the project root is in sys.path for module imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.core.data_source import load_events_from_parser_csv
from backtest.analysis.feature_signal import analyze_feature_signal, print_feature_signal

def main():
    # Create a temporary CSV file for the parser output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
        csv_path = Path(tmp_csv.name)
    try:
        # Run the log_parser script to generate CSV output
        # Determine the path to the log_parser script relative to the project root
        parser_script = project_root / 'scripts' / 'log_parser.py'
        cmd = [
            'python3', str(parser_script),
            '--db', 'bet_logs.sqlite',
            '--resolved-only',
            '--output', str(csv_path),
            '--format', 'csv'
        ]
        subprocess.run(cmd, check=True, cwd=str(project_root))
        # Load events from the generated CSV
        events, _ = load_events_from_parser_csv(str(csv_path))
        # Global analysis
        print('=== Global Feature Signal Analysis ===')
        global_report = analyze_feature_signal(events)
        print_feature_signal(global_report, 'global')
        # Per-streamer analysis (top 5 streamers by event count)
        from collections import Counter
        streamer_counts = Counter(e.streamer for e in events if e.streamer)
        top_streamers = [s for s, _ in streamer_counts.most_common(5)]
        for streamer in top_streamers:
            print(f'=== Feature Signal Analysis for streamer: {streamer} ===')
            report = analyze_feature_signal(events, streamer=streamer)
            print_feature_signal(report, streamer)
    finally:
        # Clean up temporary CSV
        try:
            os.remove(csv_path)
        except OSError:
            pass

if __name__ == '__main__':
    main()
