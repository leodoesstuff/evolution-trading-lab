"""Shared strategy decision timing; valuation and risk checks remain continuous."""
import pandas as pd


def decision_due(previous, timestamp, config):
    if previous is None:
        return True
    now, last = pd.Timestamp(timestamp), pd.Timestamp(previous)
    if now <= last:
        return False
    if now.tz_convert('America/New_York').date() != last.tz_convert('America/New_York').date():
        return True
    return (now-last).total_seconds() >= getattr(config, 'decision_bars', 1)*300


def transition_bars(previous, timestamp, config):
    # Overnight time does not represent additional tradable five-minute steps.
    return min(getattr(config, 'decision_bars', 1),
               max(1, int((pd.Timestamp(timestamp)-pd.Timestamp(previous)).total_seconds()/300)))
