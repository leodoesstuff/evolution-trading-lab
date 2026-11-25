"""Explicit history coverage for new experiments and preserved-bot continuations."""
import copy
from dataclasses import replace

from .config import Config
from .engine import new_run, write_report
from .store import Store


# Approximate full sessions at 78 five-minute bars per regular trading day.
COVERAGE = {
    'Quick': (10, 5, 3),
    'Broad': (60, 20, 6),
    'Deep': (120, 40, 8),
}


def coverage_config(config, name):
    learning, evaluation, windows = COVERAGE[name]
    return replace(config, train_steps=learning * 78, eval_steps=evaluation * 78,
                   eval_windows=windows)


def continue_with_coverage(home, source_run, name='Broad', decision_bars=None):
    """Keep learned weights and optimizer state; restart comparisons in a new run."""
    store = Store(home)
    source = store.load(source_run)
    if source['phase'] != 'training':
        raise ValueError('Continue a training experiment; completed results remain archived.')
    config = coverage_config(Config(**source['config']), name)
    if decision_bars is not None:
        config = replace(config, decision_bars=decision_bars, stop_on_extinction=True)
    if config.train_steps <= source['config']['train_steps'] and config.decision_bars <= source['config'].get('decision_bars',1):
        raise ValueError('Choose a longer learning period or a slower decision interval than the source experiment.')
    run = new_run(home, source['dataset'], config)
    state = store.load(run)
    state.update(bots=copy.deepcopy(source['bots']), rng=copy.deepcopy(source['rng']),
                 generation=source['generation'],
                 continuation=dict(source_run=source_run,
                                   source_generation=source['generation'],
                                   source_cursor=source['cursor'], coverage=name,
                                   decision_bars=config.decision_bars))
    # The hall and scores from shorter windows are deliberately not compared to new scores.
    store.save(run, state)
    write_report(store, state)
    store.event(run, 'continued_with_more_history', **state['continuation'])
    return run
