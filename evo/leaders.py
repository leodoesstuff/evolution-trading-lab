"""Read-only leader selection for live dashboard views."""

LEADER_VIEW_VERSION = 3


def candidate_pool(metrics):
    """Comparable, active survivors; profit is reported separately for viewing."""
    latest = metrics[metrics.generation == metrics.generation.max()] if not metrics.empty else metrics
    if latest.empty:
        return latest, latest
    candidates = latest[latest.alive == 1]
    if 'selection_eligible' in candidates:
        modern=candidates['rules_version'].fillna(1)>=2
        candidates=candidates[~modern | candidates['selection_eligible'].fillna(False)]
    return latest, candidates


def leader_counts(metrics):
    latest, candidates = candidate_pool(metrics)
    return dict(generation=int(latest.generation.max()) if not latest.empty else None,
                completed=len(latest),survived=int((latest.alive==1).sum()) if not latest.empty else 0,
                eligible=len(candidates),profitable=int((candidates.net_return>0).sum()) if not candidates.empty else 0)


def leader_wait_message(metrics, state, worker_running=False):
    if state.get('paper') and not state['paper']['book'].alive:
        return 'The live paper bot was eliminated. Turn off Follow best bot to inspect its saved history.'
    counts=leader_counts(metrics)
    message=('No completed evaluation yet.' if not counts['completed'] else
             'No surviving bot in the latest evaluated generation meets the activity and risk rules.')
    if state.get('phase')=='finished' or state.get('status') in ('complete','stopped'):
        return message+' This experiment has ended; saved results remain available.'
    if not worker_running:
        return message+' Training is paused; refreshing the view does not advance training.'
    return message+' Training is running. New completed evaluations appear here automatically.'

def best_bot(metrics, state, ranking='Highest return'):
    """Choose among comparable surviving tests; never use a future/test result to train."""
    if state.get('paper'):
        paper = state['paper']
        if not paper['book'].alive:
            return None
        return dict(bot=paper['bot']['id'], mode='paper', reason='Live paper portfolio')
    if metrics.empty:
        return None
    latest, candidates = candidate_pool(metrics)
    generation = int(latest.generation.max())
    if candidates.empty:
        for previous, group in metrics[metrics.generation<generation].groupby('generation',sort=True):
            _, saved = candidate_pool(group)
            if not saved.empty:
                candidates=saved
        if candidates.empty:
            return None
        generation=int(candidates.generation.max())
    using_previous=generation<max(int(latest.generation.max()),state.get('generation',generation))
    primary = 'net_return' if ranking == 'Highest return' else 'score'
    leader = candidates.sort_values([primary,'drawdown','bot'],ascending=[False,True,True]).iloc[0]
    return dict(bot=leader.bot, mode=f'evaluation:g{generation}', generation=generation,
                net_return=float(leader.net_return), score=float(leader.score),
                profitable=bool(leader.net_return>0),
                using_previous=using_previous,
                reason=f'{"Saved" if using_previous else "Best"} active survivor · generation {generation+1}')

