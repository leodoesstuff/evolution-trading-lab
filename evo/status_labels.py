"""Read-only display statuses. Risk survival is distinct from winning or working."""


def qualified_champion(state):
    champion=state.get('champion')
    if champion and state.get('promotion_reason')=='Passed validation and untouched final test.':
        return champion['id']
    return None


def bot_status(state, bot, record=None, worker_running=False):
    winner=qualified_champion(state)
    if winner==bot:
        paper=state.get('paper')
        detail='Won qualification after validation and the final test.'
        if paper:
            detail+=' '+paper.get('message','')
        return dict(label='Champion',detail=detail)
    phase=state.get('phase');status=state.get('status')
    current=[b['id'] for b in state.get('bots',[])]
    if phase=='training' and bot in current and status not in ('complete','stopped'):
        index=current.index(bot);cursor=state.get('cursor',0)
        if index>=cursor:
            if status in ('paused','error') or status=='running' and not worker_running:
                return dict(label='Paused',detail='This bot is waiting for training to resume; saved progress is retained.')
            if index==cursor and worker_running and status=='running':
                return dict(label='Training',detail='Learning or completing its current practice tests.')
            return dict(label='Waiting',detail='Queued for evaluation in this generation.')
    testing=None
    if phase=='validation':
        hall=state.get('hall',[]);i=state.get('validation_cursor',0)
        if i<len(hall):testing=hall[i]['bot']['id']
    elif phase=='test' and state.get('champion'):testing=state['champion']['id']
    if testing==bot:
        return dict(label='Testing' if worker_running and status=='running' else 'Paused',
                    detail='Validation or final testing is not finished; this bot is not yet a champion.')
    if record is not None:
        if not bool(record['alive']):
            return dict(label='Eliminated',detail='Failed the risk or exit rules in its latest evaluated life. Its history remains saved.')
        if record.get('rules_version',1)>=2 and not bool(record.get('selection_eligible',True)):
            return dict(label='Inactive',detail='Survived the risk limit but did not meet the trading-activity requirements.')
        if winner:
            return dict(label='Survived',detail='Survived evaluation; training is finished and this bot was not selected as champion.')
        if phase=='finished' or status in ('complete','stopped'):
            return dict(label='Survived',detail='Saved survivor. This experiment has ended; this bot is not still training.')
        return dict(label='Survived',detail='Completed its latest practice test within the risk limit. This does not mean it won.')
    return dict(label='Saved' if status in ('complete','stopped') else 'Waiting',
                detail='No completed evaluation is recorded for this snapshot.')


def practice_status(state, participant, worker_running=False):
    if not participant['book'].alive:
        return 'Eliminated'
    if state['status']=='stopped':return 'Stopped'
    if state['status']=='finished':return 'Finished'
    if state['status'] in ('paused','error') or state['status'] in ('running','waiting') and not worker_running:return 'Paused'
    if not state.get('market_open') or state.get('connection')!='connected':return 'Waiting'
    return 'Learning live' if state.get('learning') else 'Practicing live'
