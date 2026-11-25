"""Small dashboard read models; never enumerate the raw trade/equity archive."""
import pandas as pd


def portfolio_choices(store, state, metrics):
    indexed = store.rows('SELECT bot,mode FROM portfolios WHERE run=? ORDER BY mode,bot', (state['run'],))
    if not indexed.empty:
        return indexed
    # Compatibility with old runs that predate saved portfolio snapshots. Their
    # small metrics table tells us which lives exist without scanning equity.
    pairs=[]
    for row in metrics.itertuples():
        pairs.append((row.bot,f'training:g{row.generation}'))
        if row.alive:pairs.append((row.bot,f'evaluation:g{row.generation}'))
    for row in state.get('validation',[]):pairs.append((row['bot'],'validation'))
    if state.get('test') and state.get('champion'):pairs.append((state['champion']['id'],'final_test'))
    if state.get('paper'):pairs.append((state['paper']['bot']['id'],'paper'))
    return pd.DataFrame(pairs,columns=['bot','mode']).drop_duplicates().sort_values(['mode','bot'])


def bot_directory(state, metrics, query='', page=1, size=25):
    latest = metrics.drop_duplicates('bot').set_index('bot').to_dict('index') if not metrics.empty else {}
    entries = {b['id']:dict(parent=b['parent'],birth=b['birth'],alive=b['alive']) for b in state['bots']}
    for bot,record in latest.items():entries.setdefault(bot,record)
    from .names import bot_name
    filtered=[(bot,info,latest.get(bot)) for bot,info in entries.items() if query.lower() in bot_name(bot).lower() or query.lower() in bot.lower()]
    total=len(filtered)
    max_page=max(1,(total+size-1)//size)
    page=min(max(1,page),max_page)
    return filtered[(page-1)*size:page*size],total,max_page
