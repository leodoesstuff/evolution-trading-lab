"""Small score reports: no scans over raw trade or equity archives."""
import json
import pandas as pd
import streamlit as st
from .day_rules import breeding_eligible


def load_metrics(store,run):
    metrics=store.rows('SELECT * FROM metrics WHERE run=? ORDER BY generation DESC,score DESC',(run,))
    reports=store.rows('SELECT bot,generation,payload FROM diagnostics WHERE run=?',(run,))
    if not reports.empty:
        extra=[]
        for row in reports.itertuples():
            r=json.loads(row.payload)
            extra.append(dict(bot=row.bot,generation=row.generation,rules_version=r.get('rules_version',1),
                              selection_eligible=r.get('selection_eligible',False),mean_exposure=r.get('mean_exposure'),
                              roundtrips=r.get('roundtrips'),active_days=r.get('active_days'),days=r.get('days'),
                              reason=r.get('eligibility_reason')))
        metrics=metrics.merge(pd.DataFrame(extra),on=['bot','generation'],how='left',sort=False)
    return metrics


def score_report(store,run,bot,mode,snapshot,config):
    a=snapshot.get('analytics',{})
    eq=snapshot['equity'];positions=sum(q*p for q,p in zip(snapshot['shares'],snapshot['prices']))
    cells=st.columns(4)
    cells[0].metric('Net profit',f'${eq-config.initial_cash:+,.2f}')
    cells[1].metric('Invested right now',f'{positions/max(eq,1e-8):.1%}')
    cells[2].metric('Average exposure',f'{a.get("exposure_sum",0)/max(a.get("bars",0),1):.1%}' if a else 'Not recorded')
    cells[3].metric('Completed round trips',str(a.get('roundtrips',0)) if a else 'Not recorded')
    st.caption('Exposure is the share of equity invested in stocks. A round trip ends when a stock position is fully sold. These are separate from the number of fills.')
    if a:
        days=a.get('days',{})
        st.write(f'Active days: **{sum(bool(d.get("active")) for d in days.values())} / {len(days)}** · Fees: **${a.get("fees",0):.2f}** · Modeled slippage: **${a.get("slippage",0):.2f}** · Daily loss stops: **{a.get("loss_stops",0)}**')
        st.caption('Activity is an eligibility requirement, not a bonus for making trades. Fees and slippage are already deducted from net profit.')
        if days:
            rows=[dict(Day=day,Profit=d['end']-d['start'],Return=d['end']/max(d['start'],1e-8)-1,Active=d.get('active',False),Stopped=d.get('halted',False)) for day,d in sorted(days.items())]
            with st.expander('Profit and risk by trading day'):st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    if ':g' in mode:
        generation=int(mode.split(':g')[1])
        report=store.rows('SELECT payload FROM diagnostics WHERE run=? AND bot=? AND generation=?',(run,bot,generation))
        if not report.empty:
            r=json.loads(report.iloc[0].payload)
            st.markdown('#### Evaluation score breakdown')
            parts=r.get('score_parts',{})
            rows=[dict(Component='Profit after costs',Contribution=parts.get('profit',0)),
                  dict(Component='Drawdown penalty',Contribution=-parts.get('drawdown_penalty',0)),
                  dict(Component='Daily inconsistency penalty',Contribution=-parts.get('consistency_penalty',0))]
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',column_config={'Contribution':st.column_config.NumberColumn(format='percent')})
            st.write(f'Combined score: **{r["score"]:.4f}** · {r.get("eligibility_reason","")}')
            warmup=generation<config.warmup_generations
            can_breed=breeding_eligible(r,config,generation)
            st.write(f'Positive-profit gate for qualification: **{"Met" if r["net_return"]>0 else "Not met"}** · Continued learning / reproduction: **{"Eligible" if can_breed else "Not eligible"}**')
            st.caption('This report combines all common practice windows; the portfolio above shows only the selected session. Active survivors within risk limits can continue learning even while unprofitable. The hall of fame and champion qualification require positive profit.' if config.training_selection=='active_survivors' else 'This saved experiment uses the original selection policy: after warmup, reproduction requires positive profit. Viewing a losing survivor does not change that policy.')
