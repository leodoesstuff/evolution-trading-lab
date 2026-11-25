import pandas as pd

from evo.config import Config
from evo.day_rules import eligible
from evo.engine import qualifies
from evo.leaders import best_bot, leader_counts, leader_wait_message


def records():
    rows=[
        dict(bot='old-profit',generation=3,alive=1,net_return=.5,score=.4,drawdown=.01,selection_eligible=True),
        dict(bot='inactive-profit',generation=4,alive=1,net_return=.1,score=.08,drawdown=.01,selection_eligible=False),
        dict(bot='eliminated-profit',generation=4,alive=0,net_return=.2,score=-1000,drawdown=.2,selection_eligible=False),
        dict(bot='smaller-loss',generation=4,alive=1,net_return=-.01,score=-.04,drawdown=.05,selection_eligible=True),
        dict(bot='steadier-loss',generation=4,alive=1,net_return=-.02,score=-.03,drawdown=.01,selection_eligible=True),
    ]
    return pd.DataFrame([dict(rules_version=2,**r) for r in rows])


def test_follow_losing_active_survivor_without_treating_it_as_qualified():
    metrics=records();before=metrics.copy(deep=True)
    leader=best_bot(metrics,{})
    assert leader['bot']=='smaller-loss' and not leader['profitable']
    assert leader['mode']=='evaluation:g4'
    assert best_bot(metrics,{},'Risk-adjusted score')['bot']=='steadier-loss'
    result=metrics[metrics.bot==leader['bot']].iloc[0].to_dict()
    assert not eligible(result,require_profit=True)
    assert not qualifies(result,.15,Config(rules_version=2))
    pd.testing.assert_frame_equal(metrics,before)


def test_counts_separate_survival_activity_and_profit():
    metrics=records()
    assert leader_counts(metrics)==dict(generation=4,completed=4,survived=3,eligible=2,profitable=0)
    metrics.loc[metrics.bot=='smaller-loss','net_return']=.01
    assert best_bot(metrics,{})['profitable']
    assert leader_counts(metrics)['profitable']==1


def test_no_eligible_survivor_reports_actual_worker_state():
    metrics=records();metrics.loc[metrics.generation==4,'selection_eligible']=False
    previous=best_bot(metrics,dict(generation=4))
    assert previous['bot']=='old-profit' and previous['using_previous']
    assert previous['mode']=='evaluation:g3'
    metrics['selection_eligible']=False
    assert best_bot(metrics,{}) is None
    assert 'Training is paused' in leader_wait_message(metrics,dict(phase='training',status='paused'),False)
    assert 'Training is running' in leader_wait_message(metrics,dict(phase='training',status='running'),True)
    assert 'experiment has ended' in leader_wait_message(metrics,dict(phase='finished',status='complete'),False)
    assert 'No completed evaluation' in leader_wait_message(pd.DataFrame(),dict(phase='training'),True)
