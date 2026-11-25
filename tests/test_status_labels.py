import copy
import numpy as np
import pytest
from evo.status_labels import bot_status,qualified_champion,practice_status
from evo.sim import Book
from evo.config import Config


def state(**kwargs):
    s=dict(status='running',phase='training',bots=[{'id':'a'},{'id':'b'},{'id':'c'}],cursor=1,generation=3)
    s.update(kwargs);return s


def test_working_bot_is_distinct_from_survivors_and_queued_bots():
    s=state();record=dict(alive=True)
    assert bot_status(s,'a',record,True)['label']=='Survived'
    assert bot_status(s,'b',record,True)['label']=='Training'
    assert bot_status(s,'c',None,True)['label']=='Waiting'
    assert bot_status(s,'b',record,False)['label']=='Paused'
    assert bot_status(state(status='paused'),'b',record,True)['label']=='Paused'
    assert bot_status(state(status='ready'),'b',None,False)['label']=='Waiting'
    assert bot_status(s,'old',dict(alive=False),True)['label']=='Eliminated'


def test_champion_requires_finished_qualification_not_just_selection():
    s=state(status='complete',phase='finished',champion={'id':'a'},promotion_reason='Champion failed validation.')
    assert qualified_champion(s) is None
    assert bot_status(s,'a',dict(alive=True))['label']=='Survived'
    s.update(phase='paper',status='paper_waiting',promotion_reason='Passed validation and untouched final test.',paper={'message':'Market closed.'})
    assert qualified_champion(s)=='a'
    assert bot_status(s,'a',dict(alive=False))['label']=='Champion'  # Winning snapshot may predate another life of this lineage.
    other=bot_status(s,'b',dict(alive=True))
    assert other['label']=='Survived' and 'training is finished' in other['detail']


def test_testing_and_inactivity_are_not_champion_or_elimination():
    s=state(phase='validation',hall=[{'bot':{'id':'b'}}],validation_cursor=0)
    assert bot_status(s,'b',dict(alive=True),True)['label']=='Testing'
    assert bot_status(s,'b',dict(alive=True),False)['label']=='Paused'
    inactive=dict(alive=True,rules_version=2,selection_eligible=np.bool_(False))
    assert bot_status(state(phase='finished',status='complete'),'c',inactive)['label']=='Inactive'


def test_display_status_does_not_modify_checkpoint_or_records():
    s=state();record=dict(alive=True,rules_version=2,selection_eligible=True)
    before=copy.deepcopy(s);before_record=record.copy()
    bot_status(s,'b',record,True)
    assert s==before and record==before_record


def test_practice_labels_follow_session_state_not_stale_active_field():
    p={'book':Book.fresh(Config()),'status':'active'}
    s={'status':'paused','market_open':False,'connection':'market closed','learning':True}
    assert practice_status(s,p,False)=='Paused'
    s['status']='waiting'
    assert practice_status(s,p,True)=='Waiting'
    s.update(status='running',market_open=True,connection='connected')
    assert practice_status(s,p,True)=='Learning live'
    s['learning']=False
    assert practice_status(s,p,True)=='Practicing live'
    s['status']='stopped'
    assert practice_status(s,p,False)=='Stopped'
    p['book'].alive=False
    assert practice_status(s,p,False)=='Eliminated'
